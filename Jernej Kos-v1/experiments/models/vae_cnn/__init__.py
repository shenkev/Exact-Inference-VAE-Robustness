from collections import namedtuple

import tensorflow as tf
import tensorflow.contrib.slim as slim

from ..vae_gan import layers
from ... import model


class Model(model.GenerativeModelBase):
    """VAE-CNN model."""
    name = 'vae-cnn'
    batch_size = 128

    # Results of internal model operations.
    model_type = namedtuple('VAECNN', [
        'width',
        'height',
        'x',
        'z_x_mean',
        'z_x_log_sigma',
        'z_x',
        'x_tilde',
        'x_tilde_mean',
    ])

    @property
    def output_dimensions(self):
        return layers.get_dimensions(self.width, self.height)

    def _build(self, x, sample=1):
        """Builds the model."""

        # Reshape input as needed.
        x, width, height = layers.pad_power2(x, self.width, self.height, self.channels)
        # Normal distribution for VAE sampling.
        eps = tf.random_normal([self.batch_size, self.latent_dim], mean=0, stddev=0.01)

        with slim.arg_scope([layers.encoder, layers.decoder],
                            width=width,
                            height=height,
                            channels=self.channels,
                            latent_dim=self.latent_dim,
                            is_training=self._training):
            # Get latent representation for sampling.
            z_x_mean, z_x_log_sigma = layers.encoder(x)
            # Sample from latent space.
            z_x = []
            for _ in xrange(sample):
                z_x.append(z_x_mean + tf.exp(z_x_log_sigma / 2) * eps)
            if sample > 1:
                z_x = tf.add_n(z_x) / sample
            else:
                z_x = z_x[0]
            # Generate output.
            x_tilde = layers.decoder(z_x)

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # Generate reconstruction.
                x_tilde_mean = layers.decoder(z_x_mean)

            return self.model_type(
                width=width,
                height=height,
                x=x,
                z_x_mean=z_x_mean,
                z_x_log_sigma=z_x_log_sigma,
                z_x=z_x,
                x_tilde=x_tilde,
                x_tilde_mean=x_tilde_mean,
            )

    def _build_optimizer(self):
        return None, tf.train.AdamOptimizer(learning_rate=1e-3)

    def _build_loss(self, model, labels):
        """Loss function."""

        # Transform back to logits as TF expects logits not probabilities.
        epsilon = 10e-8
        x_tilde = tf.clip_by_value(model.x_tilde, epsilon, 1 - epsilon)
        x_tilde = tf.log(x_tilde / (1 - x_tilde))

        # Reconstruction loss.
        R_loss = model.width * model.height * self.channels * tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=slim.flatten(x_tilde), labels=slim.flatten(model.x)),
            axis=-1
        )

        # KL divergence.
        KL_loss = -0.5 * tf.reduce_sum(
            1 + model.z_x_log_sigma - tf.square(model.z_x_mean) - tf.exp(model.z_x_log_sigma),
            axis=-1
        )

        return R_loss + KL_loss

    def _build_gradients(self, optimizer, gradients, loss):
        gradients.setdefault('vae', []).append(
            optimizer.compute_gradients(loss, var_list=self._get_model_variables())
        )

    def _build_apply_gradients(self, optimizer, gradients, global_step):
        return optimizer.apply_gradients(gradients['vae'], global_step=global_step)

    def encode_op(self, x, sample=False, with_variance=False):
        model = self._model(x)
        if sample:
            return model.z_x
        elif with_variance:
            return model.z_x_mean, model.z_x_log_sigma_sq
        else:
            return model.z_x_mean

    def decode_op(self, z):
        # Compute output dimensions.
        width, height = layers.get_dimensions(self.width, self.height)

        with tf.variable_scope(self._model.var_scope, reuse=True):
            x_tilde = layers.decoder(z,
                                     width=width,
                                     height=height,
                                     channels=self.channels,
                                     latent_dim=self.latent_dim,
                                     is_training=self._training)
            return x_tilde

    def reconstruct_op(self, x, sample=False, sample_times=1):
        model = self._model(x, sample=sample_times)
        if sample:
            return model.x_tilde
        else:
            return model.x_tilde_mean

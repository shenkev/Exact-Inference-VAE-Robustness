from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from . import layers
from ... import model, utils


log2pi = tf.log(2.0 * np.pi)

class Model(model.GenerativeModelBase):
    """VAE-GAN model."""
    name = 'vae-gan'

    # Results of internal model operations.
    model_type = namedtuple('VAEGAN', [
        'width',
        'height',
        'z_x_mean',
        'z_x_log_sigma_sq',
        'z_x',
        'x_tilde',
        'x_tilde_mean',
        'l_x_tilde',
        'd_x',
        'l_x',
        'd_x_p',
        'objective',
    ])

    @property
    def output_dimensions(self):
        return layers.get_dimensions(self.width, self.height)

    def _build(self, x, sample=30):
        """Builds the model."""
        print("WHAT THE FUCKK")
        num_IW_samples = sample

        # Reshape input as needed.
        x, width, height = layers.pad_power2(x, self.width, self.height, self.channels)
        x_IW = tf.tile(x,tf.constant([num_IW_samples,1]))

        # Normal distribution for GAN sampling.
        z_p = tf.random_normal([self.batch_size, self.latent_dim], 0, 1)
        z_p_IW = tf.random_normal([self.batch_size* num_IW_samples,self.latent_dim], 0, 1)

        # Normal distribution for VAE sampling.
        # eps = tf.random_normal([self.batch_size, self.latent_dim], 0, 1)
        eps_IW = tf.random_normal([self.batch_size* num_IW_samples,self.latent_dim], 0, 1)

        with slim.arg_scope([layers.encoder, layers.decoder, layers.discriminator],
                            width=width,
                            height=height,
                            channels=self.channels,
                            latent_dim=self.latent_dim,
                            is_training=self._training):
            # Get latent representation for sampling.
            z_x_mean, z_x_log_sigma_sq = layers.encoder(x)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                z_x_mean_IW, z_x_log_sigma_sq_IW = layers.encoder(x_IW)
            # z_x_mean_IW = tf.reshape(z_x_mean_IW,[self.batch_size, num_IW_samples,self.latent_dim])
            # z_x_log_sigma_sq_IW = tf.reshape(z_x_log_sigma_sq_IW,[self.batch_size, num_IW_samples,self.latent_dim])

            std_dev = tf.sqrt(tf.exp(z_x_log_sigma_sq_IW))
            z_x = z_x_mean_IW + eps_IW*std_dev
            print("THIS IS Z_X!")
            print(z_x)

            # Sample from latent space.
            # z_x = []
            # for _ in xrange(sample):
            #     z_x.append(tf.add(z_x_mean_IW, tf.multiply(tf.sqrt(tf.exp(z_x_log_sigma_sq_IW)), eps_IW)))
            # if sample > 1:
            #     z_x = tf.add_n(z_x) / sample
            # else:
            #     z_x = z_x[0]

            tol = 1e-5 # epsilon?
            def gaussian_likelihood(data, mean, log_variance):
                """Log-likelihood of data given ~ N(mean, exp(log_variance))

                Parameters
                ----------
                data :
                    Samples from Gaussian centered at mean
                mean :
                    Mean of the Gaussian distribution
                log_variance :
                    Log variance of the Gaussian distribution

                Returns
                -------
                log_likelihood : float

                """
                # print data.get_shape().as_list()
                num_components = data.get_shape().as_list()[1]
                variance = tf.exp(log_variance)
                log_likelihood = (
                    -(log2pi * (num_components / 2.0))
                    - tf.reduce_sum(
                        (tf.square(data - mean) / (2 * variance)) + (log_variance / 2.0),
                        1)
                )

                return log_likelihood

            def standard_gaussian_likelihood(data):
                """Log-likelihood of data given ~ N(0, 1)
                Parameters
                ----------
                data :
                    Samples from Guassian centered at 0
                Returns
                -------
                log_likelihood : float
                """

                num_components = data.get_shape().as_list()[1]
                log_likelihood = (
                    -(log2pi * (num_components / 2.0))
                    - tf.reduce_sum(tf.square(data) / 2.0, 1)
                )

                return log_likelihood

            print("LETS SEE HARRIS")
            log_q_z = gaussian_likelihood(z_x,z_x_mean_IW,z_x_log_sigma_sq_IW)
            # log_q_z = tf.reshape(log_q_z,[self.batch_size,num_IW_samples])
            print(log_q_z)

            log_p_z = standard_gaussian_likelihood(z_x)

            mean_decoder = layers.decoder(z_x)
            # Bernoulli log-likelihood reconstruction
            # TODO: other distributon types
            def bernoulli_log_joint(x):
                return tf.reduce_sum(
                    (x * tf.log(tol + mean_decoder))
                        + ((1 - x) * tf.log(tol + 1 - mean_decoder)),
                    1)

            # log_p_given_z = bernoulli_log_joint(x_IW)
            # HC: Use Lth layer loss for reconstruction rather than bernoulli 
            d_x, l_x_IW = layers.discriminator(x_IW)
            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                x_tilde_IW = layers.decoder(z_x)
                _, l_x_tilde_IW = layers.discriminator(x_tilde_IW)

            print("Shape of l_x_IW")
            print(l_x_IW)
            # log_p_given_z = -tf.reduce_sum(tf.square(l_x_IW - l_x_tilde_IW), axis=1) / width / height / self.channels
            # HC: Just use Gaussian on the Lth Layer space with Identity Variance
            log_p_given_z = gaussian_likelihood(l_x_IW, l_x_tilde_IW, 0.0) # Variance is set to Identity so Log Var = 0

            print("Shape of Z_X")
            print(z_x)
            print("Shape of lop_p_given_z")
            print(log_p_given_z)

            # Compute the log of the importance weights
            log_weights = log_p_given_z + log_p_z - log_q_z
            log_weights = tf.reshape(log_weights, [self.batch_size,num_IW_samples])
            print("log_weights")

            # Use the log-sum-exp trick to compute the total weights
            log_weights_max =  tf.reduce_max(log_weights, 1, keep_dims=True)

            weights_iw =  tf.log(tf.reduce_mean(tf.exp(log_weights - log_weights_max), 1))
            objective = tf.reduce_mean(log_weights_max) + tf.reduce_mean(weights_iw)


            print("OBJECTIVE!")
            print(objective)

            log_weights_total = log_weights_max + tf.log(tf.reduce_sum(tf.exp(log_weights - log_weights_max), 1, keep_dims=True))
            log_weights_norm = log_weights - log_weights_total

            # Use categorical to sample which sample to use
            dist = tf.distributions.Categorical(log_weights_norm)
            samples = tf.reshape(dist.sample(1), [self.batch_size,1])
            b_indices = tf.expand_dims(tf.range(self.batch_size),1)
            gather_id = tf.concat([samples, b_indices], axis=1)
            print("DONE GATHERING ID")
            z_x = tf.gather_nd(tf.reshape(z_x, [num_IW_samples, self.batch_size, -1]), gather_id)
            print("Resampled Z_X:")
            print(z_x)
            print(z_x.shape)

            with tf.variable_scope(tf.get_variable_scope(), reuse=True):
                # Generate output from resampled z's.
                x_tilde = layers.decoder(z_x)

                print("shape test")
                print(x)
                print(x_tilde)

                _, l_x_tilde = layers.discriminator(x_tilde)

                # Generate reconstruction.
                x_tilde_mean = layers.decoder(z_x_mean_IW)
                print(x_tilde_mean)

                # Generate a new image.
                x_p = layers.decoder(z_p)
                # Run discriminator on original inputs.
                d_x, l_x = layers.discriminator(x)
                # Run discriminator on generated outputs.
                d_x_p, _ = layers.discriminator(x_p)

            return self.model_type(
                width=width,
                height=height,
                z_x_mean=z_x_mean, # Note: This seems to be used to compute the KL loss
                z_x_log_sigma_sq=z_x_log_sigma_sq,
                z_x=z_x,
                x_tilde=x_tilde,
                x_tilde_mean=x_tilde_mean,
                l_x=l_x,
                l_x_tilde=l_x_tilde,
                d_x=d_x,
                d_x_p=d_x_p,
                objective = objective,
            )

    def _build_optimizer(self):
        """Different optimizers are needed for different learning rates."""
        lr_encoder = tf.placeholder(tf.float32, shape=[])
        lr_decoder = tf.placeholder(tf.float32, shape=[])
        lr_discriminator = tf.placeholder(tf.float32, shape=[])
        return (
            [lr_encoder, lr_decoder, lr_discriminator],
            [
                tf.train.AdamOptimizer(lr_encoder, epsilon=1.0),
                tf.train.AdamOptimizer(lr_decoder, epsilon=1.0),
                tf.train.AdamOptimizer(lr_discriminator, epsilon=1.0)
            ]
        )

    def _build_loss(self, model, labels):
        """Loss functions for KL divergence, Discrim, Generator, Lth Layer Similarity."""

        # We clip gradients of KL divergence to prevent NANs.
        KL_loss = tf.reduce_sum(
            -0.5 * tf.reduce_sum(
                1 + tf.clip_by_value(model.z_x_log_sigma_sq, -10.0, 10.0) -
                tf.square(tf.clip_by_value(model.z_x_mean, -10.0, 10.0)) -
                tf.exp(tf.clip_by_value(model.z_x_log_sigma_sq, -10.0, 10.0)),
                1
            )
        ) / model.width / model.height / self.channels

        # Discriminator loss.
        D_loss = tf.reduce_mean(-1.0 * (tf.log(tf.clip_by_value(model.d_x, 1e-5, 1.0)) +
                                        tf.log(tf.clip_by_value(1.0 - model.d_x_p, 1e-5, 1.0))))

        # Generator loss.
        G_loss = tf.reduce_mean(-1.0 * (tf.log(tf.clip_by_value(model.d_x_p, 1e-5, 1.0))))

        # Lth Layer Loss - the 'learned similarity measure'.
        LL_loss = tf.reduce_sum(tf.square(model.l_x - model.l_x_tilde)) / model.width / model.height / self.channels

        # Calculate the losses specific to encoder, decoder, discriminator.
        # L_e = tf.clip_by_value(KL_loss + LL_loss, -100, 100)
        # L_e = tf.clip_by_value(-model.objective + LL_loss, -1000, 1000)
        L_e = tf.clip_by_value(-model.objective, -10000, 10000)
        L_g = tf.clip_by_value(LL_loss + G_loss, -100, 100)
        L_d = tf.clip_by_value(D_loss, -100, 100)

        return L_e, L_g, L_d

    def _build_gradients(self, optimizers, gradients, losses):
        for name, optimizer, loss in zip(['encoder', 'decoder', 'discriminator'], optimizers, losses):
            gradients.setdefault(name, []).append(
                optimizer.compute_gradients(
                    loss,
                    var_list=self._get_model_variables(name, tf.GraphKeys.TRAINABLE_VARIABLES)
                )
            )

    def _build_apply_gradients(self, optimizers, gradients, global_step):
        operations = []
        for name, optimizer in zip(['encoder', 'decoder', 'discriminator'], optimizers):
            # if name != 'decoder' and name != 'discriminator':  # Same as name == 'encoder'
            if name != 'decoder':  # Same as name == 'encoder'
                operations.append(optimizer.apply_gradients(gradients[name], global_step=global_step))

        return tf.group(*operations)

    def _initialize_learning_rate_adjustments(self):
        # We balance the decoder and discriminator learning rate by using a sigmoid function,
        # encouraging the decoder and discriminator to be about equal.
        return 0.5, 0.5

    def _get_learning_rate_adjustments(self, model):
        return model.d_x, model.d_x_p

    def _adjust_learning_rate(self, adjustments, learning_rate, feed_dict):
        e_learning_rate = 1e-3
        g_learning_rate = 1e-3
        d_learning_rate = 1e-3
        d_real, d_fake = adjustments

        feed_dict.update({
            # Encoder.
            learning_rate[0]: e_learning_rate * utils.sigmoid(np.mean(d_real), -0.5, 15),
            # Decoder.
            learning_rate[1]: g_learning_rate * utils.sigmoid(np.mean(d_real), -0.5, 15),
            # Discriminator.
            learning_rate[2]: d_learning_rate * utils.sigmoid(np.mean(d_fake), -0.5, 15),
        })

    def encode_op(self, x, sample=True, with_variance=False):
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

    def discriminator_l_op(self, x):
                # Reshape input as needed.
        # x, width, height = layers.pad_power2(x, self.width, self.height, self.channels)
        width, height = self.width, self.height

        # with slim.arg_scope([layers.discriminator],
        #                     width=width,
        #                     height=height,
        #                     channels=self.channels,
        #                     latent_dim=self.latent_dim,
        #                     is_training=self._training,
        #                     reuse=True):
        with tf.variable_scope(self._model.var_scope, reuse=True):
            discrimination, lth_layer = layers.discriminator(x, width, height)
            return lth_layer

    def reconstruct_op(self, x, sample=True, sample_times=30):
        model = self._model(x, sample=sample_times)
        if sample:
            return model.x_tilde
        else:
            return model.x_tilde_mean

from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
import pickle
from tqdm import tqdm

# this is kind of hacky, model_class is the vae_gan class but its parent abstract-class is in the same folder
from stolen_jernej_code_vae_gan import Model as model_class, utils
from stolen_jernej_code_vae_gan.mnist import mnist_data
from stolen_jernej_code_vae_gan.report import Report
import stolen_jernej_code_vae_gan.other_util_plotting as util2

from mcmc.util import plot, plot_save
from mcmc.mcmc2 import run_experiment, trim_32_to_28

sess = ed.get_session() # need to make sure tf and edward share the global session

# Setup reporting.
run = '1'
report_filename = 'report-{}.txt'.format(run)
report = Report('results/' + report_filename)
report.add("Available GPUs", len(utils.get_available_gpus()))

# Load dataset
mnist_wrapper_object = mnist_data()
mnist_dataset = mnist_wrapper_object.get_data()

# Massage dataset
rescale = False
if rescale == True:
    util2.rescale_dataset(mnist_dataset)

util2.trim_dataset(mnist_dataset, model_class)

# Load model
version = 'v0'
model_attributes = {
    'dataset': mnist_wrapper_object,
    'latent_dim': 50,
    'version': version
}

model = model_class(sess, **model_attributes)

model.build()
model_sample_reconstructions = 0
model.set_defaults(reconstruction={'sampling': model_sample_reconstructions})

checkpoint = 'models/mnist-vae-gan-v0.weights.tfmod'
model.load(checkpoint)

# Try to reconstruct same test images using model to make sure model loaded properly
test_set, test_set_labels = mnist_dataset.test.images[:128], mnist_dataset.test.labels[:128]

utils.plot_digits(mnist_wrapper_object, '{}-originals'.format(mnist_wrapper_object.name), test_set, n=10)

# Plot reconstructions.
with report.add_time_block('time_model_reconstruct_originals'):
    original_reconstructions = model.reconstruct(test_set)
    if np.isnan(original_reconstructions).any():
        print('ERROR: Model produced NaN values for reconstructions.')
        exit(1)

utils.plot_digits(mnist_wrapper_object, '{}-{}-reconstructions'.format(mnist_wrapper_object.name, model.name),
                  original_reconstructions, n=10)

def l2_loss(x_gt, x_hmc):
    with tf.variable_scope('l2_loss', reuse=True):
        return tf.norm(x_gt - x_hmc, axis=len(x_hmc.get_shape().as_list())-1)


def recon_loss(x_gt, x_hmc):
    with tf.variable_scope('recon_loss', reuse=True):
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hmc, labels=x_gt), axis=len(x_hmc.get_shape().as_list())-1)


def l_latent_loss(l_th_x_gt, l_th_x_hmc):
    with tf.variable_scope('l_latent_loss', reuse=True):
        return tf.norm(l_th_x_gt - l_th_x_hmc, axis=len(l_th_x_hmc.get_shape().as_list())-1)

# =============================== INFERENCE ====================================
inference_batch_size = 448  # use a multiple of 32, I'm too lazy to hack around his reconstruct code which only takes mul of 32
start_ind = 0

f = open("./adversarial_examples_v0.pckl", 'rb')
attack_set, attack_set_labels, adversarial_examples, adversarial_targets = pickle.load(f)
f.close()

attack_set = attack_set[start_ind:]
attack_set_labels = attack_set_labels[start_ind:]
adversarial_examples = adversarial_examples[start_ind:]
adversarial_targets = adversarial_targets[start_ind:]

x_ad = adversarial_examples[0:inference_batch_size]
vae_recon = trim_32_to_28(model.reconstruct(x_ad))  # model.reconstruct literally won't run if I put it 30 lines down

for i in range(x_ad.shape[0]):
    plot_save(attack_set[i].reshape(1, 784), # first number is sample number
              './out/{}_x_gt_label_{}_target{}.png'.format(str(start_ind+i+1).zfill(3), attack_set_labels[i], adversarial_targets[i]))
    plot_save(x_ad[i].reshape(1, 784),
              './out/{}_x_adversarial.png'.format(str(start_ind+i+1).zfill(3)))


config = {
    'model': 'hmc',
    'inference_batch_size': inference_batch_size,
    'T': 20000,
    'img_dim': 28,
    'step_size': None,
    'leapfrog_steps': None,
    'friction': None,
    'z_dim': 50,
    'likelihood_variance': 0.48,
    'useDiscL': False,
    'keep_ratio': 0.05
}

# Hack this shit
tf.logging.set_verbosity(tf.logging.ERROR)
model._training = tf.constant([False])

qz, qz_kept = run_experiment(model.decode_op, model.encode_op, x_ad, config, model.discriminator_l_op)

# =============================== EVALUATION ====================================

num_samples = 50
samples = qz_kept.sample(num_samples)

x_samples = trim_32_to_28(model.decode_op(tf.reshape(samples, [-1, config.get('z_dim')])))
l_samples = model.discriminator_l_op(x_samples)
x_samples = tf.reshape(x_samples, [num_samples, inference_batch_size, 784])
l_samples = tf.reshape(l_samples, [num_samples, inference_batch_size, 1024])
x_ad_tensor = tf.constant(x_ad)
l_ad_tensor = model.discriminator_l_op(x_ad_tensor)

# vae_recon = trim_32_to_28(model.decode_op(tf.slice(model.encode_op(
#     tf.concat([x_ad_tensor, tf.zeros([(32 - inference_batch_size % 32), 784])], 0)),
#     [0, 0], [inference_batch_size % 32, config.get('z_dim')])))
vae_l2_loss = l2_loss(x_ad_tensor, vae_recon)
vae_recon_loss = recon_loss(x_ad_tensor, vae_recon)
vae_latent_loss = l_latent_loss(l_ad_tensor, model.discriminator_l_op(vae_recon))

l2_losses = l2_loss(x_ad_tensor, x_samples)
recon_losses = recon_loss(tf.tile(tf.expand_dims(x_ad_tensor, 0), [num_samples, 1, 1]), x_samples)
l_latent_losses = l_latent_loss(l_ad_tensor, l_samples)
mean_imgs = trim_32_to_28(model.decode_op(tf.reduce_mean(samples, axis=0)))

l2_losses, recon_losses, l_latent_losses, x_samples, mean_imgs, vae_recon, vae_l2_loss, vae_recon_loss, vae_latent_loss \
    = sess.run([l2_losses, recon_losses, l_latent_losses, x_samples, mean_imgs, vae_recon, vae_l2_loss, vae_recon_loss, vae_latent_loss])

min_l2 = np.argmin(l2_losses, axis=0)
avg_l2_loss = np.mean(l2_losses, axis=0)
min_recon = np.argmin(recon_losses, axis=0)
avg_recon_loss = np.mean(recon_losses, axis=0)
min_latent = np.argmin(l_latent_losses, axis=0)
avg_latent_loss = np.mean(l_latent_losses, axis=0)

sample_to_vis = 3

f = open('log.txt', 'ab')
for i in tqdm(range(inference_batch_size)):
    img_num = str(start_ind+i+1).zfill(3)
    for j in range(sample_to_vis):
        plot_save(x_samples[j:j+1, i], './out/{}_mcmc_sample_{}.png'.format(img_num, j + 1))

    plot_save(mean_imgs[i:i+1], './out/{}_mcmcMean.png'.format(img_num))

    best_recon_loss = recon_losses[min_recon[i], i]
    best_l2_loss = l2_losses[min_l2[i], i]
    best_latent_loss = l_latent_losses[min_latent[i], i]
    best_recon_sample = x_samples[min_recon[i], i:i+1]
    best_l2_sample = x_samples[min_l2[i], i:i+1]
    best_latent_sample = x_samples[min_latent[i], i:i+1]

    print ("---------- Summary Image {} ------------".format(img_num))
    print("VAE recon loss: " + str(vae_recon_loss[i]))
    print("VAE L2 loss: " + str(vae_l2_loss[i]))
    print("VAE latent loss: " + str(vae_latent_loss[i]))
    print ("Best mcmc recon loss: " + str(best_recon_loss))
    print ("Best mcmc L2 loss: " + str(best_l2_loss))
    print ("Best mcmc latent loss: " + str(best_latent_loss))
    print ("Average mcmc recon loss: " + str(avg_recon_loss[i]))
    print ("Average mcmc l2 loss " + str(avg_l2_loss[i]))
    print ("Average mcmc latent loss " + str(avg_latent_loss[i]))

    plot_save(vae_recon[i:i+1], './out/{}_vae_recon.png'.format(img_num))
    plot_save(best_recon_sample, './out/{}_best_recon.png'.format(img_num))
    plot_save(best_l2_sample, './out/{}_best_l2.png'.format(img_num))
    plot_save(best_latent_sample, './out/{}_best_latent.png'.format(img_num))

    print ("---------- Summary Image {} ------------".format(start_ind+i+1), file=f)
    print("VAE recon loss: " + str(vae_recon_loss[i]), file=f)
    print("VAE L2 loss: " + str(vae_l2_loss[i]), file=f)
    print("VAE latent loss: " + str(vae_latent_loss[i]), file=f)
    print("Best mcmc recon loss: " + str(best_recon_loss), file=f)
    print("Best mcmc L2 loss: " + str(best_l2_loss), file=f)
    print("Best mcmc latent loss: " + str(best_latent_loss), file=f)
    print("Average mcmc recon loss: " + str(avg_recon_loss[i]), file=f)
    print("Average mcmc l2 loss " + str(avg_l2_loss[i]), file=f)
    print("Average mcmc latent loss " + str(avg_latent_loss[i]), file=f)

f.close()

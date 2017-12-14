from __future__ import print_function

import numpy as np
import tensorflow as tf
import edward as ed
import pickle

# this is kind of hacky, model_class is the vae_gan class but its parent abstract-class is in the same folder
from jernej_code_vae_gan import Model as model_class, utils
from jernej_code_vae_gan.mnist import mnist_data
from jernej_code_vae_gan.report import Report
import jernej_code_vae_gan.other_util_plotting as util2

from mcmc.util import plot, plot_save
from mcmc.mcmc_jernej import run_experiment, compare_vae_hmc_loss

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

# checkpoint = 'models/mnist-vae-gan-v0.weights.tfmod'
# model.load(checkpoint)

model.train(mnist_dataset,epochs=1)
print('trained!')

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

# # =============================== INFERENCE ====================================
# inference_batch_size = 100
# start_ind = 0

# f = open("./adversarial_examples_v0.pckl", 'rb')
# attack_set, attack_set_labels, adversarial_examples, adversarial_targets = pickle.load(f)
# f.close()

# attack_set = attack_set[start_ind:]
# attack_set_labels = attack_set_labels[start_ind:]
# adversarial_examples = adversarial_examples[start_ind:]
# adversarial_targets = adversarial_targets[start_ind:]

# x_ad = adversarial_examples[0:inference_batch_size]

# for i in range(x_ad.shape[0]):
#     plot_save(attack_set[i].reshape(1, 784), # first number is sample number
#               './out/{}_x_gt_label_{}_target{}.png'.format(str(start_ind+i+1).zfill(3), attack_set_labels[i], adversarial_targets[i]))
#     plot_save(x_ad[i].reshape(1, 784),
#               './out/{}_x_adversarial.png'.format(str(start_ind+i+1).zfill(3)))


# config = {
#     'model': 'hmc',
#     'inference_batch_size': inference_batch_size,
#     'T': 15000,
#     'img_dim': 28,
#     'step_size': None,
#     'leapfrog_steps': None,
#     'friction': None,
#     'z_dim': 50,
#     'likelihood_variance': 0.48,
#     'useDiscL': False,
#     'keep_ratio': 0.05,
#     'img_num': 0,
#     'sample_to_vis': 3
# }

# # Hack this shit
# tf.logging.set_verbosity(tf.logging.ERROR)
# model._training = tf.constant([False])

# qz, qz_kept = run_experiment(model.decode_op, model.encode_op, x_ad, config, model.discriminator_l_op)

# num_samples = 40
# samples_to_check = qz_kept.sample(num_samples).eval()

# f = open('log.txt', 'ab')
# for i in range(inference_batch_size):
#     config['img_num'] = str(start_ind+i+1).zfill(3)
#     best_recon_loss, average_recon_loss, best_l2_loss, average_l2_loss, best_latent_loss, average_latent_loss, \
#     vae_recon_loss, vae_l2_loss, vae_latent_loss\
#         = compare_vae_hmc_loss(model.decode_op, model.encode_op, model.discriminator_l_op,
#                                x_ad[i:i+1], samples_to_check[:, i, :], config)

#     print ("---------- Summary Image {} ------------".format(start_ind+i+1), file=f)
#     print("VAE recon loss: " + str(vae_recon_loss), file=f)
#     print("VAE L2 loss: " + str(vae_l2_loss), file=f)
#     print("VAE latent loss: " + str(vae_latent_loss), file=f)
#     print("Best mcmc recon loss: " + str(best_recon_loss), file=f)
#     print("Best mcmc L2 loss: " + str(best_l2_loss), file=f)
#     print("Best mcmc latent loss: " + str(best_latent_loss), file=f)
#     print("Average mcmc recon loss: " + str(average_recon_loss), file=f)
#     print("Average mcmc l2 loss " + str(average_l2_loss), file=f)
#     print("Average mcmc latent loss " + str(average_latent_loss), file=f)

# f.close()

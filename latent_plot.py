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

from mcmc.util import plot, plot_save, plot_z
from mcmc.mcmc_jernej import run_experiment, compare_vae_hmc_loss

sess = ed.get_session() # need to make sure tf and edward share the global session

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

# print('get dataset')
# print(model.dataset.get_data().test.images[:1024].shape)
# print('encode dataset')
# print(model.encode(model.dataset.get_data().test.images[:1024]).shape)


with open('Jernej Kos-v1/adversarial_reconstructions_v0_with_labels.pckl', 'rb') as f:
    attack_set, labels, adversarial_examples, adversarial_targets, reconstructions, untargeted_indices, targeted_indices = pickle.load(f)
    attack_set = np.reshape(attack_set.data.numpy(),(992,784))
    print(attack_set.shape)
    adversarial_examples = adversarial_examples
    print(adversarial_examples)
    print(adversarial_targets.shape)
    # print(targeted_indices[0:32])

with open('Jernej Kos-v1/mcmcfiles/003_z.pckl', 'rb') as f:
    print('hi kevin')
    all_z, checked_z, best_l2, best_recon, best_latent = pickle.load(f)
    mcmc_sample = np.reshape(checked_z[best_l2[0]],(50,))


print("CHECKED Z ")
print(mcmc_sample.shape)

# plot_save(model.decode_op(model.encode_op(adversarial_examples[0])).eval(),"blah3")


# model._training=tf.constant([False])
# plot_save(model.decode_op(tf.constant(mcmc_sample.reshape(1,50))).eval(),"blah2")

# z_encoded = model.encode(np.tile(adversarial_examples[0], [32, 1]))
# z_decode = model.decode(z_encoded)

# z_decode = model.decode(np.tile(np.reshape(mcmc_sample,(1,50)), [32, 1]))
# plot_z(z_decode[0:1,:], "blah.png")
# print(z_decode.shape)
# plot_save(z_decode, "blah.png")
# with open('Jernej Kos-v1/adversarial_examples_v0.pckl', 'rb') as f:
#     attack_set, attack_set_labels, adversarial_examples, adversarial_targets = pickle.load(f)
#     print(attack_set.shape)
#     print(attack_set_labels.shape)
#     print(adversarial_examples.shape)
#     print(adversarial_targets.shape)

# print('encode one adversairal example')
# print(model.encode(adversarial_examples).shape)

print('Plotting adversarial examples in latent space.')

print(adversarial_examples.shape)
print(adversarial_examples[0:10,:].shape)
print(model.encode(adversarial_examples[0:32,:]).shape)


# realindex = 2
# utils.plot_latent_space_w_hmc("latentspace{}".format(realindex), [model], 1056,
#     adversarial=adversarial_examples[0+realindex:32+realindex,:],
#     attack=attack_set[0+realindex:32+realindex,:],
#     mcmc=mcmc_sample)

goodis = [2, 5, 16, 22, 24, 29, 31, 37, 40, 43, 47, 57, 59, 66, 72, 73, 77, 96, 100, 109, 118, 119, 140, 142, 143, 145, 154, 156, 166, 168, 176, 178]
for i in goodis:
    realindex = i
    with open('Jernej Kos-v1/mcmcfiles/{}_z.pckl'.format(str(realindex+1).zfill(3)), 'rb') as f:
        print('hi kevin {}'.format(i))
        all_z, checked_z, best_l2, best_recon, best_latent = pickle.load(f)
        mcmc_sample = np.reshape(checked_z[best_l2[0]],(50,))
    utils.plot_latent_space_w_hmc("latentspace{}".format(realindex), [model], 1056,
        adversarial=adversarial_examples[0+realindex:32+realindex,:],
        attack=attack_set[0+realindex:32+realindex,:],
        mcmc=mcmc_sample)

# #
# # print('Plotting nearest neighbours of adversarial examples in latent space.')
# # with report.add_time_block('time_model_plot_nn_adversarial'):
# #     from sklearn.neighbors import BallTree
# #
# #     latent_training = model.encode(data_sets.train.images)
# #     latent_adversarial = model.encode(adversarial_examples)
# #     tree = BallTree(latent_training)
# #     nn_distances, nn_indices = tree.query(latent_adversarial)
# #     nn_images = data_sets.train.images[nn_indices.flatten()]
# #     reconstructed_images = model.reconstruct(nn_images)
# #
# #     utils.plot_digits(dataset, '{}-adversarial-nn'.format(prefix), nn_images, n=10)
# #     utils.plot_digits(dataset, '{}-adversarial-nn-reconstruct'.format(prefix), reconstructed_images, n=10)
#
#
#
print("done.")

import numpy as np
import tensorflow as tf
import edward as ed

# this is kind of hacky, model_class is the vae_gan class but its parent abstract-class is in the same folder
from stolen_jernej_code_vae_gan import Model as model_class, utils
from stolen_jernej_code_vae_gan.mnist import mnist_data
from stolen_jernej_code_vae_gan.report import Report
import stolen_jernej_code_vae_gan.other_util_plotting as util2

from mcmc.util import plot
from mcmc.mcmc import run_experiment, compare_vae_hmc_loss

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

# =============================== INFERENCE ====================================
inference_batch_size = 1
x_gt, _ = mnist_dataset.test.next_batch(inference_batch_size)
plot(x_gt)

config = {
    'model': 'hmc',
    'inference_batch_size': 1,
    'T': 1000,
    'img_dim': 28,
    'step_size': None,
    'leapfrog_steps': None,
    'friction': None,
    'z_dim': 50,
    'likelihood_variance': 0.45
}

# Hack this shit
model._training = tf.constant([False])
qz, qz_kept = run_experiment(model.decode_op, model.encode_op, x_gt, config)
# compare_vae_hmc_loss(model.decode_op, model.encode_op, x_gt, qz_kept, num_samples=100)

import numpy as np
import tensorflow as tf

# this is kind of hacky, model_class is the vae_gan class but its parent abstract-class is in the same folder
from jernej_code_vae_gan import Model as model_class, utils
from jernej_code_vae_gan.mnist import mnist_data
from jernej_code_vae_gan.report import Report
import jernej_code_vae_gan.other_util_plotting as util2
from jernej_code_vae_gan.simple_classifier import Classifier as classifier_class

"""
    TO RUN: You need to download the trained model weights and put them in the /models folder
    Might also need to make the /results folder
"""


# Initialize TensorFlow session.
config = tf.ConfigProto(allow_soft_placement=True)
session = tf.InteractiveSession(config=config)

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

model = model_class(session, **model_attributes)

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

# Plot latent space visualizations.
model_latent_visualization = True
if model_latent_visualization:
    util2.latent_visualization(mnist_wrapper_object, mnist_dataset, model, report)

model_latent_distances = True
if model_latent_distances:
    util2.report_latent_distances(mnist_wrapper_object, mnist_dataset, model, report)

# Load the classifier
classifier_sample = False
classifier = classifier_class(model, num_classes=mnist_wrapper_object.class_count,
                              sample=classifier_sample, version=version)
# classifier doesn't actually need model, we should look to remove it from its class

classifier.build()

checkpoint = 'models/mnist-simple-classifier-vae-gan-v0.weights.tfmod'
classifier.load(checkpoint)

# Check classifier accuracy across all classes
with report.add_time_block('time_classifier_evaluate'):
    accuracy = classifier.evaluate(mnist_dataset.test.images, mnist_dataset.test.labels)
print('Classifier accuracy:', accuracy)
report.add('classifier_accuracy', accuracy)

# Check classifier accuracy for each class
predictions = classifier.predict(mnist_dataset.test.images)
for label, correct, count, accuracy, _ in utils.accuracy_combinations(
        mnist_wrapper_object, mnist_dataset.test.labels, predictions):
    print('Classifier accuracy (class={}): {}/{} ({})'.format(
          label, correct, count, accuracy))
    report.add('original_label{}_correct'.format(label), correct)
    report.add('original_label{}_count'.format(label), count)
    report.add('original_label{}_accuracy'.format(label), accuracy)

# Save the report.
report.save()

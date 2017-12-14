import numpy as np
import utils

"""
    This file contains the logging and visualization methods from main.py of Jernej Kos's repo, with slight
    modifications. Also contains the dataset massaging methods.

"""


def rescale_dataset(mnist_dataset):
    print('Applying rescale transformation to the dataset.')
    mnist_dataset.train._images *= 2.0
    mnist_dataset.train._images -= 1.0
    mnist_dataset.validation._images *= 2.0
    mnist_dataset.validation._images -= 1.0
    mnist_dataset.test._images *= 2.0
    mnist_dataset.test._images -= 1.0

"""
    This isn't optional since their encoder only processes batches of batch_size, no batches with fewer examples
    which is dumb
"""
def trim_dataset(mnist_dataset, model_class):
    # Ensure that data set lengths are a multiple of the batch size.
    mnist_dataset.train._images = utils.clip_to_batch_size(model_class, mnist_dataset.train._images)
    mnist_dataset.train._labels = utils.clip_to_batch_size(model_class, mnist_dataset.train._labels)
    mnist_dataset.validation._images = utils.clip_to_batch_size(model_class, mnist_dataset.validation._images)
    mnist_dataset.validation._labels = utils.clip_to_batch_size(model_class, mnist_dataset.validation._labels)
    mnist_dataset.test._images = utils.clip_to_batch_size(model_class, mnist_dataset.test._images)
    mnist_dataset.test._labels = utils.clip_to_batch_size(model_class, mnist_dataset.test._labels)
    mnist_dataset.train._num_examples = mnist_dataset.train.images.shape[0]
    mnist_dataset.validation._num_examples = mnist_dataset.validation.images.shape[0]
    mnist_dataset.test._num_examples = mnist_dataset.test.images.shape[0]


def latent_visualization(mnist_wrapper_object, mnist_dataset, model, report):
    print('Plotting latent space.')
    with report.add_time_block('time_model_plot_latent_space'):
        utils.plot_latent_space('{}-{}-latent-space'.format(mnist_wrapper_object.name, model.name), [model], 1000)

    # Plot reconstruction of class means.
    means = []
    mean_decodings = []
    for label in xrange(mnist_wrapper_object.class_count):
        latent = model.encode(mnist_dataset.test.images[mnist_dataset.test.labels == label])
        mean = np.mean(latent, axis=0).reshape([1, -1])
        decoding = model.decode(np.tile(mean, [model.batch_size, 1]))
        means.append(mean)
        mean_decodings.append(decoding[0])

    utils.plot_digits(mnist_wrapper_object, '{}-{}-latent-means'.format(mnist_wrapper_object.name, model.name),
                      np.asarray(mean_decodings), n=mnist_wrapper_object.class_count)

    # Plot morphing one class into another.
    def morph(a, b, steps=10):
        latent_a = means[a]
        latent_b = means[b]
        interpolations = []
        for i in xrange(steps + 1):
            interpolation = latent_a + (latent_b - latent_a) * float(i) / steps
            interpolations.append(model.decode(np.tile(interpolation, [model.batch_size, 1]))[0])
        return np.asarray(interpolations)

    for a, b in [(0, 1), (1, 2)]:
        utils.plot_digits(mnist_wrapper_object, '{}-{}-morph-{}-{}'.format(mnist_wrapper_object.name, model.name, a, b),
                          morph(a, b, steps=100), n=10)


def report_latent_distances(mnist_wrapper_object, mnist_dataset, model, report):
    # Compute distances between different pairs of classes in latent space.
    print('Distances between different pairs of classes in latent space.')
    labels = mnist_dataset.test.labels
    latent = model.encode(mnist_dataset.test.images)
    for a in xrange(mnist_wrapper_object.class_count):
        only_a = latent[labels == a]
        values = []
        for b in xrange(a + 1):
            only_b = latent[labels == b]
            count = min(only_a.shape[0], only_b.shape[0])

            distances = []
            for latent_a in only_a:
                for latent_b in only_b:
                    distances.append(np.linalg.norm(latent_a - latent_b))

            values.append('Mean: {0:.2f} Std: {1:.2f}'.format(
                np.mean(distances),
                np.std(distances),
            ))

        print(a, ' '.join(values))
        report_str = ''
        for i in range(len(values)):
            report_str += 'Dest: {} {} \n'.format(a+1+i, values[i])
        report.add('Distance between Pairs of Classes in Latent Space \n Source: {} \n'.format(a), report_str)
from __future__ import print_function

import os
import random
import argparse
import importlib
import matplotlib
import pickle

import numpy as np
import tensorflow as tf

from experiments import utils
from experiments.ensemble import EnsembleClassifier
from experiments.report import Report

# Initailize matplotlib.
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('dataset', type=str, choices=['mnist', 'svhn', 'faces', 'cifar10'])
parser.add_argument('model', type=str, choices=['vae', 'vae_cnn', 'vae_gan'])
parser.add_argument('classifier', type=str, choices=['simple'])
parser.add_argument('attack', type=str, choices=[
    'random',
    'fgs', 'fgs_targeted',
    'optimization', 'optimization_targeted', 'optimization_latent', 'optimization_lvae'
])

# Load all classes.
args = parser.parse_known_args()[0]
dataset_class = getattr(importlib.import_module('experiments.datasets.{}'.format(args.dataset)), 'Dataset')
model_class = getattr(importlib.import_module('experiments.models.{}'.format(args.model)), 'Model')
classifier_class = getattr(importlib.import_module('experiments.classifiers.{}'.format(args.classifier)), 'Classifier')
attack_class = getattr(importlib.import_module('experiments.attacks.{}'.format(args.attack)), 'Attack')

# General options.
parser.add_argument('--model-dir', type=str, default='models')
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--only-existing', action='store_true')
parser.add_argument('--no-eval', action='store_true')
parser.add_argument('--version', type=str, default=None)
parser.add_argument('--ensemble', action='store_true')
parser.add_argument('--ensemble-size', type=int, default=2)
parser.add_argument('--ensemble-combination', type=str, choices=['mean', 'majority'], default='mean')
parser.add_argument('--ensemble-cross-reconstruct', action='store_true')

# Dataset options.
parser.add_argument('--dataset-transform', type=str, choices=['rescale'])
dataset_class.add_options(parser)

# Model options.
parser.add_argument('--model-resume', action='store_true')
parser.add_argument('--model-no-train', action='store_false', dest='model_train')
parser.add_argument('--model-train-epochs', type=int, default=100)
parser.add_argument('--model-checkpoint-every', type=int, default=10)
parser.add_argument('--model-latent-dim', type=str, default='50')
parser.add_argument('--model-latent-visualization', action='store_true')
parser.add_argument('--model-latent-distances', action='store_true')
parser.add_argument('--model-sample-reconstructions', type=int, default=0)

# Classifier options.
parser.add_argument('--classifier-resume', action='store_true')
parser.add_argument('--classifier-no-train', action='store_false', dest='classifier_train')
parser.add_argument('--classifier-train-epochs', type=int, default=100)
parser.add_argument('--classifier-checkpoint-every', type=int, default=10)
parser.add_argument('--classifier-sample', action='store_true')

# Attack options.
parser.add_argument('--attack-generate-examples', action='store_true')
parser.add_argument('--attack-examples-path', type=str, default=None)
parser.add_argument('--attack-set-size', type=int, default=1000)
parser.add_argument('--attack-reconstruct-loops', type=int, default=2)
attack_class.add_options(parser)

args = parser.parse_args()

if not args.model_resume and not args.model_train:
    print('ERROR: You cannot use --model-no-train without --model-resume.')
    exit(1)

if not args.classifier_resume and not args.classifier_train:
    print('ERROR: You cannot use --classifier-no-train without --classifier-resume.')
    exit(1)

# In case of an ensemble, different models may have different latent dimensions.
args.model_latent_dim = [int(x) for x in args.model_latent_dim.split(',')]

# Setup random seed.
print('Using {} as random seed.'.format(args.seed))
random.seed(args.seed)
np.random.seed(args.seed)
tf.set_random_seed(args.seed)

# Setup reporting.
report_filename = 'report-{}-target-{}.txt'.format(args.version, args.attack_target) if args.version else 'report.txt'
report = Report('results/' + report_filename)
report.extend(vars(args))
report.add("Available GPUs", len(utils.get_available_gpus()))

# Initialize TensorFlow session.
config = tf.ConfigProto(allow_soft_placement=True)
session = tf.InteractiveSession(config=config)

# Load dataset.
dataset = dataset_class(args)
print('Loading dataset "{}".'.format(dataset_class.name))

data_sets = dataset.get_data()
data_set_label_count = dataset.class_count
print('Dataset has {} classes.'.format(data_set_label_count))

if args.dataset_transform == 'rescale':
    print('Applying rescale transformation to the dataset.')
    data_sets.train._images *= 2.0
    data_sets.train._images -= 1.0
    data_sets.validation._images *= 2.0
    data_sets.validation._images -= 1.0
    data_sets.test._images *= 2.0
    data_sets.test._images -= 1.0

# Ensure that data set lengths are a multiple of the batch size.
data_sets.train._images = utils.clip_to_batch_size(model_class, data_sets.train._images)
data_sets.train._labels = utils.clip_to_batch_size(model_class, data_sets.train._labels)
data_sets.validation._images = utils.clip_to_batch_size(model_class, data_sets.validation._images)
data_sets.validation._labels = utils.clip_to_batch_size(model_class, data_sets.validation._labels)
data_sets.test._images = utils.clip_to_batch_size(model_class, data_sets.test._images)
data_sets.test._labels = utils.clip_to_batch_size(model_class, data_sets.test._labels)
data_sets.train._num_examples = data_sets.train.images.shape[0]
data_sets.validation._num_examples = data_sets.validation.images.shape[0]
data_sets.test._num_examples = data_sets.test.images.shape[0]

# Construct a test set subset.
test_set, test_set_labels = data_sets.test.images[:128], data_sets.test.labels[:128]

model_attributes = {
    'dataset': dataset,
    'latent_dim': args.model_latent_dim[0],
    'version': args.version
}

utils.plot_digits(dataset, '{}-originals'.format(dataset.name), test_set, n=10)

# Load Model
print('Loading model class "{}".'.format(model_class.name))
model = model_class(session, **model_attributes)

print('Building model "{}".'.format(model.name))
model.build()
model.set_defaults(reconstruction={'sampling': args.model_sample_reconstructions})

checkpoint = os.path.join(args.model_dir, '{}-{}.weights.tfmod'.format(dataset.name, model.name))

if args.model_resume or args.only_existing:
    print('Loading pre-trained model.')
    model.load(checkpoint)

if args.model_train and not args.only_existing:
    print('Training the model.')
    with report.add_time_block('time_model_train'):
        model.train(
            data_sets,
            epochs=args.model_train_epochs,
            checkpoint=checkpoint,
            checkpoint_every=args.model_checkpoint_every,
        )

# Plot reconstructions.
print('Reconstructing inputs.')
with report.add_time_block('time_model_reconstruct_originals'):
    original_reconstructions = model.reconstruct(test_set)
    if np.isnan(original_reconstructions).any():
        print('ERROR: Model produced NaN values for reconstructions.')
        exit(1)

utils.plot_digits(dataset, '{}-{}-reconstructions'.format(dataset.name, model.name),
                  original_reconstructions, n=10)

# Plot latent space.
if args.model_latent_visualization:
    print('Plotting latent space.')
    with report.add_time_block('time_model_plot_latent_space'):
        utils.plot_latent_space('{}-{}-latent-space'.format(dataset.name, model.name), [model], 1000)

    # Plot reconstruction of class means.
    means = []
    mean_decodings = []
    for label in xrange(dataset.class_count):
        latent = model.encode(data_sets.test.images[data_sets.test.labels == label])
        mean = np.mean(latent, axis=0).reshape([1, -1])
        decoding = model.decode(np.tile(mean, [model.batch_size, 1]))
        means.append(mean)
        mean_decodings.append(decoding[0])

    utils.plot_digits(dataset, '{}-{}-latent-means'.format(dataset.name, model.name),
                      np.asarray(mean_decodings), n=dataset.class_count)

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
        utils.plot_digits(dataset, '{}-{}-morph-{}-{}'.format(dataset.name, model.name, a, b),
                          morph(a, b, steps=100), n=10)

if args.model_latent_distances:
    # Compute distances between different pairs of classes in latent space.
    print('Distances between different pairs of classes in latent space.')
    labels = data_sets.test.labels
    latent = model.encode(data_sets.test.images)
    for a in xrange(dataset.class_count):
        only_a = latent[labels == a]
        values = []
        for b in xrange(a + 1):
            only_b = latent[labels == b]
            count = min(only_a.shape[0], only_b.shape[0])

            distances = []
            for latent_a in only_a:
                for latent_b in only_b:
                    distances.append(np.linalg.norm(latent_a - latent_b))

            values.append('"{0:.2f} ({1:.2f})"'.format(
                np.mean(distances),
                np.std(distances),
            ))

        print(a, ' '.join(values))

# Load classifier.
print('Loading classifier "{}".'.format(classifier_class.name))
classifier = classifier_class(model, num_classes=data_set_label_count, sample=args.classifier_sample, version=args.version)
print('Building classifier.')
classifier.build()
checkpoint = os.path.join(args.model_dir,
                          '{}-{}-{}.weights.tfmod'.format(dataset.name, classifier.name, model.name))
if args.classifier_resume or args.only_existing:
    print('Loading pre-trained classifier.')
    classifier.load(checkpoint)
if args.classifier_train and not args.only_existing:
    print('Training the classifier.')
    with report.add_time_block('time_classifier_train'):
        classifier.train(
            data_sets,
            epochs=args.classifier_train_epochs,
            checkpoint=checkpoint,
            checkpoint_every=args.classifier_checkpoint_every,
        )

print('Evaluating classifier.')
with report.add_time_block('time_classifier_evaluate'):
    accuracy = classifier.evaluate(data_sets.test.images, data_sets.test.labels)
print('Classifier accuracy:', accuracy)
report.add('classifier_accuracy', accuracy)

predictions = classifier.predict(data_sets.test.images)
for label, correct, count, accuracy, _ in utils.accuracy_combinations(
        dataset, data_sets.test.labels, predictions):
    print('Classifier accuracy (class={}): {}/{} ({})'.format(
          label, correct, count, accuracy))
    report.add('original_label{}_correct'.format(label), correct)
    report.add('original_label{}_count'.format(label), count)
    report.add('original_label{}_accuracy'.format(label), accuracy)

# Load attack.
print('Loading attack "{}".'.format(args.attack))
attack = attack_class(model, classifier, args) # Bad Hack - sorry
if args.attack_generate_examples:
    print('Generating adversarial examples.')
    attack_set = utils.clip_to_batch_size(model_class, data_sets.test.images[:args.attack_set_size])
    attack_set_labels = utils.clip_to_batch_size(model_class, data_sets.test.labels[:args.attack_set_size])
    print('Attacking first {} examples from the test set.'.format(attack_set.shape[0]))
    
    with report.add_time_block('time_attack_generate'):
        adversarial_examples, adversarial_targets = attack.adversarial_examples(attack_set, attack_set_labels)
    
    if np.isnan(adversarial_examples).any():
        print('ERROR: Attack produced adversarial examples with NaN values!')
        exit(1)

    f = open('adversarial_examples_{}_target_{}.pckl'.format(args.version, args.attack_target), 'wb')
    pickle.dump([attack_set, attack_set_labels, adversarial_examples, adversarial_targets], f)
    f.close()
else:
    f = open(args.attack_examples_path, 'rb')
    attack_set, attack_set_labels, adversarial_examples, adversarial_targets = pickle.load(f)
    f.close()

adversarial_reconstructions = model.reconstruct(adversarial_examples)
f = open('adversarial_reconstructions_{}_target_{}.pckl'.format(args.version, args.attack_target), 'wb')
pickle.dump([attack_set, attack_set_labels, adversarial_examples, adversarial_targets, adversarial_reconstructions], f)
f.close()

if args.no_eval:
    exit()

# target_examples = attack.get_target_examples()
# if target_examples is not None:
#     target_reconstruction, target_source = target_examples

#     utils.plot_digits(
#         dataset,
#         '{}-{}-{}-{}-target-reconstruction'.format(dataset.name, model.name, classifier.name, attack.name),
#         target_reconstruction,
#         n=1
#     )

#     utils.plot_digits(
#         dataset,
#         '{}-{}-{}-{}-target-source'.format(dataset.name, model.name, classifier.name, attack.name),
#         target_source,
#         n=1
#     )

predictions = classifier.predict(adversarial_examples)

print('Distances between adversarial examples and original images.')

def compute_distances(name, difference):
    distances = {
        'l0': lambda x: np.linalg.norm(x, ord=0),
        'l2': lambda x: np.linalg.norm(x, ord=2),
        'li': lambda x: np.linalg.norm(x, ord=np.inf),
        'l2_l0': lambda x: distances['l2'](x) / distances['l0'](x),
        'rmse': lambda x: np.sqrt(np.mean(x ** 2)),
    }

    values = {}
    for value in difference:
        for key, function in distances.items():
            values.setdefault(key, []).append(function(value))

    print('{}:'.format(name.replace('_', ' ').capitalize()))
    for key in distances.keys():
        value = np.mean(values[key])
        print('  {0}={1:.3f}'.format(key, value))
        report.add('adversarial_distance_{}_{}'.format(name, key), value)

for label in xrange(dataset.class_count):
    difference = adversarial_examples[attack_set_labels == label] - attack_set[attack_set_labels == label]
    if not len(difference):
        continue

    compute_distances('class_{}'.format(label), difference)

difference = adversarial_examples - attack_set
compute_distances('overall', difference)

# Sanity check distributions.
print('Original: mean={0:.3f} min={1:.3f} max={2:.3f}'.format(
    np.mean(attack_set), np.min(attack_set), np.max(attack_set)))
print('Adversarial: mean={0:.3f} min={1:.3f} max={2:.3f}'.format(
    np.mean(adversarial_examples), np.min(adversarial_examples), np.max(adversarial_examples)))

# Additional handling for targeted attacks.
if attack.targeted:
    # Filter examples to exclude those that had the same ground truth label as the target.
    different_target_indices = (attack_set_labels != adversarial_targets)
    filtered_labels = attack_set_labels[different_target_indices]
    filtered_predictions = predictions[different_target_indices]
    filtered_targets = adversarial_targets[different_target_indices]
    print('Original example count:', attack_set_labels.shape[0])
    print('Examples with different ground truth label as the target:', filtered_labels.shape[0])
    report.add('targeted_different_gt_count', filtered_labels.shape[0])
else:
    different_target_indices = (attack_set_labels != -1)
    filtered_labels = attack_set_labels
    filtered_predictions = predictions
    filtered_targets = None

utils.plot_digits(dataset, '{}-originals-attack-set'.format(dataset.name), attack_set[different_target_indices], n=10)
utils.plot_digits(dataset, '{}-attack-set-reconstructions'.format(dataset.name),
                  model.reconstruct(attack_set[different_target_indices]), n=10)

correct = np.sum(filtered_predictions == filtered_labels)
count = filtered_labels.shape[0]
accuracy = np.mean(filtered_predictions == filtered_labels)

print('Classifier accuracy on adversarial examples: {}/{} ({})'.format(correct, count, accuracy))
print('Generating reconstructions of adversarial examples.')
report.add('loop0_adversarial_correct', correct)
report.add('loop0_adversarial_accuracy', accuracy)

for src_class, correct, count, accuracy, matching_rate in utils.accuracy_combinations(
        dataset, filtered_labels, filtered_predictions, filtered_targets):
    print('Classifier accuracy on adversarial examples (class={}): {}/{} ({})'.format(
          src_class, correct, count, accuracy))
    report.add('loop0_adversarial_src{}_correct'.format(src_class), correct)
    report.add('loop0_adversarial_src{}_count'.format(src_class), count)
    report.add('loop0_adversarial_src{}_accuracy'.format(src_class), accuracy)
    report.add('loop0_adversarial_src{}_matching_rate'.format(src_class), matching_rate)

utils.plot_digits(dataset, '{}-{}-{}-{}-adversarial-examples'.format(
                  dataset.name, model.name, classifier.name, attack.name),
                  adversarial_examples[different_target_indices], n=10)

# Compute matching rate.
if attack.targeted:
    incorrect, matching, rate = utils.get_matching_rate(filtered_labels, filtered_predictions, filtered_targets)
    print('Targeted attack matching rate: {}/{} ({})'.format(matching, incorrect, rate))

    report.add('loop0_adversarial_targeted_incorrect', incorrect)
    report.add('loop0_adversarial_targeted_matching', matching)
    report.add('loop0_adversarial_targeted_matching_rate', rate)


# def reconstruction_loops(model, index=None):
#     if index is not None:
#         prefix = '{}-{}-{}-{}-ensemble-{}'.format(dataset.name, model.name, classifier.name, attack.name, index)
#         report_prefix = 'loop{{}}_adversarial_{}'.format(index)
#     else:
#         prefix = '{}-{}-{}-{}'.format(dataset.name, model.name, classifier.name, attack.name)
#         report_prefix = 'loop{}_adversarial'

#     adversarial_reconstructions = model.reconstruct(adversarial_examples)

#     utils.plot_digits(dataset, '{}-adversarial-reconstructions'.format(prefix),
#                       adversarial_reconstructions, n=10)

#     utils.plot_digits(dataset, '{}-adversarial-reconstructions-labels'.format(prefix),
#                       adversarial_reconstructions, n=10, labels=(attack_set_labels, predictions))

#     # Feed reconstructed versions back to the model and assess accuracy.
#     result = {}
#     for loop in xrange(args.attack_reconstruct_loops):
#         result[loop] = {}
#         if index is not None:
#             result[loop]['model_predictions'] = classifier.classifiers[index].predict(
#                 utils.crop_for_model(model, adversarial_reconstructions))

#         predictions_2nd = classifier.predict(utils.crop_for_model(model, adversarial_reconstructions))
#         if attack.targeted:
#             filtered_predictions_2nd = predictions_2nd[different_target_indices]
#         else:
#             filtered_predictions_2nd = predictions_2nd

#         incorrect_indices = (filtered_predictions_2nd != filtered_labels)
#         correct = np.sum(filtered_predictions_2nd == filtered_labels)
#         count = filtered_labels.shape[0]
#         accuracy = np.mean(filtered_predictions_2nd == filtered_labels)

#         print('Loop', loop + 1)
#         print('[{}] Classifier accuracy on reconstructed adversarial examples: {}/{} ({})'.format(
#               index or '0', correct, count, accuracy))

#         if attack.targeted:
#             incorrect, matching, rate = utils.get_matching_rate(
#                 filtered_labels, filtered_predictions_2nd, filtered_targets)
#             print('[{}] Targeted attack matching rate: {}/{} ({})'.format(index or '0', matching, incorrect, rate))

#             report.add('{}_targeted_incorrect'.format(report_prefix.format(loop + 1)), incorrect)
#             report.add('{}_targeted_matching'.format(report_prefix.format(loop + 1)), matching)
#             report.add('{}_targeted_matching_rate'.format(report_prefix.format(loop + 1)), rate)

#         report.add('{}_correct'.format(report_prefix.format(loop + 1)), correct)
#         report.add('{}_accuracy'.format(report_prefix.format(loop + 1)), accuracy)

#         for src_class, correct, count, accuracy, matching_rate in utils.accuracy_combinations(
#                 dataset, filtered_labels, filtered_predictions_2nd, filtered_targets):
#             print('[{}] Classifier accuracy on adversarial examples (class={}): {}/{} ({})'.format(
#                   index or '0', src_class, correct, count, accuracy))
#             report.add('{}_src{}_correct'.format(report_prefix.format(loop + 1), src_class), correct)
#             report.add('{}_src{}_count'.format(report_prefix.format(loop + 1), src_class), count)
#             report.add('{}_src{}_accuracy'.format(report_prefix.format(loop + 1), src_class), accuracy)
#             report.add('{}_src{}_matching_rate'.format(report_prefix.format(loop + 1), src_class), matching_rate)

#         adversarial_reconstructions = model.reconstruct(
#             utils.crop_for_model(model, adversarial_reconstructions))

#         if attack.targeted:
#             filtered_reconstructions = adversarial_reconstructions[different_target_indices]
#         else:
#             filtered_reconstructions = adversarial_reconstructions

#         utils.plot_digits(dataset, '{}-adversarial-reconstructions-loop-{}'.format(prefix, loop + 1),
#                           adversarial_reconstructions, n=10)

#         utils.plot_digits(dataset, '{}-adversarial-reconstructions-loop-{}-incorrect'.format(prefix, loop + 1),
#                           filtered_reconstructions[incorrect_indices], n=10)

#         result[loop]['reconstructions'] = adversarial_reconstructions
#         result[loop]['predictions'] = predictions_2nd

#     return result

# results = {}
# results[0] = reconstruction_loops(model)

# # Generate plots.
# predictions_loop1 = results[0][0]['predictions']
# predictions_loop2 = results[0][1]['predictions']
# reconstructions_loop1 = results[0][0]['reconstructions']
# indices_wrong = (predictions_loop1 != attack_set_labels)
# if attack.targeted:
#     indices_notarget = (attack_set_labels != adversarial_targets)
#     indices_wrong &= indices_notarget
# else:
#     indices_notarget = None

# prefix = '{}-{}-{}-{}'.format(dataset.name, model.name, classifier.name, attack.name)

# attack_set_reconstructions = model.reconstruct(attack_set)
# attack_set_reconstructions_sample_1 = model.reconstruct(attack_set, sample=True, sample_times=1)
# attack_set_reconstructions_sample_12 = model.reconstruct(attack_set, sample=True, sample_times=12)
# attack_set_reconstructions_sample_50 = model.reconstruct(attack_set, sample=True, sample_times=50)
# adversarial_reconstructions = model.reconstruct(adversarial_examples)
# adversarial_reconstructions_sample_1 = model.reconstruct(adversarial_examples, sample=True, sample_times=1)
# adversarial_reconstructions_sample_12 = model.reconstruct(adversarial_examples, sample=True, sample_times=12)
# adversarial_reconstructions_sample_50 = model.reconstruct(adversarial_examples, sample=True, sample_times=50)

# utils.plot_adversarial_digits(
#     '{}-adversarial-digits-attack'.format(prefix),
#     dataset,
#     model,
#     attack_set,
#     attack_set_reconstructions,
#     attack_set_reconstructions_sample_1,
#     attack_set_reconstructions_sample_12,
#     attack_set_reconstructions_sample_50,
#     adversarial_examples,
#     adversarial_reconstructions,
#     adversarial_reconstructions_sample_1,
#     adversarial_reconstructions_sample_12,
#     adversarial_reconstructions_sample_50,
#     reconstructions_loop1,
#     attack_set_labels,
#     predictions,
#     predictions_loop1,
#     predictions_loop2,
#     target_labels=adversarial_targets if attack.targeted else None,
#     indices=indices_wrong
# )

# utils.plot_adversarial_digits(
#     '{}-adversarial-digits-all'.format(prefix),
#     dataset,
#     model,
#     attack_set,
#     attack_set_reconstructions,
#     attack_set_reconstructions_sample_1,
#     attack_set_reconstructions_sample_12,
#     attack_set_reconstructions_sample_50,
#     adversarial_examples,
#     adversarial_reconstructions,
#     adversarial_reconstructions_sample_1,
#     adversarial_reconstructions_sample_12,
#     adversarial_reconstructions_sample_50,
#     reconstructions_loop1,
#     attack_set_labels,
#     predictions,
#     predictions_loop1,
#     predictions_loop2,
#     target_labels=adversarial_targets if attack.targeted else None,
#     indices=indices_notarget
# )

# Plot adversarial examples in latent space.
if args.model_latent_visualization:
    print('Plotting adversarial examples in latent space.')
    with report.add_time_block('time_model_plot_latent_space_adversarial'):
        utils.plot_latent_space('{}-{}-latent-space-adversarial'.format(dataset.name, model.name), [model], 10000,
                                adversarial_examples)

    print('Plotting nearest neighbours of adversarial examples in latent space.')
    with report.add_time_block('time_model_plot_nn_adversarial'):
        from sklearn.neighbors import BallTree

        latent_training = model.encode(data_sets.train.images)
        latent_adversarial = model.encode(adversarial_examples)
        tree = BallTree(latent_training)
        nn_distances, nn_indices = tree.query(latent_adversarial)
        nn_images = data_sets.train.images[nn_indices.flatten()]
        reconstructed_images = model.reconstruct(nn_images)

        utils.plot_digits(dataset, '{}-adversarial-nn'.format(prefix), nn_images, n=10)
        utils.plot_digits(dataset, '{}-adversarial-nn-reconstruct'.format(prefix), reconstructed_images, n=10)


# Save the report.
report.save()

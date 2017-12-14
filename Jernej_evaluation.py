import numpy as np
from jernej_code_vae_gan import utils

"""
    List of variables required to run Jernej's code below
"""

report = None
model = None
classifier = None
adversarial_examples = None # adversarial images
dataset = None # mnist_wrapper_object
data_sets = None # mnist_dataset

attack_set = None # the images you attacked
attack_set_labels = None # the labels for the images you attacked (generated adversarial images for) (=ground truth)
attack_targeted = True
adversarial_targets = None # the target class of the attack. E.g. make all the images 9
attack_name = 'optimization-latent-v0'
attack_reconstruct_loops = 2

model_latent_visualization = True

"""
    Jernej's code to evaluate the model once adversarial images have been generated, is below
"""

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
if attack_targeted:
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
                  dataset.name, model.name, classifier.name, attack_name),
                  adversarial_examples[different_target_indices], n=10)

# Compute matching rate.
if attack_targeted:
    incorrect, matching, rate = utils.get_matching_rate(filtered_labels, filtered_predictions, filtered_targets)
    print('Targeted attack matching rate: {}/{} ({})'.format(matching, incorrect, rate))

    report.add('loop0_adversarial_targeted_incorrect', incorrect)
    report.add('loop0_adversarial_targeted_matching', matching)
    report.add('loop0_adversarial_targeted_matching_rate', rate)


def reconstruction_loops(model, index=None):
    if index is not None:
        prefix = '{}-{}-{}-{}-ensemble-{}'.format(dataset.name, model.name, classifier.name, attack_name, index)
        report_prefix = 'loop{{}}_adversarial_{}'.format(index)
    else:
        prefix = '{}-{}-{}-{}'.format(dataset.name, model.name, classifier.name, attack_name)
        report_prefix = 'loop{}_adversarial'

    adversarial_reconstructions = model.reconstruct(adversarial_examples)

    utils.plot_digits(dataset, '{}-adversarial-reconstructions'.format(prefix),
                      adversarial_reconstructions, n=10)

    utils.plot_digits(dataset, '{}-adversarial-reconstructions-labels'.format(prefix),
                      adversarial_reconstructions, n=10, labels=(attack_set_labels, predictions))

    # Feed reconstructed versions back to the model and assess accuracy.
    result = {}
    for loop in xrange(attack_reconstruct_loops):
        result[loop] = {}
        if index is not None:
            result[loop]['model_predictions'] = classifier.classifiers[index].predict(
                utils.crop_for_model(model, adversarial_reconstructions))

        predictions_2nd = classifier.predict(utils.crop_for_model(model, adversarial_reconstructions))
        if attack_targeted:
            filtered_predictions_2nd = predictions_2nd[different_target_indices]
        else:
            filtered_predictions_2nd = predictions_2nd

        incorrect_indices = (filtered_predictions_2nd != filtered_labels)
        correct = np.sum(filtered_predictions_2nd == filtered_labels)
        count = filtered_labels.shape[0]
        accuracy = np.mean(filtered_predictions_2nd == filtered_labels)

        print('Loop', loop + 1)
        print('[{}] Classifier accuracy on reconstructed adversarial examples: {}/{} ({})'.format(
              index or '0', correct, count, accuracy))

        if attack_targeted:
            incorrect, matching, rate = utils.get_matching_rate(
                filtered_labels, filtered_predictions_2nd, filtered_targets)
            print('[{}] Targeted attack matching rate: {}/{} ({})'.format(index or '0', matching, incorrect, rate))

            report.add('{}_targeted_incorrect'.format(report_prefix.format(loop + 1)), incorrect)
            report.add('{}_targeted_matching'.format(report_prefix.format(loop + 1)), matching)
            report.add('{}_targeted_matching_rate'.format(report_prefix.format(loop + 1)), rate)

        report.add('{}_correct'.format(report_prefix.format(loop + 1)), correct)
        report.add('{}_accuracy'.format(report_prefix.format(loop + 1)), accuracy)

        for src_class, correct, count, accuracy, matching_rate in utils.accuracy_combinations(
                dataset, filtered_labels, filtered_predictions_2nd, filtered_targets):
            print('[{}] Classifier accuracy on adversarial examples (class={}): {}/{} ({})'.format(
                  index or '0', src_class, correct, count, accuracy))
            report.add('{}_src{}_correct'.format(report_prefix.format(loop + 1), src_class), correct)
            report.add('{}_src{}_count'.format(report_prefix.format(loop + 1), src_class), count)
            report.add('{}_src{}_accuracy'.format(report_prefix.format(loop + 1), src_class), accuracy)
            report.add('{}_src{}_matching_rate'.format(report_prefix.format(loop + 1), src_class), matching_rate)

        adversarial_reconstructions = model.reconstruct(
            utils.crop_for_model(model, adversarial_reconstructions))

        if attack_targeted:
            filtered_reconstructions = adversarial_reconstructions[different_target_indices]
        else:
            filtered_reconstructions = adversarial_reconstructions

        utils.plot_digits(dataset, '{}-adversarial-reconstructions-loop-{}'.format(prefix, loop + 1),
                          adversarial_reconstructions, n=10)

        utils.plot_digits(dataset, '{}-adversarial-reconstructions-loop-{}-incorrect'.format(prefix, loop + 1),
                          filtered_reconstructions[incorrect_indices], n=10)

        result[loop]['reconstructions'] = adversarial_reconstructions
        result[loop]['predictions'] = predictions_2nd

    return result

results = {}
results[0] = reconstruction_loops(model)

# Generate plots.
predictions_loop1 = results[0][0]['predictions']
predictions_loop2 = results[0][1]['predictions']
reconstructions_loop1 = results[0][0]['reconstructions']
indices_wrong = (predictions_loop1 != attack_set_labels)
if attack_targeted:
    indices_notarget = (attack_set_labels != adversarial_targets)
    indices_wrong &= indices_notarget
else:
    indices_notarget = None

prefix = '{}-{}-{}-{}'.format(dataset.name, model.name, classifier.name, attack_name)

attack_set_reconstructions = model.reconstruct(attack_set)
attack_set_reconstructions_sample_1 = model.reconstruct(attack_set, sample=True, sample_times=1)
attack_set_reconstructions_sample_12 = model.reconstruct(attack_set, sample=True, sample_times=12)
attack_set_reconstructions_sample_50 = model.reconstruct(attack_set, sample=True, sample_times=50)
adversarial_reconstructions = model.reconstruct(adversarial_examples)
adversarial_reconstructions_sample_1 = model.reconstruct(adversarial_examples, sample=True, sample_times=1)
adversarial_reconstructions_sample_12 = model.reconstruct(adversarial_examples, sample=True, sample_times=12)
adversarial_reconstructions_sample_50 = model.reconstruct(adversarial_examples, sample=True, sample_times=50)

utils.plot_adversarial_digits(
    '{}-adversarial-digits-attack'.format(prefix),
    dataset,
    model,
    attack_set,
    attack_set_reconstructions,
    attack_set_reconstructions_sample_1,
    attack_set_reconstructions_sample_12,
    attack_set_reconstructions_sample_50,
    adversarial_examples,
    adversarial_reconstructions,
    adversarial_reconstructions_sample_1,
    adversarial_reconstructions_sample_12,
    adversarial_reconstructions_sample_50,
    reconstructions_loop1,
    attack_set_labels,
    predictions,
    predictions_loop1,
    predictions_loop2,
    target_labels=adversarial_targets if attack_targeted else None,
    indices=indices_wrong
)

utils.plot_adversarial_digits(
    '{}-adversarial-digits-all'.format(prefix),
    dataset,
    model,
    attack_set,
    attack_set_reconstructions,
    attack_set_reconstructions_sample_1,
    attack_set_reconstructions_sample_12,
    attack_set_reconstructions_sample_50,
    adversarial_examples,
    adversarial_reconstructions,
    adversarial_reconstructions_sample_1,
    adversarial_reconstructions_sample_12,
    adversarial_reconstructions_sample_50,
    reconstructions_loop1,
    attack_set_labels,
    predictions,
    predictions_loop1,
    predictions_loop2,
    target_labels=adversarial_targets if attack_targeted else None,
    indices=indices_notarget
)

# Plot adversarial examples in latent space.
if model_latent_visualization:
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

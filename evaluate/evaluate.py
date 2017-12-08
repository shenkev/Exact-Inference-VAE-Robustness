import torch
import numpy as np
from PIL import Image
from torchvision import datasets, transforms
from mnist_model import Net
from torch.autograd import Variable
import os

import pickle
import logging

import pdb

def trim_to_28_from_32(data, num_data):
    return np.array(data).reshape(num_data, 32, 32)[:, 2:30, 2:30].reshape(num_data, 784)

def evaluate(dataset_name, dataset, ground_truth, adversarial_target, model, num_classes):

    logging.info('')
    logging.info("Evaluating accuracy for " + dataset_name)

    # Tracking Indicies
    untargeted_success = []
    targeted_success = []

    # Jesse Indicies
    index_of_interest = set([2, 5, 16, 22, 24, 29, 31, 37, 40, 43, 47, 57, 59, 66, 72, 73, 77, 96, 100, 109, 118, 119, 140, 142, 143, 145, 154, 156, 166, 168, 176, 178])
    success_index_of_interest = []

    # Get model predictions
    output = model(dataset)
    pred = output.data.max(1, keepdim=True)[1].numpy()
    
    # calculate totals
    correct_per_class = [0]*num_classes
    adversarial_success_per_class = [0]*num_classes
    number_per_class = [0]*num_classes
    for index, y, y_pred in zip(range(len(ground_truth)), ground_truth, pred):
        if y != adversarial_target:
            if y_pred == y:
                correct_per_class[y] += 1
            if y_pred == adversarial_target:
                adversarial_success_per_class[y] += 1
                targeted_success.append(index)
            if y_pred != y:
                untargeted_success.append(index)
            if index in index_of_interest and y_pred == y:
                success_index_of_interest.append(index)

            number_per_class[y] += 1 
    pdb.set_trace()
    # log correct metrics
    logging.info('')
    logging.info("Correct Metrics")
    logging.info('')
    logging.info('Overall: {}/{} ({})'.format(sum(correct_per_class) - correct_per_class[adversarial_target], sum(number_per_class) - number_per_class[adversarial_target],
                                              float(sum(correct_per_class) - correct_per_class[adversarial_target]) / (sum(number_per_class) - number_per_class[adversarial_target])))
    for class_num, correct, total in zip(range(num_classes), correct_per_class, number_per_class):
        if class_num != adversarial_target:
            logging.info('Class {}: {}/{} ({})'.format(class_num, correct, total, float(correct) / total))
        else:
            logging.info('No metrics for {} as it same as adversarial target'.format(class_num))

    # log adversarial metrics
    logging.info('')
    logging.info("Adversarial Metrics")
    logging.info('')
    logging.info('Overall: {}/{} ({})'.format(sum(adversarial_success_per_class) - adversarial_success_per_class[adversarial_target], sum(number_per_class) - number_per_class[adversarial_target],
                                          float(sum(adversarial_success_per_class) - adversarial_success_per_class[adversarial_target]) / (sum(number_per_class) - number_per_class[adversarial_target])))
    for class_num, adversarial_success, total in zip(range(num_classes), adversarial_success_per_class, number_per_class):
        if class_num != adversarial_target:
            logging.info('Class {}: {}/{} ({})'.format(class_num, adversarial_success, total, float(adversarial_success) / total))
        else:
            logging.info('No metrics for {} as it same as adversarial target'.format(class_num))


    return untargeted_success, targeted_success


# Constants
NUM_CLASSES = 10
ADVERSARIAL_TARGET = 7
LOG_FILE = 'target_7_old.log'
DATA_DIR = 'pixel_loss_results_2'

# Set-up logging
logging.basicConfig(filename=LOG_FILE, format='%(message)s', level=logging.INFO)
logging.info('Begin Evaluation')


#Load data
all_data_files = [ (int(data_file.split('_')[0]), os.path.join(DATA_DIR, data_file)) for data_file in os.listdir(DATA_DIR) if 'pckl' not in data_file]

best_l2 = []
best_latent = []
best_recon = []
mcmc_random = []
mcmc_mean = []
vae_recon = []
adversarial = []
ground_truth = []
labels = []

for num, data_file in all_data_files:

    with open(data_file, 'rb') as f:
        np_data = np.array(f.readline().split()).astype(float)

    if 'best_l2' in data_file:
        best_l2.append((num, np_data))
    if 'best_latent' in data_file:
        best_latent.append((num, np_data))
    if 'best_recon' in data_file:
        best_recon.append((num, np_data))
    if 'mcmc_sample_1' in data_file:
        mcmc_random.append((num, np_data))
    if 'mcmcMean' in data_file:
        mcmc_mean.append((num, np_data))
    if 'vae_recon' in data_file:
        vae_recon.append((num, np_data))
    if 'x_adversarial' in data_file:
        adversarial.append((num, np_data))
    if 'x_gt' in data_file:
        ground_truth.append((num, np_data))
        labels.append((num, int(data_file.split('_')[7])))

best_l2 = np.array([y[1] for y in sorted(best_l2, key=lambda x: x[0])])
best_latent = np.array([y[1] for y in sorted(best_latent, key=lambda x: x[0])])
best_recon = np.array([y[1] for y in sorted(best_recon, key=lambda x: x[0])])
mcmc_random = np.array([y[1] for y in sorted(mcmc_random, key=lambda x: x[0])])
mcmc_mean = np.array([y[1] for y in sorted(mcmc_mean, key=lambda x: x[0])])
vae_recon = np.array([y[1] for y in sorted(vae_recon, key=lambda x: x[0])])
adversarial = np.array([y[1] for y in sorted(adversarial, key=lambda x: x[0])])
ground_truth = np.array([y[1] for y in sorted(ground_truth, key=lambda x: x[0])])
labels = np.array([y[1] for y in sorted(labels, key=lambda x: x[0])])

data_size = len(labels)

# with open("adversarial_reconstructions_v0.pckl", 'rb') as f:
#     ground_truth, labels, adversarial_examples, adversarial_targets, reconstructions = pickle.load(f)
# data_size = len(labels)


# Modify data for evaluation
normalize = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])

ground_truth = Variable(normalize(torch.FloatTensor(ground_truth)).view(data_size, 1, 28, 28), volatile=True)
adversarial = Variable(normalize(torch.FloatTensor(adversarial)).view(data_size, 1, 28, 28), volatile=True)
vae_recon = Variable(normalize(torch.FloatTensor(vae_recon)).view(data_size, 1, 28, 28), volatile=True)
best_l2 = Variable(normalize(torch.FloatTensor(best_l2)).view(data_size, 1, 28, 28), volatile=True)
best_latent = Variable(normalize(torch.FloatTensor(best_latent)).view(data_size, 1, 28, 28), volatile=True)
best_recon = Variable(normalize(torch.FloatTensor(best_recon)).view(data_size, 1, 28, 28), volatile=True)
mcmc_mean = Variable(normalize(torch.FloatTensor(mcmc_mean)).view(data_size, 1, 28, 28), volatile=True)
mcmc_random = Variable(normalize(torch.FloatTensor(mcmc_random)).view(data_size, 1, 28, 28), volatile=True)

# ground_truth = Variable(normalize(torch.FloatTensor(ground_truth)).view(data_size, 1, 28, 28), volatile=True)
# adversarial = Variable(normalize(torch.FloatTensor(adversarial_examples)).view(data_size, 1, 28, 28), volatile=True)
# recon = Variable(normalize(torch.FloatTensor(trim_to_28_from_32(reconstructions, data_size))).view(data_size, 1, 28, 28), volatile=True)

# Load model
model = Net()
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

# Evaluate
# evaluate(dataset_name="Ground Truth", dataset=ground_truth, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="Adversarial", dataset=adversarial, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="Reconstructions", dataset=recon, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)

# with open("adversarial_reconstructions_v0_with_labels.pckl", 'wb') as f:
#     pickle.dump([ground_truth, labels, adversarial_examples, adversarial_targets, reconstructions, untargeted_success, targeted_success], f)



evaluate(dataset_name="Best L2", dataset=best_l2, ground_truth=labels,
         adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="Best Latent", dataset=best_latent, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="Best Recon", dataset=best_recon, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="MCMC Mean", dataset=mcmc_mean, ground_truth=labels,
#          adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
# evaluate(dataset_name="MCMC Random", dataset=mcmc_random, ground_truth=labels,
         # adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)


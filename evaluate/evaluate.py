import torch
import numpy as np
from torchvision import datasets, transforms
from mnist_model import Net
from torch.autograd import Variable

import pickle
import logging

import pdb

def trim_to_28_from_32(data, num_data):
    return np.array(data).reshape(num_data, 32, 32)[:, 2:30, 2:30].reshape(num_data, 784)

def evaluate(dataset_name, dataset, ground_truth, adversarial_target, model, num_classes):

    logging.info('')
    logging.info("Evaluating accuracy for " + dataset_name)

    # Get model predictions
    output = model(dataset)
    pred = output.data.max(1, keepdim=True)[1].numpy()
    
    # calculate totals
    correct_per_class = [0]*num_classes
    adversarial_success_per_class = [0]*num_classes
    number_per_class = [0]*num_classes
    for y, y_pred in zip(ground_truth, pred):
        if y != adversarial_target:
            if y_pred == y:
                correct_per_class[y] += 1
            if y_pred == adversarial_target:
                adversarial_success_per_class[y] += 1
            number_per_class[y] += 1 
    
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


# Constants
NUM_CLASSES = 10
ADVERSARIAL_TARGET = 7
LOG_FILE = 'results.log'

# Set-up logging
logging.basicConfig(filename=LOG_FILE, format='%(message)s', level=logging.INFO)
logging.info('Begin Evaluation')

# Load data 
with open("adversarial_reconstructions_v0.pckl", 'rb') as f:
    attack_set, attack_set_labels, adversarial_examples, adversarial_targets, vae_reconstructions = pickle.load(f)
data_size = len(attack_set_labels)

# Modify data for evaluation
normalize = transforms.Compose([transforms.Normalize((0.1307,), (0.3081,))])
original_set = Variable(normalize(torch.FloatTensor(attack_set)).view(data_size, 1, 28, 28), volatile=True)
adversarial_set = Variable(normalize(torch.FloatTensor(adversarial_examples)).view(data_size, 1, 28, 28), volatile=True)
vae_reconstructions = Variable(normalize(torch.FloatTensor(trim_to_28_from_32(vae_reconstructions, data_size))).view(data_size, 1, 28, 28), volatile=True)


# Load model
model = Net()
model.load_state_dict(torch.load("checkpoint.pth"))
model.eval()

# Evaluate
evaluate(dataset_name="Original Set", dataset=original_set, ground_truth=attack_set_labels,
         adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
evaluate(dataset_name="Adversarial Set", dataset=adversarial_set, ground_truth=attack_set_labels,
         adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)
evaluate(dataset_name="VAE reconstructions", dataset=vae_reconstructions, ground_truth=attack_set_labels,
         adversarial_target=ADVERSARIAL_TARGET, model=model, num_classes=NUM_CLASSES)


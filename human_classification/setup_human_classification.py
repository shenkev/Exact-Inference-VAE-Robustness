import csv
from shutil import copyfile
import numpy as np
import pickle

imgs_to_check = 10
img_path = './out/'
classify_path = './human/'

header = ['orig_filename', 'orig_filetype', 'gt_class', 'target_class', 'classifier_class', 'new_filename', 'human_class']
myFile = open(classify_path + 'data.csv', 'w')

# This the lazy way to find out the gt_class and target_class, we assume the image indices match
data_path = 'adversarial_examples_v0.pckl'
f = open(data_path, 'rb')
attack_set, attack_set_labels, adversarial_examples, adversarial_targets = pickle.load(f)
f.close()

data_path = 'v0_target_7_old_class.pckl'
f = open(data_path, 'rb')
labels, gt_class, adv_class, vae_class, l2_class, latent_class, recon_class, mcmcmean_class, mcmcrandom_class = pickle.load(f)
f.close()

# Pick shuffled file names
total_imgs = imgs_to_check*3  # check best_l2, hmc_mean and adversarial img
new_imgnums = np.random.choice(range(1, total_imgs+1), total_imgs, replace=False)

with myFile:
    writer = csv.writer(myFile)
    writer.writerow(header)

    for i in range(imgs_to_check):
        # This code is pretty brittle to filename change but w.e.
        img_num = str(i+1).zfill(3)

        # L2
        best_l2_img = '{}_best_l2.png'.format(img_num)

        new_filename = "{}.png".format(str(new_imgnums[i*3]).zfill(3))
        row = [best_l2_img, 'hmc_l2', attack_set_labels[i], adversarial_targets[i], l2_class[i], new_filename, -1]
        writer.writerow(row)

        copyfile(img_path + best_l2_img, classify_path + new_filename)

        # mcmc mean
        mcmc_mean = '{}_mcmcMean.png'.format(img_num)

        new_filename = "{}.png".format(str(new_imgnums[i*3+1]).zfill(3))
        row = [mcmc_mean, 'mcmc_mean', attack_set_labels[i], adversarial_targets[i], mcmcmean_class[i], new_filename, -1]
        writer.writerow(row)

        copyfile(img_path + mcmc_mean, classify_path + new_filename)

        # adv. image
        adv_img = '{}_x_adversarial.png'.format(img_num)

        new_filename = "{}.png".format(str(new_imgnums[i*3+2]).zfill(3))
        row = [adv_img, 'adv', attack_set_labels[i], adversarial_targets[i], adv_class[i], new_filename, -1]
        writer.writerow(row)

        copyfile(img_path + adv_img, classify_path + new_filename)
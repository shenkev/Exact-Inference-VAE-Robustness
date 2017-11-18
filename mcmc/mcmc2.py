from edward.models import Empirical, Normal
import edward as ed
import tensorflow as tf

from util import plot, plot_save
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

"""
    Because I screwed up the first mcmc.py by doing for-loops over samples

"""

jernej_Q_P = True

def trim_32_to_28(input):
    trimed_list = None
    for i in range(input.shape[0]):
        trimed = tf.reshape(tf.slice(tf.reshape(input[i], [32, 32]), [2, 2], [28, 28]), [1, 784])
        if trimed_list is None:
            trimed_list = trimed
        else:
            trimed_list = tf.concat([trimed_list, trimed], 0)
    return trimed_list

def build_experiment(P, Q, x_gt, config, DiscL):

    model = config.get('model')
    T = config.get('T')
    img_dim = config.get('img_dim')
    step_size = config.get('step_size')
    leapfrog_steps = config.get('leapfrog_steps')
    friction = config.get('friction')
    z_dim = config.get('z_dim')
    likelihood_variance = config.get('likelihood_variance')

    inference_batch_size = int(x_gt.shape[0])

    z = Normal(loc=tf.zeros([inference_batch_size, z_dim]), scale=tf.ones([inference_batch_size, z_dim]))  # sample z

    if jernej_Q_P:
        normalized_dec_x = P(z)
        normalized_dec_x = trim_32_to_28(normalized_dec_x)
    else:
        normalized_dec_x = P(z)[0]

    # Compute lth layer of Disc from decoded image
    if config['useDiscL']:
        disc_l_normalized_dec_x = DiscL(normalized_dec_x)
        X = Normal(loc=disc_l_normalized_dec_x, scale=likelihood_variance) # Using L2 Loss in the Discriminator Lth Layer space for HMC P(X|z)
        x_gt = DiscL(x_gt)  # This is hacky but it allows us to keep the if-else logic simple
    else:
        X = Normal(loc=normalized_dec_x, scale=likelihood_variance*tf.ones([inference_batch_size, img_dim * img_dim])) # Using the L2 loss on Image space for HMC P(X|z)

    qz = Empirical(params=tf.Variable(tf.zeros([T, inference_batch_size, z_dim])))

    # ======================= Pick Model and Initialize ======================= #

    print ("Using " + str(model) + " model. Beginning inference ...")

    if model == 'sghmc':
        inference = ed.SGHMC({z: qz}, data={X: x_gt})

    elif model == 'sgld':
        inference = ed.SGLD({z: qz}, data={X: x_gt})

    elif model == 'metro':
        inference = ed.MetropolisHastings({z: qz}, data={X: x_gt})

    elif model == 'gibbs':
        inference = ed.Gibbs({z: qz}, data={X: x_gt})

    else:  # default to hmc
        inference = ed.HMC({z: qz}, data={X: x_gt})

    if step_size is not None and model == 'sghmc':
        inference.initialize(step_size=step_size, friction=friction)

    elif step_size is not None and model == 'sgld':
        inference.initialize(step_size==step_size)

    elif step_size is not None and model == 'hmc':
        inference.initialize(step_size=step_size, n_steps=leapfrog_steps)

    else:
        inference.initialize()

    # =======================                            ======================= #

    # load model weights to avoid init_uninited_vars()... or do something else
    init_uninited_vars()

    return inference, qz


def run_experiment(P, Q, x_gt, config, DiscL):

    """
        Example configuration
            config = {
            'inference_batch_size' : 1,
            'T' : hmc_steps,
            'img_dim' : 32,
            'step_size' : None,
            'leapfrog_steps' : None,
            'friction' : None,
            'z_dim' : 100,
            'likelihood_variance' : 0.1
        }

        step_size and leapfrog_steps go together for hmc
        step_size and friction go together for sghmc
        step_size for sgld
        you can specify dimension ordering for gibbs
    """

    hmc_steps = config.get('T')  # how many steps to run hmc for, include burn-in steps
    keep_ratio = config.get('keep_ratio')  # keep only last <keep_ratio> percentage of hmc samples (due to burn-in)

    inference, qz = build_experiment(P, Q, x_gt, config, DiscL)

    for _ in range(hmc_steps):
        info_dict = inference.update()
        inference.print_progress(info_dict)

    to_keep_index = int((1 - keep_ratio) * hmc_steps)
    qz_kept = Empirical(qz.params[to_keep_index:])

    return qz, qz_kept


def compare_vae_hmc_loss(P, Q, DiscL, x_gt, samples_to_check, config):
    print ("Starting evaluation...")

    x_samples_to_check = trim_32_to_28(P(samples_to_check)).eval()
    l_th_layer_samples = DiscL(trim_32_to_28(P(samples_to_check)))
    l_th_x_gt = DiscL(x_gt)
    x_samples_to_check = np.expand_dims(x_samples_to_check, 1)
    num_samples = samples_to_check.shape[0]
    img_num = config.get('img_num')
    sample_to_vis = config.get('sample_to_vis')

    best_recon_sample = x_samples_to_check[0]
    best_recon_loss = recon_loss(x_gt, best_recon_sample)
    best_l2_sample = x_samples_to_check[0]
    best_l2_loss = l2_loss(x_gt, best_l2_sample)
    best_latent_sample = x_samples_to_check[0]
    best_latent_loss = l_latent_loss(l_th_x_gt, l_th_layer_samples[0:0+1])

    total_recon_loss = 0.0
    total_l2_loss = 0.0
    total_latent_loss = 0.0

    for i, sample in enumerate(tqdm(x_samples_to_check)):

        for j in range(sample_to_vis):
            plot_save(x_samples_to_check[j], './out/{}_mcmc_sample_{}.png'.format(img_num, j + 1))

        avg_img = np.mean(x_samples_to_check, axis=0)
        plot_save(avg_img, './out/{}_mcmcMean.png'.format(img_num))

        r_loss = recon_loss(x_gt, sample)
        l_loss = l2_loss(x_gt, sample)
        lat_loss = l_latent_loss(l_th_x_gt, l_th_layer_samples[i:i+1])
        total_recon_loss += r_loss
        total_l2_loss += l_loss
        total_latent_loss += lat_loss

        if r_loss < best_recon_loss:
            best_recon_sample = sample
            best_recon_loss = r_loss

        if l_loss < best_l2_loss:
            best_l2_sample = sample
            best_l2_loss = l_loss

        if lat_loss < best_latent_loss:
            best_latent_sample = sample
            best_latent_loss = lat_loss

        # print ("Recon loss: " + str(r_loss))
        # print ("L2 loss: " + str(l_loss))

    average_recon_loss = total_recon_loss/num_samples
    average_l2_loss = total_l2_loss/num_samples
    average_latent_loss = total_latent_loss /num_samples

    vae_recon_loss = recon_loss(x_gt, trim_32_to_28(P(Q(x_gt))))
    vae_l2_loss = l2_loss(x_gt, trim_32_to_28(P(Q(x_gt))))
    vae_latent_loss = l_latent_loss(l_th_x_gt, P(Q(x_gt)))
    print ("---------- Summary Image {} ------------".format(img_num))
    if jernej_Q_P:
        print ("VAE recon loss: " + str(vae_recon_loss))
        print ("VAE L2 loss: " + str(vae_l2_loss))
        print ("VAE latent loss: " + str(vae_latent_loss))
    else:
        print ("VAE recon loss: " + str(recon_loss(x_gt, Q(x_gt)[0], P)))
        print ("VAE L2 loss: " + str(l2_loss(x_gt, Q(x_gt)[0], P)))

    print ("Best mcmc recon loss: " + str(best_recon_loss))
    print ("Best mcmc L2 loss: " + str(best_l2_loss))
    print ("Best mcmc latent loss: " + str(best_latent_loss))
    print ("Average mcmc recon loss: " + str(average_recon_loss))
    print ("Average mcmc l2 loss " + str(average_l2_loss))
    print ("Average mcmc latent loss " + str(average_latent_loss))

    if jernej_Q_P:
        plot_save(tf.reshape(tf.slice(tf.reshape(P(Q(x_gt)), [32, 32]), [2, 2], [28, 28]), [1, 784]).eval(),
                  './out/{}_vae_recon.png'.format(img_num))
        plot_save(best_recon_sample, './out/{}_best_recon.png'.format(img_num))
        plot_save(best_l2_sample, './out/{}_best_l2.png'.format(img_num))
        plot_save(best_latent_sample, './out/{}_best_latent.png'.format(img_num))
    else:
        plot_save(P(Q(x_gt)[0])[0].eval(), './out/{}_vae_recon.png'.format(img_num))
        plot_save(P(best_recon_sample)[0].eval(), './out/{}_best_recon.png'.format(img_num))
        plot_save(P(best_l2_sample)[0].eval(), './out/{}_best_l2.png'.format(img_num))

    return best_recon_loss, average_recon_loss, best_l2_loss, average_l2_loss, best_latent_loss, average_latent_loss,\
           vae_recon_loss, vae_l2_loss, vae_latent_loss


def l2_loss(x_gt, x_hmc):
    if jernej_Q_P:
        return tf.norm(x_gt - x_hmc).eval()
    else:
        return tf.norm(x_gt-x_hmc).eval()


def recon_loss(x_gt, x_hmc):
    if jernej_Q_P:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hmc, labels=x_gt), 1).eval()
    else:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=x_hmc[1], labels=x_gt), 1).eval()


def l_latent_loss(l_th_x_gt, l_th_x_hmc):
    return tf.norm(l_th_x_gt - l_th_x_hmc).eval()


def init_uninited_vars():
    sess = ed.get_session()
    unint_vars = []
    for var in tf.global_variables():
        if not tf.is_variable_initialized(var).eval():
            unint_vars.append(var)
    missingVarInit = tf.variables_initializer(unint_vars)
    sess.run(missingVarInit)

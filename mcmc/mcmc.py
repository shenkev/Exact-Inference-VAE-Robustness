from __future__ import print_function

from edward.models import Empirical, Normal
import edward as ed
import tensorflow as tf

from util import plot, plot_save
import matplotlib.pyplot as plt

jernej_Q_P = True

def build_experiment(P, Q, x_gt, config, DiscL):

    model = config.get('model')
    T = config.get('T')
    img_dim = config.get('img_dim')
    step_size = config.get('step_size')
    leapfrog_steps = config.get('leapfrog_steps')
    friction = config.get('friction')
    z_dim = config.get('z_dim')
    likelihood_variance = config.get('likelihood_variance')
    useDiscL = config.get('useDiscL')

    inference_batch_size = int(x_gt.shape[0])

    z = Normal(loc=tf.zeros([inference_batch_size, z_dim]), scale=tf.ones([inference_batch_size, z_dim]))  # sample z

    if jernej_Q_P:
        normalized_dec_x = P(z)
        normalized_dec_x_trimed = None
        for i in range(normalized_dec_x.shape[0]):
            trimed = tf.reshape(tf.slice(tf.reshape(normalized_dec_x[i], [32, 32]), [2, 2], [img_dim, img_dim]), [-1, 784])
            if normalized_dec_x_trimed is None:
                normalized_dec_x_trimed = trimed
            else:
                normalized_dec_x_trimed = tf.concat([normalized_dec_x_trimed, trimed], 0)
        normalized_dec_x = normalized_dec_x_trimed
    else:
        normalized_dec_x = P(z)[0]

    # Compute lth layer of Disc from decoded image
    if useDiscL:
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
    inference_batch_size = config.get('inference_batch_size')
    z_dim = config.get('z_dim')
    img_dim = config.get('img_dim')

    inference, qz = build_experiment(P, Q, x_gt, config, DiscL)

    for _ in range(hmc_steps):
        info_dict = inference.update()
        inference.print_progress(info_dict)

    to_keep_index = int((1 - keep_ratio) * hmc_steps)
    qz_kept = Empirical(qz.params[to_keep_index:])

    sample_to_vis = 5
    qz_sample = qz_kept.sample(sample_to_vis)

    for k in range(inference_batch_size):
        for i in range(sample_to_vis):
            img = P(tf.reshape(qz_sample[i, k], [1, z_dim]))[0]
            if jernej_Q_P:
                img = tf.reshape(tf.slice(tf.reshape(img, [32, 32]), [2, 2], [img_dim, img_dim]), [1, 784])
            plot_save(img.eval(), './out/{}_mcmc_sample_{}.png'.format(k+1, i+1))
            plot_save(x_gt[k].reshape(1, 784), './out/{}_x_adversarial.png'.format(k+1))

            avg_img = P(tf.reshape(tf.reduce_mean(qz_sample[:, k, :], 0), [1, z_dim]))
            if jernej_Q_P:
                avg_img = tf.reshape(tf.slice(tf.reshape(avg_img, [32, 32]), [2, 2], [img_dim, img_dim]), [1, 784])
            plot_save(avg_img.eval(), './out/{}_mcmcMean.png'.format(k+1))

    return qz, qz_kept


def compare_vae_hmc_loss(P, Q, DiscL, x_gt, qz_kept, config, num_samples=100):

    f = open('log.txt', 'w')

    inference_batch_size = config.get('inference_batch_size')
    z_dim = config.get('z_dim')

    samples_to_check = qz_kept.sample(num_samples).eval()

    for k in range(inference_batch_size):
        print ("------------- Evaluating {}/{} Images --------------".format(k + 1, inference_batch_size))

        best_recon_sample = samples_to_check[0, k:k+1] # k:k+1 hacks the shape to be [1, _]
        best_recon_loss = recon_loss(x_gt[k:k+1], best_recon_sample, P)
        best_l2_sample = samples_to_check[0, k:k+1]
        best_l2_loss = l2_loss(x_gt[k:k+1], best_l2_sample, P)
        best_l_latent_sample = samples_to_check[0, k:k+1]
        best_l_latent_loss = l_latent_loss(x_gt[k:k+1], best_l_latent_sample, P, DiscL)

        total_recon_loss = 0.0
        total_l2_loss = 0.0
        total_l_latent_loss = 0.0

        for i, sample in enumerate(samples_to_check[:, k, :]):

            sample = tf.reshape(sample, [1, z_dim])
            r_loss = recon_loss(x_gt[k:k+1], sample, P)
            l_loss = l2_loss(x_gt[k:k+1], sample, P)
            l_l_loss = l_latent_loss(x_gt[k:k+1], sample, P, DiscL)
            total_recon_loss += r_loss
            total_l2_loss += l_loss
            total_l_latent_loss += l_l_loss

            if r_loss < best_recon_loss:
                best_recon_sample = sample
                best_recon_loss = r_loss

            if l_loss < best_l2_loss:
                best_l2_sample = sample
                best_l2_loss = l_loss

            if l_l_loss < best_l_latent_loss:
                best_l_latent_sample = sample
                best_l_latent_loss = l_l_loss

            # print ("Recon loss: " + str(r_loss))
            # print ("L2 loss: " + str(l_loss))
            # print "------------- Evaluating {}/{} --------------".format(i + 1, num_samples)

        average_recon_loss = total_recon_loss/num_samples
        average_l2_loss = total_l2_loss/num_samples
        average_latent_loss = total_l_latent_loss/num_samples

        print ("---------- Summary Image {} ------------".format(k+1), file=f)
        if jernej_Q_P:
            print ("VAE recon loss: " + str(recon_loss(x_gt[k:k+1], Q(x_gt[k:k+1]), P)), file=f)
            print ("VAE L2 loss: " + str(l2_loss(x_gt[k:k+1], Q(x_gt[k:k+1]), P)), file=f)
            print ("VAE latent loss: " + str(l_latent_loss(x_gt[k:k+1], Q(x_gt[k:k+1]), P, DiscL)), file=f)
        else:
            print ("VAE recon loss: " + str(recon_loss(x_gt, Q(x_gt)[0], P)))
            print ("VAE L2 loss: " + str(l2_loss(x_gt, Q(x_gt)[0], P)))

        print ("Best mcmc recon loss: " + str(best_recon_loss), file=f)
        print ("Best mcmc L2 loss: " + str(best_l2_loss), file=f)
        print ("Best mcmc latent loss: " + str(best_l_latent_loss), file=f)
        print ("Average mcmc recon loss: " + str(average_recon_loss), file=f)
        print ("Average mcmc l2 loss " + str(average_l2_loss), file=f)
        print ("Average mcmc latent loss " + str(average_latent_loss), file=f)

        if jernej_Q_P:
            plot_save(tf.reshape(tf.slice(tf.reshape(P(Q(x_gt[k:k+1])), [32, 32]), [2, 2], [28, 28]), [1, 784]).eval(),
                      './out/{}_vae_recon.png'.format(k+1))
            plot_save(tf.reshape(tf.slice(tf.reshape(P(best_recon_sample), [32, 32]), [2, 2], [28, 28]), [1, 784]).eval(),
                      './out/{}_best_recon.png'.format(k+1))
            plot_save(tf.reshape(tf.slice(tf.reshape(P(best_l2_sample), [32, 32]), [2, 2], [28, 28]), [1, 784]).eval(),
                      './out/{}_best_l2.png'.format(k+1))
            plot_save(tf.reshape(tf.slice(tf.reshape(P(best_l_latent_sample), [32, 32]), [2, 2], [28, 28]), [1, 784]).eval(),
                      './out/{}_best_latent.png'.format(k+1))
        else:
            plot_save(P(Q(x_gt)[0])[0].eval(), './out/vae_recon.png')
            plot_save(P(best_recon_sample)[0].eval(), './out/best_recon.png')
            plot_save(P(best_l2_sample)[0].eval(), './out/best_l2.png')

    f.close()
    # return best_recon_sample, best_recon_loss, average_recon_loss, best_l2_sample, best_l2_loss, average_l2_loss


def l2_loss(x_gt, z_hmc, P):
    if jernej_Q_P:
        return tf.norm(x_gt - tf.reshape(tf.slice(tf.reshape(P(z_hmc), [32, 32]), [2, 2], [28, 28]), [1, 784])).eval()
    else:
        return tf.norm(x_gt-P(z_hmc)[0]).eval()


def recon_loss(x_gt, z_hmc, P):
    if jernej_Q_P:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(
            logits=tf.reshape(tf.slice(tf.reshape(P(z_hmc), [32, 32]), [2, 2], [28, 28]), [1, 784]), labels=x_gt), 1).eval()
    else:
        return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=P(z_hmc)[1], labels=x_gt), 1).eval()


def l_latent_loss(x_gt, z_hmc, P, DiscL):
    normalized_dec_x = P(z_hmc)
    disc_l_normalized_dec_x = DiscL(tf.reshape(tf.slice(tf.reshape(normalized_dec_x, [32, 32]), [2, 2], [28, 28]), [-1, 784]))
    return tf.norm(DiscL(x_gt) - disc_l_normalized_dec_x).eval()


def init_uninited_vars():
    sess = ed.get_session()
    unint_vars = []
    for var in tf.global_variables():
        if not tf.is_variable_initialized(var).eval():
            unint_vars.append(var)
    missingVarInit = tf.variables_initializer(unint_vars)
    sess.run(missingVarInit)

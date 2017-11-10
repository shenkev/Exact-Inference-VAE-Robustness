from edward.models import Empirical, Normal
import edward as ed
import tensorflow as tf

from util import plot
import matplotlib.pyplot as plt


def build_experiment(P, Q, x_gt, config):

    model = config.get('model')
    T = config.get('T')
    img_dim = config.get('img_dim')
    leapfrog_step_size = config.get('leapfrog_step_size')
    leapfrog_steps = config.get('leapfrog_steps')
    friction = config.get('friction')
    z_dim = config.get('z_dim')
    likelihood_variance = config.get('likelihood_variance')

    inference_batch_size = int(x_gt.shape[0])

    z = Normal(loc=tf.zeros([inference_batch_size, z_dim]), scale=tf.ones([inference_batch_size, z_dim]))  # sample z

    normalized_dec_x, dec_x_logits = P(z)

    X = Normal(loc=normalized_dec_x, scale=likelihood_variance*tf.ones([inference_batch_size, img_dim * img_dim]))

    qz = Empirical(params=tf.Variable(tf.zeros([T, inference_batch_size, z_dim])))

    print ("Using " + str(model) + " model. Beginning inference ...")

    if model == 'sghmc':
        inference = ed.SGHMC({z: qz}, data={X: x_gt})

    else:  # default to hmc
        print ("Using HMC model. Beginning inference ...")
        inference = ed.HMC({z: qz}, data={X: x_gt})

    if leapfrog_step_size is not None and model == 'sghmc':
        inference.initialize(step_size=leapfrog_step_size, friction=friction)

    elif leapfrog_step_size is not None and model == 'hmc':
        inference.initialize(step_size=leapfrog_step_size, n_steps=leapfrog_steps)

    else:
        inference.initialize()

    # load model weights to avoid init_uninited_vars()... or do something else
    init_uninited_vars()

    return inference, qz


def run_experiment(P, Q, x_gt, config):

    """
        Example configuration
            config = {
            'inference_batch_size' : 1,
            'T' : hmc_steps,
            'img_dim' : 28,
            'leapfrog_step_size' : None,
            'leapfrog_steps' : None,
            'z_dim' : 100,
            'likelihood_variance' : 0.1
        }
    """

    hmc_steps = config.get('T')  # how many steps to run hmc for, include burn-in steps
    keep_ratio = 0.05  # keep only last <keep_ratio> percentage of hmc samples (due to burn-in)

    inference, qz = build_experiment(P, Q, x_gt, config)

    for _ in range(hmc_steps):
        info_dict = inference.update()
        inference.print_progress(info_dict)

    to_keep_index = int((1 - keep_ratio) * hmc_steps)
    qz_kept = Empirical(qz.params[to_keep_index:])

    sample_to_vis = 5
    qz_sample = qz_kept.sample(sample_to_vis)

    for i in range(sample_to_vis):
        img, _ = P(qz_sample[i])
        # plot(img.eval())

    avg_img, _ = P(tf.reduce_mean(qz_sample, 0))
    plot(avg_img.eval())
    plt.savefig('./out/hmcMean.png'.format(str(i).zfill(3)), bbox_inches='tight')

    return qz, qz_kept

def compare_vae_hmc_loss(P, Q, x_gt, qz_kept, num_samples=100):

    samples_to_check = qz_kept.sample(num_samples).eval()

    best_recon_sample = samples_to_check[0]
    best_recon_loss = recon_loss(x_gt, best_recon_sample, P)
    best_l2_sample = samples_to_check[0]
    best_l2_loss = l2_loss(x_gt, best_l2_sample, P)

    total_recon_loss = 0.0
    total_l2_loss = 0.0

    for sample in samples_to_check:

        r_loss = recon_loss(x_gt, sample, P)
        l_loss = l2_loss(x_gt, sample, P)
        total_recon_loss += r_loss
        total_l2_loss += l_loss

        if r_loss < best_recon_loss:
            best_recon_sample = sample
            best_recon_loss = r_loss

        if l_loss < best_l2_loss:
            best_l2_sample = sample
            best_l2_loss = l_loss

        print ("Recon loss: " + str(r_loss))
        print ("L2 loss: " + str(l_loss))
        print "-------------"

    average_recon_loss = total_recon_loss/num_samples
    average_l2_loss = total_l2_loss/num_samples

    print "---------- Summary ------------"
    print ("VAE recon loss: " + str(recon_loss(x_gt, Q(x_gt)[0], P)))
    print ("VAE L2 loss: " + str(l2_loss(x_gt, Q(x_gt)[0], P)))
    print ("Best hmc recon loss: " + str(best_recon_loss))
    print ("Best hmc L2 loss: " + str(best_l2_loss))
    print ("Average hmc recon loss: " + str(average_recon_loss))
    print ("Average hmc l2 loss " + str(average_l2_loss))

    return best_recon_sample, best_recon_loss, average_recon_loss, best_l2_sample, best_l2_loss, average_l2_loss


def l2_loss(x_gt,z_hmc, P):
    return tf.norm(x_gt-P(z_hmc)[0]).eval()


def recon_loss(x_gt,z_hmc, P):
    return tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(logits=P(z_hmc)[1], labels=x_gt), 1).eval()


def init_uninited_vars():
    sess = ed.get_session()
    unint_vars = []
    for var in tf.global_variables():
        if not tf.is_variable_initialized(var).eval():
            unint_vars.append(var)
    missingVarInit = tf.variables_initializer(unint_vars)
    sess.run(missingVarInit)

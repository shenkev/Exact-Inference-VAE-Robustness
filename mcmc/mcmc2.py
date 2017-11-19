from edward.models import Empirical, Normal
import edward as ed
import tensorflow as tf

"""
    Because I screwed up the first mcmc.py by doing for-loops over samples

"""

jernej_Q_P = True

def trim_32_to_28(input):
    with tf.variable_scope('trim_32_to_28', reuse=True):
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


def init_uninited_vars():
    sess = ed.get_session()
    unint_vars = []
    for var in tf.global_variables():
        if not tf.is_variable_initialized(var).eval():
            unint_vars.append(var)
    missingVarInit = tf.variables_initializer(unint_vars)
    sess.run(missingVarInit)

import edward as ed
import numpy as np
import os
import tensorflow as tf

from edward.models import Normal, Bernoulli
from edward.util import Progbar
from keras.layers import Dense
from observations import mnist
from scipy.misc import imsave

ed.set_seed(42)
data_dir = "./tmp/data"
out_dir = "./tmp/out"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

M = 100  # batch size
d = 2  # number of latent dimensions


def data_iterator(array, batch_size): # array contains the dataset
    start = 0
    while True:
        stop = start + batch_size  # taking the next batch
        diff = stop - array.shape[0]
        if diff <= 0:  # still more left in dataset
            batch = array[start:stop]
            start += batch_size
        else: # ran out
            batch = np.concatenate((array[start:], array[:diff]))
            start = diff
        batch = batch.astype(np.float32)/255.0  # normalize pixels
        batch = np.random.binomial(1, batch)  # make pixels {0, 1}
        yield batch

(x_train, _), (x_test, _) = mnist(data_dir)  # ignore y's in unsupervised
x_data_iter = data_iterator(x_train, M)

# Define decoder model
z = Normal(loc=tf.zeros([M, d]), scale=tf.ones([M, d])) # Prior
hidden = Dense(256, activation='relu')(z.value())
x = Bernoulli(logits=Dense(28*28)(hidden))

# Define encoder model
x_in = tf.placeholder(tf.int32, [M, 28*28])
hidden = Dense(256, activation='relu')(tf.cast(x_in, tf.float32))
qz = Normal(loc=Dense(d)(hidden),
            scale=Dense(d, activation='softplus')(hidden))

inference = ed.KLqp(
{
    z: qz
}, data={x: x_in})  # tie x and x_in variables as well as z and qz

optimizer = tf.train.RMSPropOptimizer(0.01, epsilon=1.0)
inference.initialize(optimizer=optimizer)
tf.global_variables_initializer().run()

epochs = 100
iter_per_epoch = x_train.shape[0] // M

for epoch in range(1, epochs+1):
    print("Epoch: {0}".format(epoch))
    avg_loss = 0.0
    
    pbar = Progbar(iter_per_epoch)
    for t in range(1, iter_per_epoch+1):
        pbar.update(t)
        x_batch = next(x_data_iter)
        info_dict = inference.update(feed_dict={x_in: x_batch})
        avg_loss += info_dict['loss']
    
    avg_loss = avg_loss / iter_per_epoch
    avg_loss = avg_loss / M
    print("-log p(x) <= {:0.3f}".format(avg_loss))

    # Prior predictive check.
    images = x.eval()  # x is the reconstructed image
    for m in range(M):
        imsave(os.path.join(out_dir, '%d.png') % m, images[m].reshape(28, 28))


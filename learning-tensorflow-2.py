from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import os
import _pickle as cPickle

STEPS = 5000
BATCH_SIZE = 100

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                          strides=[1, 2, 2, 1], padding='SAME')

def conv_layer(input, shape):
    W = weight_variable(shape)
    b = bias_variable([shape[3]])
    return tf.nn.relu(conv2d(input, W) + b)

def full_layer(input, size):
    in_size = int(input.get_shape()[1])
    W = weight_variable([in_size, size])
    b = bias_variable([size])
    return tf.matmul(input, W) + b


class CifarLoader(object):
    def __init__(self, source_files):
        self._source = source_files
        self._i = 0
        self.images = None
        self.labels = None
        self.data_path = "CIFAR10/cifar-10-batches-py"

    def load(self):
        data = [self.unpickle(f) for f in self._source]
        images = np.vstack([d["data"] for d in data])
        n = len(images)
        self.images = images.reshape([n, 3, 32, 32]).transpose(0,2,3,1).astype(float) / 255
        self.labels = self.one_hot(np.hstack([d["labels"] for d in data]), 10)
        return self

    def next_batch(self, batch_size):
        x, y = (self.images[self._i:self._i + batch_size],
                self.labels[self._i:self._i + batch_size])
        self._i = (self._i + batch_size) % len(self.images)
        return x, y

    def unpickle(self,file):
        with open(os.path.join(self.data_path, file), 'rb') as fo:
            dict = cPickle.load(fo, encoding='latin1')
        return dict

    def one_hot(self, vec, vals=10):
        n = len(vec)
        out = np.zeros((n, vals))
        out[range(n), vec] = 1
        return out

class CifarDataManager(object):
    def __init__(self):
        self.train = CifarLoader(
                ["data_batch_{}".format(i) for i in range(1,6)]
            ).load()
        self.test = CifarLoader(
                ["test_batch"]
            ).load()

def display_cifar(images, size):
    n = len(images)
    plt.figure()
    plt.gca().set_axis_off()
    im = np.vstack([np.hstack([images[np.random.choice(n)] for i in range(size)])
                    for i in range(size)])
    plt.imshow(im)
    plt.show()


cifar = CifarDataManager()

x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

keep_prob = tf.placeholder(tf.float32)

conv1 = conv_layer(x, shape=[5,5,3,32])
conv1_pool = max_pool_2x2(conv1)

conv2 = conv_layer(conv1_pool, shape=[5,5,32,64])
conv2_pool = max_pool_2x2(conv2)

conv2_flat = tf.reshape(conv2_pool, [-1,8*8*64])

full_1 = tf.nn.relu(full_layer(conv2_flat, 1024))
full1_drop = tf.nn.dropout(full_1, keep_prob=keep_prob) 

y_conv = full_layer(full1_drop, 10)
cross_entropy = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(
            labels=y_,logits=y_conv))

train_step = tf.compat.v1.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(
        tf.argmax(y_conv, axis = 1), tf.argmax(
            y_, axis = 1))

accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def test(sess):
    X = cifar.test.images.reshape(10, 1000, 32, 32, 3)
    Y = cifar.test.labels.reshape(10, 1000, 10)
    acc = np.mean([sess.run(accuracy, feed_dict={x: X[i], y_: Y[i], 
                                                 keep_prob: 1.0})
                  for i in range(10)])
    print("test accuracy: {:.4}%".format(acc * 100))

with tf.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    for i in range(STEPS):
        batch = cifar.train.next_batch(BATCH_SIZE)
        if i % 100 == 0:
            train_accuracy = sess.run(accuracy, feed_dict={
                x: batch[0],
                y_: batch[1],
                keep_prob: 1.0})
            print("Step: {}, test accuracy: {:.4}%".format(i, train_accuracy * 100))
        sess.run(train_step, feed_dict={
            x: batch[0], 
            y_: batch[1], 
            keep_prob: 0.5})

    test(sess)



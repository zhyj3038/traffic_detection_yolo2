import tensorflow as tf
import numpy as np
import pandas as pd
import architecture
from utils import get_batches
from utils import plot_predictions
import sys
import os


data_path = 'combined_dataset/'
data = pd.read_csv(data_path+'labels_combined_complete.csv')
anchors = np.load(data_path+'anchors.npy')


input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
train_flag = tf.placeholder(dtype=tf.bool)
labels = tf.placeholder(tf.float32, shape=(None, 13, 13, 5, 7))
mask = tf.placeholder(tf.bool, shape=(None, 13, 13, 5))
l_rate = tf.placeholder(tf.float32, shape=())

head = architecture.build_graph(input_tensor, train_flag, 32, 5, 3)
pred = architecture.yolo_prediction(head)
loss = architecture.yolo_loss(labels, pred, mask)

optimizer = tf.train.AdamOptimizer(learning_rate=l_rate)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

saver = tf.train.Saver()

batch_size = 8
batches_per_epoch = len(os.listdir(data_path+'images_small')) // batch_size + 1
epochs = 2
learning_rate = 1e-4

with tf.Session() as sess:
    saver.restore(sess, 'model_weights/new_model_4.ckpt')
    for epoch in range(epochs):
        gen = get_batches(data, data_path+'images_small/', batch_size)

        #if epoch % 5 == 0:
        #    learning_rate /= 2
        print('Start training epoch {} with learning rate {}'.format(epoch, learning_rate))
        idx = 0
        for X, y, msk in gen:
            _, batch_loss = sess.run([train_op, loss], feed_dict={input_tensor: X, train_flag: True, labels: y,
                                                                  mask: msk, l_rate: learning_rate})
            message = "Epoch {}/{}: Training batch {}/{} Current loss is: {}\r".format(epoch, epochs, idx,
                                                                                       batches_per_epoch, batch_loss)
            idx += 1
            sys.stdout.write(message)

        saver.save(sess, 'model_weights/new_model_{}.ckpt'.format(epoch))

    y_hat = sess.run(pred, feed_dict={input_tensor: X, train_flag: False})


# plot predictions after training
threshold = 0.6
plot_predictions(y_hat, X, anchors, threshold)




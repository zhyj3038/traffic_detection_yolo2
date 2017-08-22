import tensorflow as tf
import numpy as np
import pandas as pd
import architecture
from utils import get_batches
import sys
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-e', '--epochs', type=int, default=2)
parser.add_argument('-lr', '--lrate', type=float, default=1e-3)
parser.add_argument('-bs', '--batchsize', type=int, default=8)
args = parser.parse_args()


data_path = 'combined_dataset/'
train_data = pd.read_csv(data_path+'train_lbl.csv')
val_data = pd.read_csv(data_path+'validation_lbl.csv')
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

batch_size = args.batchsize
if len(os.listdir(os.path.join(data_path, 'train'))) % batch_size == 0:
    batches_per_epoch = len(os.listdir(os.path.join(data_path, 'train'))) / batch_size
else:
    batches_per_epoch = len(os.listdir(os.path.join(data_path, 'train'))) // batch_size + 1

if len(os.listdir(os.path.join(data_path, 'validation'))) % batch_size == 0:
    val_batches = len(os.listdir(os.path.join(data_path, 'validation'))) / batch_size
else:
    val_batches = len(os.listdir(os.path.join(data_path, 'validation'))) // batch_size + 1


epochs = args.epochs
learning_rate = args.lrate

with tf.Session() as sess:
    train_loss = []
    val_loss = []

    #sess.run(tf.global_variables_initializer())
    saver.restore(sess, 'model_weights/new_model_30.ckpt')
    for epoch in range(1, epochs+1):
        gen = get_batches(train_data, data_path+'train/', batch_size)

        if epoch % 5 == 0 and epoch > 0:
            learning_rate /= 2
        print('Start training epoch {} with learning rate {}'.format(epoch, learning_rate))
        idx = 1

        for X, y, msk in gen:
            _, batch_loss = sess.run([train_op, loss], feed_dict={input_tensor: X, train_flag: True, labels: y,
                                                                  mask: msk, l_rate: learning_rate})
            message = "Epoch {}/{}: Training batch {}/{} Current loss is: {}\r".format(epoch, epochs, idx,
                                                                                       batches_per_epoch, batch_loss)
            idx += 1
            sys.stdout.write(message)

        saver.save(sess, 'model_weights/new_model_{}.ckpt'.format(epoch))

        # evaluate training and validation loss after every epoch
        train_gen = get_batches(train_data, data_path+'train/', batch_size)

        cum_train_loss = 0

        batch_cnt = 1
        for X, y, msk in train_gen:
            train_batchloss = sess.run(loss, {input_tensor: X, train_flag: False, labels: y, mask: msk})
            cum_train_loss += train_batchloss / batches_per_epoch

        val_gen = get_batches(val_data, data_path+'validation/', batch_size)
        cum_val_loss = 0
        for X, y, msk in val_gen:
            val_batchloss = sess.run(loss, {input_tensor: X, train_flag: False, labels: y, mask: msk})
            cum_val_loss += val_batchloss / val_batches

        train_loss.append(cum_train_loss)
        val_loss.append(cum_val_loss)
        print()
        print('='*50)
        print('Training loss after epoch {}/{} is: {}'.format(epoch, epochs, train_loss[-1]))
        print('Validation loss after epoch {}/{} is: {}'.format(epoch, epochs, val_loss[-1]))
        print('=' * 50)

    np.save('train_loss', np.array(train_loss))
    np.save('val_loss', np.array(val_loss))

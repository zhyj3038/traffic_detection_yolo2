import tensorflow as tf
import numpy as np
import pandas as pd
import architecture
from utils import get_batches
import sys
import os


data_path = 'combined_dataset/'
train_data = pd.read_csv(data_path+'train_lbl.csv')
val_data = pd.read_csv(data_path+'validation_lbl.csv')
anchors = np.load(data_path+'anchors.npy')


input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
train_flag = tf.placeholder(dtype=tf.bool)
labels = tf.placeholder(tf.float32, shape=(None, 13, 13, 5, 7))
mask = tf.placeholder(tf.bool, shape=(None, 13, 13, 5))

head = architecture.build_graph(input_tensor, train_flag, 32, 5, 3)
pred = architecture.yolo_prediction(head)
loss = architecture.yolo_loss(labels, pred, mask)

saver = tf.train.Saver()

batch_size = 20
if len(os.listdir(os.path.join(data_path, 'train'))) % batch_size == 0:
    train_batches = len(os.listdir(os.path.join(data_path, 'train'))) / batch_size
else:
    train_batches = len(os.listdir(os.path.join(data_path, 'train'))) // batch_size + 1

if len(os.listdir(os.path.join(data_path, 'validation'))) % batch_size == 0:
    val_batches = len(os.listdir(os.path.join(data_path, 'validation'))) / batch_size
else:
    val_batches = len(os.listdir(os.path.join(data_path, 'validation'))) // batch_size + 1

with tf.Session() as sess:

    saver.restore(sess, 'model_weights/new_model_1.ckpt')

    train_gen = get_batches(train_data, data_path+'train/', batch_size)
    idx = 1
    cum_train_loss = 0
    for X, y, msk in train_gen:
        batch_loss = sess.run(loss, feed_dict={input_tensor: X, train_flag: False, labels: y,
                                                              mask: msk})
        cum_train_loss += batch_loss
        message = "Evaluating training batch {}/{}\r".format(idx, train_batches)
        idx += 1
        sys.stdout.write(message)


    # evaluate training and validation loss after every epoch
    val_gen = get_batches(train_data, data_path+'validation/', batch_size)

    cum_val_loss = 0
    idx = 1
    for X, y, msk in val_gen:
        val_batchloss = sess.run(loss, {input_tensor: X, train_flag: False, labels: y, mask: msk})
        cum_val_loss += val_batchloss
        message = "Evaluating validation batch {}/{}\r".format(idx, val_batches)
        idx += 1
        sys.stdout.write(message)

    print()
    print('='*50)
    print('Training loss is: {}'.format(cum_train_loss / train_batches))
    print('Validation loss is: {}'.format(cum_val_loss / val_batches))
    print('=' * 50)

    #np.save('train_loss', np.array(train_loss))
    #np.save('val_loss', np.array(val_loss))

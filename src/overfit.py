import tensorflow as tf
import numpy as np
import pandas as pd
import architecture
from utils import get_batches
from utils import plot_predictions
import matplotlib.pyplot as plt
import tqdm

data = pd.read_csv('labels_complete.csv')
anchors = np.load('anchors.npy')


input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
train_flag = tf.placeholder(dtype=tf.bool)
labels = tf.placeholder(tf.float32, shape=(None, 13, 13, 5, 7))
mask = tf.placeholder(tf.bool, shape=(None, 13, 13, 5))

head = architecture.build_graph(input_tensor, train_flag, 32, 5, 3)
pred = architecture.yolo_prediction(head)
loss = architecture.yolo_loss(labels, pred, mask)

optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)

saver = tf.train.Saver()

# overfit to one image
gen = get_batches(data, batch_size=1)
x, y, msk = next(gen)

with tf.Session() as sess:
    saver.restore(sess, 'model/new_model.ckpt')
    for i in tqdm.tqdm(range(100)):
        sess.run(train_op, feed_dict={input_tensor: x, train_flag: True, labels: y, mask: msk})

    saver.save(sess, 'model/overfit.ckpt')

    y_hat = sess.run(pred, feed_dict={input_tensor: x, train_flag: False})
    threshold = 0.5
    for i in range(5):
        plot_predictions(y_hat, x.copy(), anchors, threshold+i*0.1)




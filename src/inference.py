"""
Script uses a pretrained model, performs inference and saves the
images with predicted bounding boxes to disk using threads
"""


import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import queue
import threading
from process_predictions import save_inference_images, save_heatmaps
import architecture
import argparse
import sys


def inference_batches(path, batch_size=20):
    fnames = os.listdir(path)
    if len(fnames) % batch_size == 0:
        n_batches = len(fnames) / batch_size
    else:
        n_batches = len(fnames) // batch_size + 1

    for b_idx in range(int(n_batches)):
        images = np.array([plt.imread(path + fname) for
                           fname in fnames[b_idx * batch_size:b_idx * batch_size + batch_size]])
        names = np.array([fname for fname in fnames[b_idx * batch_size:b_idx * batch_size + batch_size]])

        yield images, names


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('srcpath', type=str, help='Path to image for inference')
    parser.add_argument('-bs', '--batchsize', type=int, help='Batchsize for inference', default=20)
    parser.add_argument('-o', '--outpath', type=str, help='Path to save predicted images', default='./predictions')
    parser.add_argument('-t', '--threshold', type=float, default=0.6)
    parser.add_argument('-ap', '--anchorpath', type=str, default='./anchors.npy')
    parser.add_argument('-oip', '--originalimages', type=str,
                        help='Path to images in original resolution. If provided, bounding boxes will be '
                             'drawn on these images. Images must have same name.', default=None)
    parser.add_argument('-pl', '--plots', type=str, default='boxes', help='Defines the output plots. Either boxes or heatmaps')
    args = parser.parse_args()

    # are used for plotting, although not passed to function directly
    anchors = np.load(args.anchorpath)
    threshold = args.threshold

    input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
    train_flag = tf.placeholder(dtype=tf.bool)

    head = architecture.build_graph(input_tensor, train_flag, 32, 5, 3)
    pred = architecture.yolo_prediction(head)

    saver = tf.train.Saver()

    # using multithreading to process the predictions in parallel
    # massively speeds up inference time
    q = queue.Queue()
    num_threads = 20

    with tf.Session() as sess:
        saver.restore(sess, 'model_weights/new_model_30.ckpt')

        gen = inference_batches(args.srcpath, args.batchsize)

        # only used for CLI info here
        fnames = os.listdir(args.srcpath)
        if len(fnames) % args.batchsize == 0:
            n_batches = int(len(fnames) / args.batchsize)
        else:
            n_batches = int(len(fnames) // args.batchsize + 1)

        count = 0
        for images, names in gen:
            y_hat = sess.run(pred, {input_tensor: images, train_flag: False})
            message = "Processing batch {}/{}\r".format(count, n_batches)
            count += 1
            sys.stdout.write(message)

            if args.originalimages is not None:
                images = np.array([plt.imread(os.path.join(args.originalimages, fn)) for fn in names])

            # feed queue for threads
            for Xy in zip(images, y_hat, names):
                q.put(Xy)


            # deploy threads on queue
            for i in range(num_threads):
                if args.plots == 'boxes':
                    worker = threading.Thread(target=save_inference_images,
                                              args=(q, args.outpath, anchors, threshold))
                elif args.plots == 'heatmaps':
                    worker = threading.Thread(target=save_heatmaps,
                                              args=(q, args.outpath))
                else:
                    raise ValueError("Specify output either to boxes or heatmaps")
                worker.setDaemon(True)
                worker.start()

            q.join()


if __name__ == '__main__':
    main()

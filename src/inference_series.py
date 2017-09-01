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
from natsort import natsorted


def inference_batches(path, batch_size=20):
    fnames = natsorted(os.listdir(path))
    if len(fnames) % batch_size == 0:
        n_batches = len(fnames) / batch_size
    else:
        n_batches = len(fnames) // batch_size + 1

    for b_idx in range(int(n_batches)):
        images = np.array([plt.imread(path + fname) for
                           fname in fnames[b_idx * batch_size:b_idx * batch_size + batch_size]])
        names = np.array([fname for fname in fnames[b_idx * batch_size:b_idx * batch_size + batch_size]])

        yield images, names


def average(pred, beta, diff_thresh=0.3):
    """
    exponentially weighted moving average to smooth predictions

    :param pred:
    :param beta:
    :param diff_thresh:
    :return:
    """
    pred[:, :7, :, :, :] = 0
    base = np.zeros([13, 13, 5, 8])
    weighted_preds = np.zeros(pred.shape)
    for idx in range(len(pred)-1):
        if idx == 0:
            weighted_preds[idx] = (beta * base + (1 - beta) * pred[idx]) / (1 - beta ** (idx + 1))
        else:
            weighted_preds[idx] = (beta * weighted_preds[idx - 1] + (1 - beta) * pred[idx] ) / (1 - beta ** (idx + 1))

    # prevent objectness scores from decreasing too abruptly
    new_pred = weighted_preds.copy()
    for idx in range(1, len(weighted_preds)):
        diff = weighted_preds[idx] - weighted_preds[idx - 1]
        mask = diff[..., 0] < -diff_thresh
        new_pred[idx, mask, 0] = weighted_preds[idx - 1, mask, 0]

    # if 3 or more frames out of the previous and following frames contain an object, assume there is an object, even
    # if prediction misses it
    new_pred = weighted_preds.copy()
    window_size = 18
    for idx in range(window_size, len(weighted_preds)-window_size):

        window = np.stack([new_pred[idx-i, :, :, :, 0] for i in range(-window_size, 0)], axis=-1)
        window = np.sum(window > 0.6, axis=-1)
        mask = window >= 6
        new_pred[idx, mask, 0] = 1
        new_pred[idx, ~mask, 0] = 0

    return new_pred


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

        count = 1
        predictions = []
        im_names = []
        for images, names in gen:
            predictions.append(sess.run(pred, {input_tensor: images, train_flag: False}))
            im_names.append(names)
            message = "Processing batch {}/{}\r".format(count, n_batches)
            count += 1
            sys.stdout.write(message)

    predictions = np.array(predictions).reshape([-1, 13, 13, 5, 8])
    im_names = np.array(im_names)#.reshape([-1, 1])

    for beta_val in [0.7]:
        weighted_pred = average(predictions, beta_val)
        save_path = os.path.join(args.outpath, 'beta_{}'.format(beta_val))
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        for idx in range(n_batches):

            if args.originalimages is not None:
                images = np.array([plt.imread(os.path.join(args.originalimages, fn)) for fn in im_names[idx]])

            # feed queue for threads
            for Xy in zip(images, weighted_pred[idx*args.batchsize:idx*args.batchsize + args.batchsize], im_names[idx]):
                q.put(Xy)


            # deploy threads on queue
            for i in range(num_threads):
                if args.plots == 'boxes':
                    worker = threading.Thread(target=save_inference_images,
                                              args=(q, save_path, anchors, threshold))
                elif args.plots == 'heatmaps':
                    worker = threading.Thread(target=save_heatmaps,
                                              args=(q, args.outpath))
                else:
                    raise ValueError("Specify output either to boxes or heatmaps")
                worker.setDaemon(True)
                worker.start()

                q.join()
#
#
if __name__ == '__main__':
    main()

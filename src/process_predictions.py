import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
from skimage.io import imsave
import cv2


def convert_coord(xc, yc, w, h):
    """converts coordinate format from center, w,h to two points"""
    x1 = xc - (w - 1) / 2
    y1 = yc - (h - 1) / 2
    x2 = xc + (w - 1) / 2
    y2 = yc + (h - 1) / 2

    return np.int32(np.stack([x1, y1])).T, np.int32(np.stack([x2, y2]).T)


def convert_pred2coord(prediction, anchors, threshold):
    """
    converts relative yolo prediction coordinates to
    absolute coordinates for all predictions above the
    threshold

    :param prediction: np.array with dim batch_size*13*13*5*8
    :param anchors: np.array with shape 5*2 (width, height)
    :param threshold: float, to filter predictions
    :return:
    """
    indices = np.where(prediction[..., 0] > threshold)

    y_grid = indices[0]
    x_grid = indices[1]
    anchor_idx = indices[2]

    x_center = x_grid * 32 + prediction[indices][:, 1] * 32
    y_center = y_grid * 32 + prediction[indices][:, 2] * 32

    width = anchors[anchor_idx][:, 0] * prediction[indices][:, 3]
    height = anchors[anchor_idx][:, 1] * prediction[indices][:, 4]

    p1, p2 = convert_coord(x_center, y_center, width, height)

    return p1, p2


def draw_box(img, points):
    for i in range(points[0].shape[0]):
        cv2.rectangle(img, tuple(points[0][i]), tuple(points[1][i]), (255, 0, 0), thickness=4)

    return img


def plot_predictions(pred, X, anchors, threshold):
    """Plots 9 predictions"""
    image_array = np.zeros([9, *X.shape[1:]], dtype='uint8')
    for idx in range(9):
        pp = convert_pred2coord(pred[idx], anchors, threshold)
        image_array[idx] = draw_box(X[idx].copy(), pp)

    fig, axs = plt.subplots(3, 3, figsize=(15, 8))

    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(image_array[idx])
        ax.axis('off')


def draw_boxes(img, predictions, anchors, threshold):
    """Draws boxes based on predictions and returns the drawn image"""
    p1, p2 = convert_pred2coord(predictions, anchors, threshold)
    indices = np.where(predictions[..., 0] > threshold)
    cls_labels = {0: 'car', 1: 'pedestrian', 2: 'truck'}
    classes = np.argmax(predictions[indices][:, 5:], axis=1)
    class_color = {'car': (255, 0, 0), 'pedestrian': (0, 255, 0), 'truck': (0, 0, 255)}

    # check if boxes need to be rescaled if images have another resolution than
    # net input
    if img.shape[0:2] != (416, 416):
        w_factor = img.shape[1] / 416
        h_factor = img.shape[0] / 416
        p1[:, 0] = p1[:, 0] * w_factor
        p2[:, 0] = p2[:, 0] * w_factor

        p1[:, 1] = p1[:, 1] * h_factor
        p2[:, 1] = p2[:, 1] * h_factor

    i = 0
    for p1_, p2_ in zip(p1, p2):
        cv2.rectangle(img, tuple(p1_), tuple(p2_),
                      class_color[cls_labels[classes[i]]], thickness=3)
        i += 1

    return img


def save_inference_images(q, path, anchors, threshold):
    """
    Uses threading to draw and save predictions

    :param q: queue. q contains tuples with (img, prediction, filename)
    :param path: path where to save images
    :param anchors: np.array with anchors (width, height)
    :param threshold: float
    :return: nothing
    """
    while not q.empty():
        img, y, name = q.get()
        imsave(os.path.join(path, name), draw_boxes(img, y, anchors, threshold))
        q.task_done()


def frames2video(path):
    """
    Merges images in path into a video

    :param path: path with prediction images
    :return: nothing
    """
    fnames = os.listdir(path)
    fnames.sort()
    images = np.array([plt.imread(os.path.join(path, fname)) for fname in fnames])
    # h, w, c = images[0].shape
    videowriter = imageio.get_writer('prediction_video.mp4', fps=25)

    for im in images:
        videowriter.append_data(im)
    videowriter.close()


def create_heatmaps(img, pred):
    """
    Uses objectness probability to draw a heatmap on the image and returns it
    """
    # find anchors with highest prediction
    best_pred = np.max(pred[..., 0], axis=-1)
    # convert probabilities to colormap scale
    best_pred = np.uint8(best_pred * 255)
    # apply color map
    # cv2 colormaps create BGR, not RGB
    cmap = cv2.cvtColor(cv2.applyColorMap(best_pred, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    # resize the color map to fit image
    cmap = cv2.resize(cmap, img.shape[1::-1], interpolation=cv2.INTER_NEAREST)

    # overlay cmap with image
    return cv2.addWeighted(cmap, 1, img, 0.5, 0)


def save_heatmaps(q, path):
    """
    Threaded function which is called from the inference script
    :param q: Queue instance
    :param path: image path
    :return: nothing
    """
    while not q.empty():
        img, y, name = q.get()
        imsave(os.path.join(path, name), create_heatmaps(img, y))
        q.task_done()
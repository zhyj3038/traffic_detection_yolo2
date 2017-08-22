import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def convert_bbox_format(series):
    """
    Converts pandas series with x_center, y_center, width and height values to
    x1,y1,x2,y2 pairs for plotting
    :param series:
    :return:
    """

    x_pt1 = np.int32(series['x_center'] - (series['width'] - 1) / 2)
    x_pt2 = np.int32(series['x_center'] + (series['width'] - 1) / 2)
    y_pt1 = np.int32(series['y_center'] - (series['height'] - 1) / 2)
    y_pt2 = np.int32(series['y_center'] + (series['height'] - 1) / 2)

    return (x_pt1, y_pt1), (x_pt2, y_pt2)


def plot_boxes(df):
    """
    Takes in a dataframe, selects 9 random rows and plots images
    with bounding boxes

    :param df:
    :return:
    """

    fig, axs = plt.subplots(3, 3, figsize=(15, 15))

    # load random images from frame and plot them with bboxes
    random_idxs = np.random.choice(np.arange(len(df)), size=9, replace=False)
    # print(random_idxs)
    for idx, ax in enumerate(axs.flatten()):
        ridx = random_idxs[idx]
        img = plt.imread('image_data/' + df.iloc[ridx]['Frame'])
        img = cv2.resize(img, (608, 608))

        # find all objects in random image
        sub_df = df[df['Frame'] == df.iloc[ridx]['Frame']]

        for objct in range(sub_df.shape[0]):
            pt1, pt2 = convert_bbox_format(sub_df.iloc[objct])
            cv2.rectangle(img, pt1, pt2, (255, 0, 0), thickness=2)
        ax.imshow(img)
        ax.axis('off')


def plot_cluster(X, clusterer):
    """
    Plots the results of clustering

    :param X: np.array features, 2D
    :param clusterer: e.g. kmeans instance, must have .labels_ attribute
    :return: Nothing
    """
    x = X[:, 0]  # aspect ratio
    y = X[:, 1]  # area

    plt.scatter(x, y, c=clusterer.labels_)
    plt.xlabel('width')
    plt.ylabel('height')


def plot_anchors(anchors):
    """
    Creates a plot of anchors
    :param anchors: np.array 2D
    :return: Nothing
    """
    img = np.zeros([400, 400])
    for anchor in anchors:
        print(tuple(anchor.astype('int')))
        cv2.rectangle(img, (0, 0), tuple(anchor.astype('int')), 255, thickness=2)

    plt.imshow(img, cmap='gray')


def get_IOU(rec1, rec2):
    """
    rec1&2 are both np.arrays with x_center, y_center, width, height
    should work with any dimension as long as the last dimension is 4
    """

    rec1_xy_max = rec1[..., :2] + (rec1[..., 2:4] - 1) / 2
    rec1_xy_min = rec1[..., :2] - (rec1[..., 2:4] - 1) / 2

    rec2_xy_max = rec2[..., :2] + (rec2[..., 2:4] - 1) / 2
    rec2_xy_min = rec2[..., :2] - (rec2[..., 2:4] - 1) / 2

    intersec_max = np.minimum(rec1_xy_max, rec2_xy_max)
    intersec_min = np.maximum(rec1_xy_min, rec2_xy_min)

    intersec_wh = np.maximum(intersec_max - intersec_min + 1, 0)

    intersec_area = intersec_wh[..., 0] * intersec_wh[..., 1]

    area1 = rec1[..., 2] * rec1[..., 3]
    area2 = rec2[..., 2] * rec2[..., 3]

    union = area1 + area2 - intersec_area

    return intersec_area / union


def set_responsibilities(anchor_frames, iou_thresh=0.6):
    """
    Changes the IOU values for the anchor frames to binary values

    anchor_frames: list of frames where each frame contains all features for a specific anchor
    iou_thresh: threshold to decide which anchor is responsible
    """
    # set box with maximum IOU to 1
    anchor_frames = [frame.copy() for frame in anchor_frames]
    # find maximum IOU value over all frames
    helper_array = np.array([frame[frame.columns[0]] for frame in anchor_frames]).T
    max_indices = np.argmax(helper_array, axis=1)
    data_idx = np.arange(len(max_indices))
    for obj_idx, frame_idx in zip(data_idx, max_indices):
        temp_frame = anchor_frames[frame_idx]
        temp_frame.loc[obj_idx, temp_frame.columns[0]] = 1

    # applying the iou threshold on a copy of the dataframes
    for frame in anchor_frames:
        frame[frame.columns[0]] = np.digitize(frame[frame.columns[0]], [iou_thresh])

    return anchor_frames


def get_batches(df, image_path, batch_size=16):
    """
    Takes in a dataframe and returns X,y as well as the mask
    """

    # group objects by images and ignore if more than one object per anchor
    grouped = df[df['pot_conflict'] == False].groupby('Frame')
    # list all image filenames
    fnames = np.array(os.listdir(image_path))
    fnames_w_obj = [name for name, _ in grouped]

    # create two lists of frame. labels frame contains training labels
    # mask_vals contains frames with grid coordinates for each image with objects
    labels = {img_name: frame[['xc_rel', 'yc_rel', 'w_train', 'h_train',
                               'Car', 'Pedestrian', 'Truck']].as_matrix() for img_name, frame in grouped}

    mask_vals = {img_name: frame[['y_grid_idx', 'x_grid_idx', 'resp_anchor']].as_matrix().T \
                 for img_name, frame in grouped}

    batches_per_epoch = len(fnames) // batch_size + 1
    for batch_idx in range(batches_per_epoch):
        # select random indices for each batch
        # random_indices = np.random.choice(np.arange(len(fnames)), size=batch_size, replace=False)

        fnames_batch = np.random.choice(fnames, size=batch_size, replace=False)
        # load all images for the batch
        X = np.array([plt.imread(image_path + file) for file in fnames_batch])

        # dim: batch_size * gridx *  gridy * n_anchors
        mask = np.zeros([batch_size, 13, 13, 5])
        # dim: batch_size * gridx *  gridy * n_anchors * (4 + n_classes)
        y = np.zeros([batch_size, 13, 13, 5, 7])

        for count, img_name in enumerate(fnames_batch):
            # checks if frame is in dataframe, is only there if image has objects
            if img_name in fnames_w_obj:
                # handles all objects in an image at once by indexing with arrays
                # create boolean mask
                m0, m1, m2 = mask_vals[img_name]
                mask[count, m0, m1, m2] = 1
                # create labels
                y[count, m0, m1, m2] = labels[img_name]

        # make mask boolean
        mask = mask > 0

        yield X, y, mask


def convert_coord(xc, yc, w, h):
    x1 = xc - (w - 1) / 2
    y1 = yc - (h - 1) / 2
    x2 = xc + (w - 1) / 2
    y2 = yc + (h - 1) / 2

    return np.int32(np.stack([x1, y1])).T, np.int32(np.stack([x2, y2]).T)


def convert_pred2coord(prediction, anchors, threshold):
    indices = np.where(prediction[..., 0] > threshold)

    y_grid = indices[1]
    x_grid = indices[2]
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
    image_array = np.zeros([9, *X.shape[1:]], dtype='uint8')
    for idx in range(9):
        pp = convert_pred2coord(pred[idx], anchors, threshold)
        image_array[idx] = draw_box(X[idx].copy(), pp)

    fig, axs = plt.subplots(3, 3, figsize=(15, 8))

    for idx, ax in enumerate(axs.flatten()):
        ax.imshow(image_array[idx])
        ax.axis('off')
    plt.show()
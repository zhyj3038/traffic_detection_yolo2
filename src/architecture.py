import tensorflow as tf
import numpy as np

def get_block(inp_tensor, filters, is_training, alpha=0.1, kernel=(3, 3), maxpool=False):
    """
    Returns a block with convolution, batchnorm and leaky relu.

    is_training : tf.placeholder as flag for batchnorm
    alpha : usually 0.1
    """

    x = tf.layers.conv2d(inp_tensor, filters, kernel, padding='same', activation=None)
    x = tf.layers.batch_normalization(x, training=is_training)
    x = tf.maximum(x, alpha * x)
    if maxpool:
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same')

    return x


def build_graph(input_tensor, train_flag, start_filter_size, n_anchors, n_classes):
    # preprocessing
    mean = tf.constant(np.load('rgb_mean.npy'), dtype=tf.float32)
    x = (input_tensor - mean) / 255

    with tf.name_scope('Block_1'):
        x = get_block(x, start_filter_size, train_flag, maxpool=True)

    with tf.name_scope('Block_2'):
        x = get_block(x, start_filter_size * 2 ** 1, train_flag, maxpool=True)

    with tf.name_scope('bigBlock_1'):
        x = get_block(x, start_filter_size * 2 ** 2, train_flag)
        x = get_block(x, start_filter_size * 2 ** 1, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 2, train_flag, maxpool=True)

    with tf.name_scope('bigBlock_2'):
        x = get_block(x, start_filter_size * 2 ** 3, train_flag)
        x = get_block(x, start_filter_size * 2 ** 2, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 3, train_flag, maxpool=True)

    with tf.name_scope('doubleBigBlock_1'):
        x = get_block(x, start_filter_size * 2 ** 4, train_flag)
        x = get_block(x, start_filter_size * 2 ** 3, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 4, train_flag)
        x = get_block(x, start_filter_size * 2 ** 3, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 4, train_flag)

    with tf.name_scope('passThrough'):
        y = get_block(x, start_filter_size * 2 ** 1, train_flag)
        y = tf.space_to_depth(y, 2)

    with tf.name_scope('doubleBigBlock_2'):
        x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same')
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)
        x = get_block(x, start_filter_size * 2 ** 4, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)
        x = get_block(x, start_filter_size * 2 ** 4, train_flag, kernel=(1, 1))
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)

    with tf.name_scope('Block_3'):
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)

    with tf.name_scope('Block_4'):
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)

    x = tf.concat([x, y], axis=3)

    with tf.name_scope('Block_6'):
        x = get_block(x, start_filter_size * 2 ** 5, train_flag)

    with tf.name_scope('Prediction'):
        x = get_block(x, n_anchors * (n_classes + 5), train_flag, kernel=(1, 1))

    return x


def yolo_prediction(head_tensor):
    pred_reshaped = tf.reshape(head_tensor, shape=(-1, 13, 13, 5, 8))
    # element order: c,x,y,w,h,p1,p2,p3
    pred_confidence = tf.sigmoid(pred_reshaped[..., 0:1])
    pred_xy = tf.sigmoid(pred_reshaped[..., 1:3])
    pred_wh = tf.exp(pred_reshaped[..., 3:5])
    pred_classes = tf.nn.softmax(pred_reshaped[..., 5:])

    return tf.concat([pred_confidence, pred_xy, pred_wh, pred_classes], axis=-1)


def yolo_loss(labels, predictions, mask):
    masked_labels = tf.boolean_mask(labels, mask)
    masked_predictions = tf.boolean_mask(predictions, mask)

    # ious = tensor_iou(masked_predictions[..., 1:5], masked_labels[..., 1:5])
    # ious = tf.expand_dims(ious, axis=-1)

    xy_loss = tf.reduce_sum((masked_labels[..., :2] - masked_predictions[..., 1:3]) ** 2)
    wh_loss = tf.reduce_sum((tf.sqrt(masked_predictions[..., 3:5]) - tf.sqrt(masked_labels[..., 2:4])) ** 2)

    #     conf_loss = tf.reduce_sum((masked_predictions[..., 0] - ious) ** 2)

    conf_loss = tf.reduce_sum((1 - masked_predictions[..., 0]) ** 2)

    no_obj_loss = tf.reduce_sum((tf.boolean_mask(predictions, ~mask)[..., 0] ** 2))

    class_loss = tf.reduce_sum((masked_predictions[..., 5:] - masked_labels[..., 4:]) ** 2)

    loss = 5 * (xy_loss + wh_loss) + conf_loss + no_obj_loss + class_loss

    return loss


def build_model(inp_ph, train_flag_ph, lbl_ph, mask_ph):
    head = build_graph(inp_ph, train_flag_ph, 32, 5, 3)
    pred = yolo_prediction(head)
    loss = yolo_loss(lbl_ph, pred, mask_ph)

    return loss

#
# input_tensor = tf.placeholder(dtype=tf.float32, shape=(None, 416, 416, 3))
# train_flag = tf.placeholder(dtype=tf.bool)
# labels = tf.placeholder(tf.float32, shape=(None, 13, 13, 5, 7))
# mask = tf.placeholder(tf.bool, shape=(None, 13, 13, 5))
#
# head = build_graph(input_tensor, train_flag, 32, 5, 3)
# pred = yolo_prediction(head)
# loss = yolo_loss(labels, pred, mask)
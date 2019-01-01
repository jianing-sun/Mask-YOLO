import numpy as np
import math
import scipy
from mrcnn import utils
import random
import logging
import tensorflow as tf
from distutils.version import LooseVersion
import skimage.color
import skimage.io
import skimage.transform
import cv2
from keras.utils import Sequence
from imgaug import augmenters as iaa
import copy

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def _sigmoid(x):
    return 1. / (1. + np.exp(-x))


def _softmax(x, axis=-1, t=-100.):
    x = x - np.max(x)

    if np.min(x) < t:
        x = x / np.min(x) * t

    e_x = np.exp(x)

    return e_x / e_x.sum(axis, keepdims=True)


def decode_one_yolo_output(netout, anchors, nb_class, obj_threshold=0.3, nms_threshold=0.3):
    grid_h, grid_w, nb_box = netout.shape[:3]

    boxes = []

    # decode the output by the network
    netout[..., 4] = _sigmoid(netout[..., 4])
    netout[..., 5:] = netout[..., 4][..., np.newaxis] * _softmax(netout[..., 5:])
    netout[..., 5:] *= netout[..., 5:] > obj_threshold

    for row in range(grid_h):
        for col in range(grid_w):
            for b in range(nb_box):
                # from 4th element onwards are confidence and class classes
                classes = netout[row, col, b, 5:]

                if np.sum(classes) > 0:
                    # first 4 elements are x, y, w, and h
                    x, y, w, h = netout[row, col, b, :4]

                    x = (col + _sigmoid(x)) / grid_w  # center position, unit: image width
                    y = (row + _sigmoid(y)) / grid_h  # center position, unit: image height
                    w = anchors[2 * b + 0] * np.exp(w) / grid_w  # unit: image width
                    h = anchors[2 * b + 1] * np.exp(h) / grid_h  # unit: image height
                    confidence = netout[row, col, b, 4]

                    box = BoundBox(x - w / 2, y - h / 2, x + w / 2, y + h / 2, confidence, classes)

                    boxes.append(box)

    # suppress non-maximal boxes
    for c in range(nb_class):
        sorted_indices = list(reversed(np.argsort([box.classes[c] for box in boxes])))

        for i in range(len(sorted_indices)):
            index_i = sorted_indices[i]

            if boxes[index_i].classes[c] == 0:
                continue
            else:
                for j in range(i + 1, len(sorted_indices)):
                    index_j = sorted_indices[j]

                    if bbox_iou(boxes[index_i], boxes[index_j]) >= nms_threshold:
                        boxes[index_j].classes[c] = 0

    # remove the boxes which are less likely than a obj_threshold
    boxes = [box for box in boxes if box.get_score() > obj_threshold]

    return boxes


def NMB(boxes, class_ids, indices, image_shape, nms_threshold=0.5):
    """ Suppress non-maximal boxes
    :param boxes:
    :param class_ids:
    :param indices:
    :param nb_class:
    :param nms_threshold:
    :return:
    """
    list_to_remove = []
    # for c in range(nb_class):

    for index_i in range(len(indices)):
        # index_i = np.where(indices == indices[i])

        for index_j in range(index_i + 1, len(indices)):
            # index_j = np.where(indices == indices[j])

            if bbox_iou_2(boxes[index_i], boxes[index_j], image_shape) >= nms_threshold \
                    and class_ids[index_i] == class_ids[index_j]:
                # boxes[index_j].classes[c] = 0
                list_to_remove.append(index_j)

    indices = np.delete(indices, list_to_remove)

    return indices


def box_refinement_graph(box, gt_box):
    """Compute refinement needed to transform box to gt_box.
    box and gt_box are [N, (y1, x1, y2, x2)]
    """
    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]
    center_x = box[:, 0] + 0.5 * width
    center_y = box[:, 1] + 0.5 * height

    gt_width = gt_box[:, 2] - gt_box[:, 0]
    gt_height = gt_box[:, 3] - gt_box[:, 1]
    gt_center_x = gt_box[:, 0] + 0.5 * gt_width
    gt_center_y = gt_box[:, 1] + 0.5 * gt_height

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dy, dx, dh, dw], axis=1)
    return result


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Return: [(height, width)].
    """
    # Currently supports mobilenet only
    assert config.BACKBONE in ["mobilenet"]
    return np.array(
        [int(math.ceil(image_shape[0] / config.BACKBONE_STRIDES)),
          int(math.ceil(image_shape[1] / config.BACKBONE_STRIDES))])


def mold_image(images, config):
    """Expects an RGB image (or array of images) and subtracts
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.MEAN_PIXEL


class BoundBox:
    def __init__(self, xmin, ymin, xmax, ymax, c=None, classes=None):
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax

        self.c = c
        self.classes = classes

        self.label = -1
        self.score = -1

    def get_label(self):
        if self.label == -1:
            self.label = np.argmax(self.classes)

        return self.label

    def get_score(self):
        if self.score == -1:
            self.score = self.classes[self.get_label()]

        return self.score


def bbox_iou(box1, box2):
    intersect_w = _interval_overlap([box1.xmin, box1.xmax], [box2.xmin, box2.xmax])
    intersect_h = _interval_overlap([box1.ymin, box1.ymax], [box2.ymin, box2.ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1.xmax - box1.xmin, box1.ymax - box1.ymin
    w2, h2 = box2.xmax - box2.xmin, box2.ymax - box2.ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def bbox_iou_2(box1, box2, image_shape):
    """ Compute IoU for detections decoded with tensorflow, now yolo_output
    :param box1: [xmin, ymin, xmax, ymax]
    :param box2: [xmin, ymin, xmax, ymax]
    :return: IoU between box1 and box2
    """
    w, h = image_shape[0], image_shape[1]
    box1_xmin = box1[0] * w
    box1_ymin = box1[1] * h
    box1_xmax = box1[2] * w
    box1_ymax = box1[3] * h

    box2_xmin = box2[0] * w
    box2_ymin = box2[1] * h
    box2_xmax = box2[2] * w
    box2_ymax = box2[3] * h

    intersect_w = _interval_overlap([box1_xmin, box1_xmax], [box2_xmin, box2_xmax])
    intersect_h = _interval_overlap([box1_ymin, box1_ymax], [box2_ymin, box2_ymax])

    intersect = intersect_w * intersect_h

    w1, h1 = box1_xmax - box1_xmin, box1_ymax - box1_ymin
    w2, h2 = box2_xmax - box2_xmin, box2_ymax - box2_ymin

    union = w1 * h1 + w2 * h2 - intersect

    return float(intersect) / union


def _interval_overlap(interval_a, interval_b):
    x1, x2 = interval_a
    x3, x4 = interval_b

    if x3 < x1:
        if x4 < x1:
            return 0
        else:
            return min(x2, x4) - x1
    else:
        if x2 < x3:
            return 0
        else:
            return min(x2, x4) - x3


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        # boxes[i] = np.array([y1, x1, y2, x2])
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


def load_image_gt(dataset, config, image_id, augment=False, augmentation=None,
                  use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (x1, y1, x2, y2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    network_image_shape = config.IMAGE_SHAPE
    original_shape = image.shape
    image, scale = resize_image(image, network_image_shape)
    mask = resize_mask(mask, scale)

    # Random horizontal flips.
    # TODO: will be removed in a future update in favor of augmentation
    if augment:
        logging.warning("'augment' is deprecated. Use 'augmentation' instead.")
        if random.randint(0, 1):
            image = np.fliplr(image)
            mask = np.fliplr(mask)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmenters that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return augmenter.__class__.__name__ in MASK_AUGMENTERS

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (y1, x1, y2, x2)]
    bbox = extract_bboxes(mask)

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    return image, class_ids, bbox, mask


def resize_image(image, net_image_shape):
    """Resizes an image keeping the aspect ratio changed.

    Returns:
    image: the resized image
    scale: The scale factor used to resize the image
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (x1, y1, x2, y2) and default scale == 1.
    h, w = image.shape[:2]
    scale = [1, 1]

    # Scale
    scale[0], scale[1] = net_image_shape[0] / h, net_image_shape[1] / w

    # Resize image using bilinear interpolation
    if scale != [1, 1]:
        image = resize(image, (round(h * scale[0]), round(w * scale[1])),
                       preserve_range=True)

    return image.astype(image_dtype), scale


def resize_mask(mask, scale):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the form
            [(top, bottom), (left, right), (0, 0)]
    """
    # Suppress warning from scipy 0.13.0, the output shape of zoom() is
    # calculated with round() instead of int()
    mask = scipy.ndimage.zoom(mask, zoom=[scale[0], scale[1], 1], order=0)
    # if crop is not None:
    #     y, x, h, w = crop
    #     mask = mask[y:y + h, x:x + w]
    # else:
    #     mask = np.pad(mask, padding, mode='constant', constant_values=0)
    return mask


def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        x1, y1, x2, y2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)


def data_generator(dataset, config, shuffle=True, augment=False, augmentation=None,
                   batch_size=1, no_augmentation_sources=None, norm=False):
    """A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.

    dataset: The Dataset object to pick data from
    config: The model config object
    shuffle: If True, shuffles the samples before every epoch
    augment: (deprecated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    random_rois: If > 0 then generate proposals to be used to train the
                 network classifier and mask heads. Useful if training
                 the Mask RCNN part without the RPN.
    batch_size: How many images to return in each call
    detection_targets: If True, generate detection targets (class IDs, bbox
        deltas, and masks). Typically for debugging or visualizations because
        in trainig detection targets are generated by DetectionTargetLayer.
    no_augmentation_sources: Optional. List of sources to exclude for
        augmentation. A source is string that identifies a dataset and is
        defined in the Dataset class.

    Returns a Python generator. Upon calling next() on it, the
    generator returns two lists, inputs and outputs. The contents
    of the lists differs depending on the received arguments:
    inputs list:
    - images: [batch, H, W, C]
    - image_meta: [batch, (meta data)] Image details. See compose_image_meta()
    - rpn_match: [batch, N] Integer (1=positive anchor, -1=negative, 0=neutral)
    - rpn_bbox: [batch, N, (dy, dx, log(dh), log(dw))] Anchor bbox deltas.
    - gt_class_ids: [batch, MAX_GT_INSTANCES] Integer class IDs
    - gt_boxes: [batch, MAX_GT_INSTANCES, (y1, x1, y2, x2)]
    - gt_masks: [batch, height, width, MAX_GT_INSTANCES]. The height and width
                are those of the image unless use_mini_mask is True, in which
                case they are defined in MINI_MASK_SHAPE.

    outputs list: Usually empty in regular training. But if detection_targets
        is True then the outputs list contains target class_ids, bbox deltas,
        and masks.
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)
    error_count = 0
    no_augmentation_sources = no_augmentation_sources or []

    anchors = [BoundBox(0, 0, config.ANCHORS[2 * i], config.ANCHORS[2 * i + 1]) for i in
                range(int(len(config.ANCHORS) // 2))]

    # Anchors
    # [anchor_count, (y1, x1, y2, x2)]
    # backbone_shapes = compute_backbone_shapes(config, config.IMAGE_SHAPE)
    # anchors = utils.generate_pyramid_anchors(config.RPN_ANCHOR_SCALES,
    #                                          config.RPN_ANCHOR_RATIOS,
    #                                          backbone_shapes,
    #                                          config.BACKBONE_STRIDES,
    #                                          config.RPN_ANCHOR_STRIDE)

    # Keras requires a generator to run indefinitely.
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]

            # If the image source is not to be augmented pass None as augmentation
            if dataset.image_info[image_id]['source'] in no_augmentation_sources:
                image, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, config, image_id, augment=augment,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)
            else:
                image, gt_class_ids, gt_boxes, gt_masks = \
                    load_image_gt(dataset, config, image_id, augment=augment,
                                augmentation=augmentation,
                                use_mini_mask=config.USE_MINI_MASK)

            # used for debug
            # fig, ax = plt.subplots(nrows=1, ncols=1)
            # print(gt_boxes)
            # ax.imshow(image[:, :, ::-1])
            # for i in range(0, len(gt_boxes)):
            #     ax.add_patch(Rectangle((gt_boxes[i][0], gt_boxes[i][1]), gt_boxes[i][2] - gt_boxes[i][0],
            #                            gt_boxes[i][3] - gt_boxes[i][1],
            #                            facecolor='none', edgecolor='#FF0000', linewidth=3.0))
            # plt.show()

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets
            # rpn_match, rpn_bbox = build_rpn_targets(image.shape, anchors,
            #                                         gt_class_ids, gt_boxes, config)

            # Mask R-CNN Targets
            # Init batch arrays
            if b == 0:
                batch_yolo_target = np.zeros((batch_size, config.GRID_H, config.GRID_W,
                                              config.N_BOX, 4 + 1 + config.NUM_CLASSES))
                batch_yolo_true_boxes = np.zeros((batch_size, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))
                batch_images = np.zeros((batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros((batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros((batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros((batch_size, gt_masks.shape[0], gt_masks.shape[1],
                                           config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                print('find instances more than 15 in an image')
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # YOLO
            true_box_index = 0
            for i in range(0, gt_boxes.shape[0]):
                # gt_boxes: [instance, (x1, y1, x2, y2)]
                xmin = gt_boxes[i][0]
                ymin = gt_boxes[i][1]
                xmax = gt_boxes[i][2]
                ymax = gt_boxes[i][3]

                center_x = .5 * (xmin + xmax)
                center_x = center_x / (float(config.IMAGE_SHAPE[0]) / config.GRID_W)
                center_y = .5 * (ymin + ymax)
                center_y = center_y / (float(config.IMAGE_SHAPE[1]) / config.GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < config.GRID_W and grid_y < config.GRID_H:
                    obj_indx = gt_class_ids[i]

                    center_w = (xmax - xmin) / (float(config.IMAGE_SHAPE[0]) / config.GRID_W)
                    center_h = (ymax - ymin) / (float(config.IMAGE_SHAPE[1]) / config.GRID_H)

                    yolo_box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for i in range(len(anchors)):
                        anchor = anchors[i]
                        iou = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = i
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    batch_yolo_target[b, grid_y, grid_x, best_anchor, 0:4] = yolo_box
                    batch_yolo_target[b, grid_y, grid_x, best_anchor, 4] = 1.
                    batch_yolo_target[b, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                    # assign the true box to b_batch
                    batch_yolo_true_boxes[b, 0, 0, 0, true_box_index] = yolo_box

                    true_box_index += 1
                    true_box_index = true_box_index % config.TRUE_BOX_BUFFER

            # Add to batch
            # batch_images[b] = mold_image(image.astype(np.float32), config)
            if norm == True:
                batch_images[b] = image / 255.        # normalize image
            else:
                # plot image and bounding boxes for sanity check
                for i in range(0, gt_boxes.shape[0]):
                    if grid_x < config.GRID_W and grid_y < config.GRID_H:
                        cv2.rectangle(image[:, :, ::-1], (gt_boxes[i][0], gt_boxes[i][1]), (gt_boxes[i][2], gt_boxes[i][3]),
                                      (255, 0, 0), 3)
                        cv2.putText(image[:, :, ::-1], gt_class_ids[i],
                                    (gt_boxes[i][0] + 2, gt_boxes[i][1] + 12),
                                    0, 1.2e-3 * image.shape[0],
                                    (0, 255, 0), 2)

                batch_images[b] = image

            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks

            b += 1

            # Batch full?
            if b >= batch_size:
                # inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target,
                #           batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]

                ### for yolo mode inputs
                # inputs = [input_image, input_true_boxes, input_yolo_target]
                inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target]

                # Model
                # inputs = [input_image, input_true_boxes, input_yolo_target,
                          # input_gt_class_ids, input_gt_boxes, input_gt_masks]

                outputs = []

                yield inputs, outputs

                # start a new batch
                b = 0

        except (GeneratorExit, KeyboardInterrupt):
            raise

        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


class BatchGenerator(Sequence):
    def __init__(self,
                 all_info,
                 config,
                 mode,
                 shuffle=True,
                 jitter=False,
                 norm=False):

        # self.generator = None
        self.config = config
        self.mode = mode
        self.all_info = all_info
        self.shuffle = shuffle
        self.jitter = jitter
        self.norm = norm

        assert mode in ['yolo', 'training']

        self.anchors = [BoundBox(0, 0, self.config.ANCHORS[2 * i], self.config.ANCHORS[2 * i + 1]) for i in
                        range(int(len(self.config.ANCHORS) // 2))]

        if shuffle:
            np.random.shuffle(self.all_info)   # image, gt_class_ids, gt_boxes, gt_masks
        # self.images = [item[0] for item in all_info]

    def __len__(self):
        return int(np.ceil(float(len(self.all_info)) / self.config.BATCH_SIZE))

    def num_classes(self):
        return self.config.NUM_CLASSES

    def size(self):
        return len(self.all_info)

    def load_image(self, i):
        return cv2.imread(self.all_info[i][0])

    def __getitem__(self, idx):
        l_bound = idx * self.config.BATCH_SIZE
        r_bound = (idx + 1) * self.config.BATCH_SIZE

        if r_bound > len(self.all_info):
            r_bound = len(self.all_info)
            l_bound = max(0, r_bound - self.config.BATCH_SIZE)

        instance_count = 0

        batch_images = np.zeros((r_bound - l_bound,) + (224, 224, 3), dtype=np.float32)
        batch_yolo_target = np.zeros((r_bound - l_bound, self.config.GRID_H, self.config.GRID_W,
                                      self.config.N_BOX, 4 + 1 + self.config.NUM_CLASSES))
        batch_yolo_true_boxes = np.zeros((r_bound - l_bound, 1, 1, 1, self.config.TRUE_BOX_BUFFER, 4))

        batch_gt_class_ids = np.zeros((r_bound - l_bound, self.config.TRUE_BOX_BUFFER), dtype=np.int32)
        batch_gt_boxes = np.zeros((r_bound - l_bound, self.config.TRUE_BOX_BUFFER, 4), dtype=np.int32)
        batch_gt_masks = np.zeros((r_bound - l_bound, 224, 224,
                                   self.config.MAX_GT_INSTANCES), dtype=np.bool)

        # x_batch = np.zeros((r_bound - l_bound, self.config.IMAGE_SHAPE[1], self.config.IMAGE_SHAPE[0], 3))  # input images
        # b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, self.config.TRUE_BOX_BUFFER,
        #                     4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        # y_batch = np.zeros((r_bound - l_bound, self.config.GRID_H, self.config.GRID_W, self.config.N_BOX,
        #                     4 + 1 + self.config.NUM_CLASSES))  # desired network output

        for train_instance in self.all_info[l_bound:r_bound]:

            image = train_instance[0]
            gt_class_ids = train_instance[1]
            gt_boxes = train_instance[2]
            gt_masks = train_instance[3]

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > self.config.TRUE_BOX_BUFFER:
                print('find instances more than 15 in an image')
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), self.config.TRUE_BOX_BUFFER, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            ### YOLO
            true_box_index = 0
            for i in range(0, gt_boxes.shape[0]):
                # gt_boxes: [instance, (x1, y1, x2, y2)]
                xmin = gt_boxes[i][0]
                ymin = gt_boxes[i][1]
                xmax = gt_boxes[i][2]
                ymax = gt_boxes[i][3]

                center_x = .5 * (xmin + xmax)
                center_x = center_x / (float(self.config.IMAGE_SHAPE[0]) / self.config.GRID_W)
                center_y = .5 * (ymin + ymax)
                center_y = center_y / (float(self.config.IMAGE_SHAPE[1]) / self.config.GRID_H)

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config.GRID_W and grid_y < self.config.GRID_H:
                    obj_indx = gt_class_ids[i]

                    center_w = (xmax - xmin) / (float(self.config.IMAGE_SHAPE[0]) / self.config.GRID_W)
                    center_h = (ymax - ymin) / (float(self.config.IMAGE_SHAPE[1]) / self.config.GRID_H)

                    yolo_box = [center_x, center_y, center_w, center_h]

                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou = -1

                    shifted_box = BoundBox(0,
                                           0,
                                           center_w,
                                           center_h)

                    for j in range(0, len(self.anchors)):
                        anchor = self.anchors[j]
                        iou = bbox_iou(shifted_box, anchor)

                        if max_iou < iou:
                            best_anchor = j
                            max_iou = iou

                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 0:4] = yolo_box
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 4] = 1.
                    batch_yolo_target[instance_count, grid_y, grid_x, best_anchor, 5 + obj_indx] = 1

                    # assign the true box to b_batch
                    batch_yolo_true_boxes[instance_count, 0, 0, 0, true_box_index] = yolo_box

                    true_box_index += 1
                    true_box_index = true_box_index % self.config.TRUE_BOX_BUFFER

            # assign input image to x_batch
            if self.norm:
                batch_images[instance_count] = image / 255.
            else:
                # plot image and bounding boxes for sanity check
                img = image[:, :, ::-1].astype(np.uint8).copy()
                for i in range(0, gt_boxes.shape[0]):
                    # if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    if train_instance[2][i][2] > train_instance[2][i][0] \
                            and train_instance[2][i][3] > train_instance[2][i][1]:
                        cv2.rectangle(img, (gt_boxes[i][0], gt_boxes[i][1]),
                                      (gt_boxes[i][2], gt_boxes[i][3]),
                                      (255, 0, 0), 2)
                        cv2.putText(img, str(gt_class_ids[i]),
                                    (gt_boxes[i][0] + 2, gt_boxes[i][1] + 12),
                                    0, 1.2e-3 * image.shape[0],
                                    (0, 255, 0), 1)

                batch_images[instance_count] = img

            batch_gt_class_ids[instance_count, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[instance_count, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[instance_count, :, :, :gt_masks.shape[-1]] = gt_masks

            # increase instance counter in current batch
            instance_count += 1

        if self.mode == 'yolo':
            inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target]
            outputs = []
        elif self.mode == 'training':
            inputs = [batch_images, batch_yolo_true_boxes, batch_yolo_target,
                      batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
            outputs = []
        else:
            raise NotImplementedError

        # return [x_batch, b_batch], y_batch
        return inputs, outputs


def draw_boxes(image, boxes, labels):
    image_h, image_w, _ = image.shape

    for box in boxes:
        xmin = int(box.xmin * image_w)
        ymin = int(box.ymin * image_h)
        xmax = int(box.xmax * image_w)
        ymax = int(box.ymax * image_h)

        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), (0, 255, 0), 1)
        cv2.putText(image,
                    labels[box.get_label()] + ' ' + str(box.get_score()),
                    (xmin, ymax - 13),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.5e-3 * image_h,
                    (0, 255, 0), 1)

    return image


def unmold_mask(mask, bbox, image_shape):
    """Converts a mask generated by the neural network to a format similar
    to its original shape.
    mask: [height, width] of type float. A small, typically 28x28 mask.
    bbox: [x1, y1, x2, y2]. The box to fit the mask in.

    Returns a binary mask with the same size as the original image.
    """
    threshold = 0.5
    w, h = image_shape[0], image_shape[1]
    x1, y1, x2, y2 = bbox

    # recover from normalized coordinates to pixel level
    x1 = min(max(0, int(x1 * w)), w)
    # minimum is 1 instead 0 in case resizing with zero denominator
    x2 = min(max(1, int(x2 * w)), w)
    y1 = min(max(0, int(y1 * h)), h)
    y2 = min(max(1, int(y2 * h)), h)

    # print('x2-x1, y2-y2', (max(1, x2 - x1), max(1, y2 - y1)))
    mask = resize(mask, (max(1, y2 - y1), max(1, x2 - x1)))
    # mask = resize(mask, (max(1, x2 - x1), max(1, y2 - y1)))
    mask = np.where(mask >= threshold, 1, 0).astype(np.bool)
    # mask = np.transpose(mask)

    # Put the mask in the right location.
    full_mask = np.zeros(image_shape[:2], dtype=np.bool)
    full_mask[y1:y2, x1:x2] = mask
    # full_mask[x1:x2, y1:y2] = mask
    return full_mask


def resize_one_image(image, gt_box, gt_mask, new_shape):
    original_w, original_h = image.shape[0], image.shape[1]

    new_w, new_h = new_shape[0], new_shape[1]
    new_image = cv2.resize(image, (new_w, new_h))
    new_image = new_image[:, :, ::-1]

    gt_box[0], gt_box[2] = int(gt_box[0] * float(new_w) / original_w), int(gt_box[1] * float(new_w) / original_w)
    gt_box[1], gt_box[3] = int(gt_box[1] * float(new_h) / original_h), int(gt_box[3] * float(new_h) / original_h)

    gt_box[0], gt_box[2] = max(min(gt_box[0], new_w), 0), max(min(gt_box[2], new_w), 0)
    gt_box[1], gt_box[3] = max(min(gt_box[1], new_h), 0), max(min(gt_box[3], new_h), 0)

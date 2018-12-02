from mrcnn import utils, model, visualize

import os
import random
import datetime
import re
import math
import logging
from collections import OrderedDict
import multiprocessing
import numpy as np
import tensorflow as tf
import keras
import keras.backend as K
import keras.layers as KL
import keras.engine as KE
import keras.models as KM
from mrcnn import utils

# used for buliding mobilenet backbone
from keras_applications import get_keras_submodule
from keras_applications.mobilenet import _depthwise_conv_block

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


backend = get_keras_submodule('backend')


############################################################
# Build an incomplete mobilenetv1 graph as backbone
############################################################


def relu6(x):
    return backend.relu(x, max_value=6)


def conv_block(inputs, filters, alpha=1.0, kernel=(3,3), strides=(1,1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = KL.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(inputs)
    x = KL.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = KL.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return KL.Activation(relu6, name='conv1_relu')(x)


def mobilenet_graph(input_image, architecture, stage5=False, alpha=1.0, depth_multiplier=1):
    """ Build a incompleted mobilenetv1 graph so as to generate enough spatial feature map
    resolution (28x28), one more depthwise block added for bigger d dimension.
    architecture: can be mobilenet, resnet50 or resnet100
    stage5: boolean. if create stage5 for the network or not
    alpha and depth_multiplier are parameters for mobilenet, the regular setting is 1 for both
    """
    assert architecture == 'mobilenet'

    # 224x224x3
    x = conv_block(input_image, 32, strides=(2, 2))

    # 112x112x32
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, strides=(2, 2), block_id=2)

    # 56x56x64
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, strides=(2, 2), block_id=4)

    # 28x28x256
    x = _depthwise_conv_block(x, 256, alpha, depth_multiplier, block_id=5)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=6)  # added by me

    return x   # output feature map shape [28x28x512]


def yolo_branch_graph(x, true_boxes, config, alpha=1.0, depth_multiplier=1):
    # 28x28x512
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, strides=(2, 2), block_id=7)

    # 14x14x512
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=11)
    x = _depthwise_conv_block(x, 512, alpha, depth_multiplier, block_id=12)

    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, strides=(2, 2), block_id=13)

    # 7x7x1024
    x = _depthwise_conv_block(x, 1024, alpha, depth_multiplier, block_id=14)

    # yolo output
    x = KL.Conv2D(config.N_BOX * (4 + 1 + config.NUM_CLASSES), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = KL.Reshape((config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES))(x)
    output = KL.Lambda(lambda args: args[0])([output, true_boxes])



############################################################
# Mask YOLO class
############################################################


class MaskYOLO():
    """ Build the overall structure of MaskYOLO class
    which only generate bbox and mask, no classification involved
    """
    def __init__(self, mode, config, model_dir):
        """
        mode: Either "training" or "inference"
        config: A Sub-class of the Config class
        model_dir: Directory to save training logs and trained weights
        """
        assert mode in ['training', 'inference']
        self.mode = mode
        self.config = config
        self.model_dir = model_dir
        self.keras_model = self.build(mode=mode, config=config)

    def build(self, mode, config):
        assert mode in ['training', 'inference']

        # Image size must be divided by 2 multiple times
        h, w = config.IMAGE_SHAPE[:2]
        if h / 2**6 != int(h / 2**6) or w / 2**6 != int(w / 2**6):
            raise Exception("Image size must be dividable by 2 at least 6 times "
                            "to avoid fractions when downscaling and upscaling."
                            "For example, use 256, 320, 384, 448, 512, ... etc. ")

        # Inputs
        input_image = KL.Input(shape=[None, None, config.IMAGE_SHAPE[2]], name="input_image")

        if mode == "training":

            # input_yolo_anchors and true_boxes
            input_yolo_anchors = KL.Input(shape=[1, 1, 1, config.TRUE_BOX_BUFFER, 4])
            input_true_boxes = KL.Input(shape=(1, 1, 1, config.TRUE_BOX_BUFFER, 4))

            # GT Masks (zero padded)
            if config.USE_MINI_MASK:
                input_gt_masks = KL.Input(shape=[config.MINI_MASK_SHAPE[0],
                                                 config.MINI_MASK_SHAPE[1], None],
                                          name="input_gt_masks", dtype=bool)
            else:
                input_gt_masks = KL.Input(shape=[config.IMAGE_SHAPE[0],
                                                 config.IMAGE_SHAPE[1], None],
                                          name="input_gt_masks", dtype=bool)
        elif mode == "inference":
            raise NotImplementedError

        C4 = mobilenet_graph(input_image, config.BACKBONE, stage5=False)
        myolo_feature_maps = rpn_feature_maps = C4

        if mode == "training":
            anchors = self.get_anchors(config.IMAGE_SHAPE)

    def get_anchors(self, image_shape):
        backbone_shapes = compute_backbone_shapes(self.config, image_shape)
        a = generate_anchors()


def generate_anchors(yolo_stride, shape, feature_stride, anchor_stride, ANCHORS):
    """ Generate anchors on feature map
    ANCHORS: knn generated anchors in config.py
    shape: [height, weight] spatial shape of the feature map over which to
           generate anchors
    feature_stride: stride of the feature map relative to the image in pixels
    anchor_stride: stride of anchors on the feature map. 1 or 2
    """
    widths = (ANCHORS[::2] * yolo_stride) / feature_stride
    heights = (ANCHORS[1:2] * yolo_stride) / feature_stride


def compute_backbone_shapes(config, image_shape):
    """Computes the width and height of each stage of the backbone network.
    Return: [(height, width)].
    """
    # Currently supports mobilenet only
    assert config.BACKBONE in ["mobilenet"]
    return np.array(
        [int(math.ceil(image_shape[0] / config.BACKBONE_STRIDES)),
          int(math.ceil(image_shape[1] / config.BACKBONE_STRIDES))])

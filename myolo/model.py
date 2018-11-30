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

# Requires TensorFlow 1.3+ and Keras 2.0.8+.
from distutils.version import LooseVersion
assert LooseVersion(tf.__version__) >= LooseVersion("1.3")
assert LooseVersion(keras.__version__) >= LooseVersion('2.0.8')


############################################################
# Mask YOLO class
############################################################

def darknet_graoh(input_image, architecture, stage5=False, train_bn=True):

    def space_to_depth_x2(x):
        return tf.space_to_depth(x, block_size=2)

    assert architecture == "darknet"
    # Layer 1
    x = KL.Conv2D(32, (3, 3), strides=(1, 1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = KL.BatchNormalization(name='norm_1')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)
    x = KL.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = KL.Conv2D(64, (3, 3), strides=(1, 1), padding='same', name='conv_2', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_2')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)
    x = KL.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = KL.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_3', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_3')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = KL.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_4', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_4')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = KL.Conv2D(128, (3, 3), strides=(1, 1), padding='same', name='conv_5', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_5')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)
    x = KL.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = KL.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_6', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_6')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = KL.Conv2D(128, (1, 1), strides=(1, 1), padding='same', name='conv_7', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_7')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = KL.Conv2D(256, (3, 3), strides=(1, 1), padding='same', name='conv_8', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_8')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)
    x = KL.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = KL.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_9', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_9')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = KL.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_10', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_10')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = KL.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_11', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_11')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = KL.Conv2D(256, (1, 1), strides=(1, 1), padding='same', name='conv_12', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_12')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = KL.Conv2D(512, (3, 3), strides=(1, 1), padding='same', name='conv_13', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_13')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = KL.MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_14', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_14')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = KL.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_15', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_15')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_16', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_16')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = KL.Conv2D(512, (1, 1), strides=(1, 1), padding='same', name='conv_17', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_17')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_18', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_18')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_19', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_19')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_20', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_20')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = KL.Conv2D(64, (1, 1), strides=(1, 1), padding='same', name='conv_21', use_bias=False)(
        skip_connection)
    skip_connection = KL.BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = KL.LeakyReLU(alpha=0.1)(skip_connection)
    skip_connection = KL.Lambda(space_to_depth_x2)(skip_connection)

    x = KL.concatenate([skip_connection, x])

    # Layer 22
    x = KL.Conv2D(1024, (3, 3), strides=(1, 1), padding='same', name='conv_22', use_bias=False)(x)
    x = KL.BatchNormalization(name='norm_22')(x)
    x = KL.LeakyReLU(alpha=0.1)(x)

    # Layer 23
    x = KL.Conv2D(BOX * (4 + 1 + CLASS), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = KL.Reshape((GRID_H, GRID_W, BOX, 4 + 1 + CLASS))(x)

    # small hack to allow true_boxes to be registered when Keras build the model
    # for more information: https://github.com/fchollet/keras/issues/2790
    output = KL.Lambda(lambda args: args[0])([output, true_boxes])

    model = KM.Model([input_image, true_boxes], output)

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
            # input_yolo_anchors
            input_yolo_anchors = KL.Input(shape=[1, 1, 1, config.TRUE_BOX_BUFFER, 4])

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

        if callable(config.BACKBONE):
            raise NotImplementedError
        else:


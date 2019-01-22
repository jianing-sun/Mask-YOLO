import numpy as np
import os
import cv2
from example.rice.rice_dataset import RiceDataset, RiceConfig
import myolo.model as modellib
from myolo import myolo_utils as mutils
from myolo import visualize

from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT_DIR = '/Users/jianingsun/Documents/Research/Nov/mask_food'
config = RiceConfig()
FOOD_DIR = os.path.join(ROOT_DIR, "datasets/food")

config.display()

# Training dataset
dataset_train = RiceDataset()
dataset_train.load_rice(FOOD_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = RiceDataset()
dataset_val.load_rice(FOOD_DIR, "val")
dataset_val.prepare()

image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=0, augment=None,
                                                               augmentation=None,
                                                               use_mini_mask=config.USE_MINI_MASK)
# image = image[:, :, ::-1]
# visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids, dataset_train.class_names)
# image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)

model = modellib.MaskYOLO(mode="inference",
                          config=config,
                          yolo_pretrain_dir=None,
                          yolo_trainable=True)
# print(model.keras_model.summary())
model.load_weights('./saved_model_Jan06-18-03.h5')

# model.infer_yolo(image, './yolo_model_Jan04-19-37.h5')
# model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=20, layers='all')
# image = cv2.imread('./test.jpg')
# image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
model.detect(image, './saved_model_Jan06-18-03.h5')




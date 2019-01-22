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
RICE_DIR = os.path.join(ROOT_DIR, "datasets/rice")

config.display()

# Training dataset
dataset_train = RiceDataset()
dataset_train.load_rice(RICE_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = RiceDataset()
dataset_val.load_rice(RICE_DIR, "val")
dataset_val.prepare()

# img = cv2.imread('/Users/jianingsun/Documents/Research/Nov/mask_food/datasets/rice/train/1.jpg')
# file = open('./rice_boxes', 'w')

image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=0, augment=None,
                                                               augmentation=None,
                                                               use_mini_mask=config.USE_MINI_MASK)
#     file.write(str(gt_boxes[0]) + '\n')
# file.close()
# visualize.display_instances(image, gt_boxes, gt_masks, gt_class_ids, dataset_train.class_names)
image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
image = image[:, :, ::-1]
model = modellib.MaskYOLO(mode="training",
                          config=config,
                          yolo_pretrain_dir=None,
                          yolo_trainable=True)
print(model.keras_model.summary())
# model.load_weights('./saved_model_Jan02-09-54.h5')

# model.infer_yolo(image, './saved_model_Jan01-21-14.h5')
# model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=10, layers='all')
# image = cv2.imread('./test.jpg')
# image = cv2.cvtColor(cv2.resize(image, (224, 224)), cv2.COLOR_BGR2RGB)
model.detect(image, './saved_model_Jan02-10-04.h5')

# image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=440, augment=None,
#                                                                augmentation=None,
#                                                                use_mini_mask=config.USE_MINI_MASK)
#
#
# def img_frombytes(data):
#     size = data.shape[::-1]
#     databytes = np.packbits(data, axis=1)
#     return Image.frombytes(mode='1', size=size, data=databytes)
#
#
# fig, ax = plt.subplots(nrows=1, ncols=1)
# print(gt_boxes)
# ax.imshow(image[:, :, ::-1])
# for i in range(0, len(gt_boxes)):
#     ax.add_patch(Rectangle((gt_boxes[i][0], gt_boxes[i][1]), gt_boxes[i][2]-gt_boxes[i][0], gt_boxes[i][3]-gt_boxes[i][1],
#                            facecolor='none', edgecolor='#FF0000', linewidth=3.0))
# plt.show()
#
# print(gt_boxes)
# for i in range(0, len(gt_boxes)):
#     mask_image = img_frombytes(gt_masks[:, :, i])
#     mask_image.show()
#     mask_image.close()



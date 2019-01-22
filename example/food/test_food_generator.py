import os
from example.rice.rice_dataset import RiceDataset, RiceConfig
from myolo import myolo_utils as mutils
import matplotlib.pyplot as plt
from myolo import visualize
import numpy as np


config = RiceConfig()
config.display()

ROOT_DIR = '/Users/jianingsun/Documents/Research/Nov/mask_food'
RICE_DIR = os.path.join(ROOT_DIR, "datasets/food")

# Training dataset
dataset_train = RiceDataset()
dataset_train.load_rice(RICE_DIR, "train")
dataset_train.prepare()

# Validation dataset
dataset_val = RiceDataset()
dataset_val.load_rice(RICE_DIR, "val")
dataset_val.prepare()

# all_info = []
# for id in range(0, 500):
#     image, gt_class_ids, gt_boxes, gt_masks = \
#         mutils.load_image_gt(dataset_train, config, id,
#                       use_mini_mask=config.USE_MINI_MASK)
#     all_info.append([image, gt_class_ids, gt_boxes, gt_masks])

# Data generators
train_info = []
for id in range(0, 50):
    image, gt_class_ids, gt_boxes, gt_masks = \
        mutils.load_image_gt(dataset_train, config, id,
                             use_mini_mask=config.USE_MINI_MASK)
    train_info.append([image, gt_class_ids, gt_boxes, gt_masks])

val_info = []
for id in range(0, 6):
    image, gt_class_ids, gt_boxes, gt_masks = \
        mutils.load_image_gt(dataset_val, config, id,
                             use_mini_mask=config.USE_MINI_MASK)
    val_info.append([image, gt_class_ids, gt_boxes, gt_masks])

train_generator = mutils.BatchGenerator(train_info, config, mode='training',
                                        shuffle=True, jitter=False, norm=True)

val_generator = mutils.BatchGenerator(val_info, config, mode='training',
                                      shuffle=False, jitter=False, norm=False)


img = train_generator[0][0][0][1]
plt.imshow(img.astype('float'))
gt_masks = train_generator[0][0][5][1]
gt_boxes = train_generator[0][0][4][1]
gt_class_ids = train_generator[0][0][3][1]
visualize.display_instances(img, gt_boxes, gt_masks, gt_class_ids, dataset_train.class_names)

print('ss')



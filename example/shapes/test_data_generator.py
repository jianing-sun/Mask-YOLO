from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
from myolo import myolo_utils as mutils
import matplotlib.pyplot as plt
import numpy as np

config = ShapesConfig()
config.display()

# Training dataset
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_train.prepare()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# all_info = []
# for id in range(0, 500):
#     image, gt_class_ids, gt_boxes, gt_masks = \
#         mutils.load_image_gt(dataset_train, config, id,
#                       use_mini_mask=config.USE_MINI_MASK)
#     all_info.append([image, gt_class_ids, gt_boxes, gt_masks])

# Data generators
train_info = []
for id in range(0, 500):
    image, gt_class_ids, gt_boxes, gt_masks = \
        mutils.load_image_gt(dataset_train, config, id,
                             use_mini_mask=config.USE_MINI_MASK)
    train_info.append([image, gt_class_ids, gt_boxes, gt_masks])

val_info = []
for id in range(0, 50):
    image, gt_class_ids, gt_boxes, gt_masks = \
        mutils.load_image_gt(dataset_val, config, id,
                             use_mini_mask=config.USE_MINI_MASK)
    val_info.append([image, gt_class_ids, gt_boxes, gt_masks])

train_generator = mutils.BatchGenerator(train_info, config, mode='yolo',
                                        shuffle=True, jitter=False, norm=True)

val_generator = mutils.BatchGenerator(val_info, config, mode='yolo',
                                              shuffle=True, jitter=False, norm=False)


img = val_generator[0][0][0][5]
plt.imshow(img.astype('uint8'))

print('ss')


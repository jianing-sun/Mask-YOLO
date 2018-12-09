import numpy as np
from example.dataset_shapes import ShapesDataset, ShapesConfig
import myolo.model as modellib
from myolo import myolo_utils as mutils
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


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

model = modellib.MaskYOLO(mode="training", config=config)
model.train(dataset_train, dataset_val, learning_rate=config.LEARNING_RATE, epochs=1, layers='all')

# image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=440, augment=None,
#                               augmentation=None,
#                               use_mini_mask=config.USE_MINI_MASK)
# fig, ax = plt.subplots(nrows=1, ncols=1)
# print(gt_boxes)
# ax.imshow(image[:, :, ::-1])
# for i in range(0, len(gt_boxes)):
#     ax.add_patch(Rectangle((gt_boxes[i][0], gt_boxes[i][1]), gt_boxes[i][2]-gt_boxes[i][0], gt_boxes[i][3]-gt_boxes[i][1],
#                            facecolor='none', edgecolor='#FF0000', linewidth=3.0))
# plt.show()

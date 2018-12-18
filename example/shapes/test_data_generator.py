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

all_info = []
for id in range(0, 500):
    image, gt_class_ids, gt_boxes, gt_masks = \
        mutils.load_image_gt(dataset_train, config, id,
                      use_mini_mask=config.USE_MINI_MASK)
    all_info.append([image, gt_class_ids, gt_boxes, gt_masks])


batch_generator = mutils.BatchGenerator(all_info, config, mode='yolo', shuffle=True, jitter=False, norm=True)
img = batch_generator[0][0][0][5]
plt.imshow(img.astype('float'))

train_generator = mutils.data_generator(dataset_train, config, shuffle=True,
                                        augmentation=None,
                                        batch_size=config.BATCH_SIZE,
                                        norm=False)
print(train_generator[0])
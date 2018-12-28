import numpy as np
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
import myolo.model as modellib
import random
from myolo import myolo_utils as mutils
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


config = ShapesConfig()
config.display()

model = modellib.MaskYOLO(mode="inference", config=config)
model.load_weights('./1226_model_yolo_val1_1129.h5')
model.keras_model.summary()

# Validation dataset
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

# Test on a random image
image_id = random.choice(dataset_val.image_ids)
original_image, gt_class_id, gt_bbox, gt_mask = mutils.load_image_gt(dataset_val, config,
                                                                     image_id, use_mini_mask=False)
input_image = original_image / 255.

plt.figure(figsize=(5, 5))
plt.imshow(input_image)

input_image = np.expand_dims(input_image, axis=0)
dummy_true_boxes = np.zeros((1, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))

model.detect_for_one()
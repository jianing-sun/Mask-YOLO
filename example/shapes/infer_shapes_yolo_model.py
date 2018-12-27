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

model = modellib.MaskYOLO(mode="yolo", config=config)
model.load_weights('./1226_yolo_val0_40.h5')
model.keras_model.summary()

''' Validation dataset '''
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, config.IMAGE_SHAPE[0], config.IMAGE_SHAPE[1])
dataset_val.prepare()

''' Test on a random image '''
image_id = random.choice(dataset_val.image_ids)
original_image, gt_class_id, gt_bbox, gt_mask = mutils.load_image_gt(dataset_val, config,
                                                                     image_id, use_mini_mask=False)
input_image = original_image / 255.

plt.figure(figsize=(5, 5))
plt.imshow(input_image)
input_image = np.expand_dims(input_image, axis=0)

dummy_true_boxes = np.zeros((1, 1, 1, 1, config.TRUE_BOX_BUFFER, 4))
dummy_target = np.zeros(shape=[1, config.GRID_H, config.GRID_W, config.N_BOX, 4 + 1 + config.NUM_CLASSES])

netout = model.keras_model.predict([input_image, dummy_true_boxes, dummy_target])[0]

boxes = mutils.decode_one_yolo_output(netout[0],
                                      anchors=config.ANCHORS,
                                      nb_class=config.NUM_CLASSES)

image = mutils.draw_boxes(input_image[0], boxes, labels=config.LABELS)

plt.imshow(image)
plt.show()
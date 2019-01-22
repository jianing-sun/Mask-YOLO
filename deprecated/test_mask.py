import tensorflow as tf
import matplotlib.pyplot as plt
from myolo import myolo_utils as mutils
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
from PIL import Image
import numpy as np


def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)


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

image, gt_class_ids, gt_boxes, gt_masks = mutils.load_image_gt(dataset_train, config, image_id=440, augment=None,
                              augmentation=None,
                              use_mini_mask=config.USE_MINI_MASK)

img = plt.imread('/Users/jianingsun/Desktop/pizza-test.jpeg')
# img = img_frombytes(gt_masks[:, :, 0])
# im2arr = np.array(img)
# shape = im2arr.shape
shape = img.shape
# im2arr = im2arr.reshape([1, shape[0], shape[1]])
img = img.reshape([1, shape[0], shape[1], shape[2]])
a = tf.image.crop_and_resize(img, [[0,0,0,0], [0.2, 0.6, 1.3, 0.9]], box_ind=[0, 0], crop_size=(100, 100))
sess = tf.Session()
b = a.eval(session=sess)
plt.imshow(b[0] / 255)
plt.imshow(b[0].astype('uint8'))
print('over')

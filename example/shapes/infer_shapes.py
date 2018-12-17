import random

import matplotlib.pyplot as plt
from myolo import visualize

import myolo.model as modellib
from example.shapes.dataset_shapes import ShapesDataset, ShapesConfig
from myolo import myolo_utils as mutils


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


inference_config = ShapesConfig()
inference_config.BATCH_SIZE = 1
inference_config.display()

''' Recreate the model in inference mode '''
model = modellib.MaskYOLO(mode="inference",
                          config=inference_config)

''' Load trained weights '''
model.load_weights('./model_1216.h5', by_name=True)

''' Training dataset '''
dataset_train = ShapesDataset()
dataset_train.load_shapes(500, inference_config.IMAGE_SHAPE[0], inference_config.IMAGE_SHAPE[1])
dataset_train.prepare()

''' Validation dataset '''
dataset_val = ShapesDataset()
dataset_val.load_shapes(50, inference_config.IMAGE_SHAPE[0], inference_config.IMAGE_SHAPE[1])
dataset_val.prepare()

''' Test on a random image '''
image_id = random.choice(dataset_val.image_ids)
original_image, gt_class_id, gt_bbox, gt_mask = mutils.load_image_gt(dataset_val, inference_config,
                                                                     image_id, use_mini_mask=False)

# visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
#                             dataset_train.class_names, figsize=(8, 8))

results = model.detect_for_one([original_image], verbose=1)

r = results[0]
visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                            dataset_val.class_names, r['scores'], ax=get_ax())

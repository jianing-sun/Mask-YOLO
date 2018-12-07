import numpy as np
from example.dataset_shapes import ShapesDataset, ShapesConfig
import myolo.model as modellib


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
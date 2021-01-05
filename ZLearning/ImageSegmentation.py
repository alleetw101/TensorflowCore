# ImageSegmentation
# https://github.com/alleetw101/TensorflowCore <2020>
#
# Model trained with Tensorflow Dataset https://www.tensorflow.org/datasets/catalog/oxford_iiit_pet

import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)

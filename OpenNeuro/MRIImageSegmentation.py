# MRIImageSegmentation
# https://github.com/alleetw101/TensorflowCore <2021>

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import niftiProcessing
import MRIProcessingDS003434

# Preprocessing
ds_path ='OpenNeuroDS003434newbi4fmri2020/ds003434/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz'
mask_path = 'OpenNeuroDS003434newbi4fmri2020/s1s1SagittalSlicesMasks'
scans = niftiProcessing.load_mri_scan(ds_path, use_float64=False, denoise=True)
masks = MRIProcessingDS003434.load_mask(mask_path, layered=True)

ds = tf.data.Dataset.from_tensor_slices((scans, masks)).shuffle(20)

batch_size = 4
val_size = int(0.1 * len(scans))
test_size = int(0.1 * len(scans))
train_size = int(len(scans) - val_size - test_size)

train_ds = ds.take(train_size).cache().batch(batch_size).repeat()
val_ds = ds.skip(train_size).take(val_size).batch(batch_size)
test_ds = ds.skip(train_size + val_size).batch(batch_size)


def display(display_list):
    mask = np.ma.masked_where(display_list[1] == 0, display_list[1])

    plt.figure(figsize=(10, 9))
    plt.imshow(display_list[0], cmap='gray')
    plt.imshow(mask, alpha=0.8)
    plt.show()


for image, mask in train_ds.skip(20).take(1):
    display([image[0].numpy(), mask[0].numpy()])


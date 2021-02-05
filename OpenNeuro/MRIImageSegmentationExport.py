# MRIImageSegmentationExport
# https://github.com/alleetw101/TensorflowCore <2021>

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import niftiProcessing

model = tf.keras.models.load_model('/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/SavedModelMRIImageSegmentation')

sample_path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz'
sample = niftiProcessing.load_mri_scan(sample_path, denoise=True)
sample = np.concatenate((sample,)*3, axis=-1)
sample_image = sample[86]
prediction = model.predict(sample_image[tf.newaxis, ...])

prediction[prediction < 0] = 0
prediction[prediction > 0] = 255
prediction = prediction[0][..., :1]

prediction = np.ma.masked_where(prediction == 0, prediction)

plt.figure(figsize=(9, 9))
plt.xticks([])
plt.yticks([])
plt.imshow(sample_image, cmap='gray')
plt.imshow(prediction, alpha=0.5)
plt.show()


sample_image = sample_image[..., :1]
png = np.concatenate((sample_image, prediction), axis=-1)
print(prediction.shape)
print(sample_image.shape)
print(png.shape)
print(png[80])

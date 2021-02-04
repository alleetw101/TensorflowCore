# MRIImageSegmentationExport
# https://github.com/alleetw101/TensorflowCore <2021>

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import niftiProcessing

model = tf.keras.models.load_model('/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/SavedModelMRIImageSegmentation')

sample_path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-02/ses-01/anat/sub-02_ses-01_T1w.nii.gz'
sample = niftiProcessing.load_mri_scan(sample_path, denoise=True)
sample = np.stack((sample,)*3, axis=-2)
sample = np.squeeze(sample, axis=-1)
sample_image = sample[20]
print(sample_image.shape)
prediction = model.predict(sample_image[tf.newaxis, ...])
print(prediction[0][124])
# prediction += abs(np.amin(prediction))
# prediction /= np.amax(prediction)

prediction[prediction < 0] = 0
prediction[prediction > 0] = 1

# prediction = np.ma.masked_where(prediction == 0, prediction)

plt.figure(figsize=(9,9))
plt.xticks([])
plt.yticks([])
plt.imshow(prediction[0])
plt.imshow(sample_image, cmap='gray')
plt.imshow(prediction[0], alpha=0.5)
plt.show()

print(prediction[0][124])
print(sample_image[124])
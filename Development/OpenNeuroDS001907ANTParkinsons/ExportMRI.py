# ExportMri
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI 3D projection using dataset from https://openneuro.org/datasets/ds001907/versions/2.0.3

from PIL import Image
import SimpleITK as sitk
import tensorflow as tf
import MRIProcessingDS001907
import numpy as np

path = 'ds001907/sub-RC4101/ses-1/anat/sub-RC4101_ses-1_T1w.nii.gz'

img = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(img)
data /= np.amax(data)
data *= 255
# data = tf.expand_dims(data, -1)

def axial_slices3d(data_array) -> np.ndarray:
    return np.transpose(data_array, [1, 2, 0])

def sagittal_slices3d(data_array) -> np.ndarray:
    return np.flip(data_array, axis=1)

axial = sagittal_slices3d(data)
image = Image.fromarray(axial[110])
image = image.convert('L')
image.save('testiamge.png')
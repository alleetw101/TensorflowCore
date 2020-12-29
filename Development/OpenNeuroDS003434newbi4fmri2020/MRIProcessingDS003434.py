# MRIRepresentation
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI 3D projection using dataset from https://openneuro.org/datasets/ds001907/versions/2.0.3
# MRI left/right is opposite of personal left/right

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf

import time


def load_mri_scan(filepath: str, pad: bool = True, pad_shape=(176, 256, 256), normalize: bool = True,
                  normalize_range=(0, 1), expand_dims: bool = True) -> np.ndarray:
    image = sitk.GetArrayFromImage(sitk.ReadImage(filepath))
    if pad:
        if len(pad_shape) != 3:
            raise AttributeError('pad_shape must be a tuple of three int values')
        image = np.pad(image, [(0, 176 - image.shape[0]), (0, 256 - image.shape[1]), (0, 256 - image.shape[2])],
                       mode='constant', constant_values=0.0)
    if expand_dims:
        image = np.expand_dims(image, axis=-1)

    if normalize:
        image = image / np.amax(image)
        if normalize_range != (0, 1):
            image *= (normalize_range[1] - normalize_range[0])
            image += normalize_range[0]

    return image


def plot_slice(data: np.ndarray, plot_index: int, step_size: int = 10, figsize=(10, 10), vplots: int = 1, hplots: int = 1):
    plt.figure(figsize=figsize)
    for position in range(vplots * hplots):
        plt.subplot(vplots, hplots, position + 1)
        plt.imshow(data[plot_index + (position * step_size)], cmap='gray')

    plt.show()


def axial_slices3d(data_array) -> np.ndarray:
    if data_array.ndim == 3:
        output = np.transpose(data_array, [1, 2, 0])
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [1, 2, 0, 3])
    else:
        raise AttributeError('Number of dimensions must be 3 or 4')
    return output


def coronal_slices3d(data_array) -> np.ndarray:
    if data_array.ndim == 3:
        output = np.transpose(data_array, [2, 1, 0])
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [2, 1, 0, 3])
    else:
        raise AttributeError('Number of dimensions must be 3 or 4')
    return np.flip(output, axis=[0])


def sagittal_slices3d(data_array) -> np.ndarray:
    return data_array


path = 'sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz'

data = load_mri_scan(path)
data = sagittal_slices3d(data)
# plot_slice(data, plot_index=110)

print(data.shape)
print(data.shape[0])
print(np.amax(data))
print(np.amin(data))
print(np.dtype(data[0][0][0][0]))
# print(data[110][110])

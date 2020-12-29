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
    elif data_array.ndim == 4 and data_array.shape[-1] <= 3:
        output = np.transpose(data_array, [1, 2, 0, 3])
    elif data_array.ndim == 4 or data_array.ndim == 5:
        output = data_array
    else:
        raise AttributeError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


def coronal_slices3d(data_array) -> np.ndarray:
    if data_array.ndim == 3:
        output = np.transpose(data_array, [2, 1, 0])
        output = np.flip(output, axis=[0])
    elif data_array.ndim == 4 and data_array.shape[-1] <= 3:
        output = np.transpose(data_array, [2, 1, 0, 3])
        output = np.flip(output, axis=[0])
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [0, 3, 2, 1])
        output = np.rot90(output, axes=(2, 3))
    elif data_array.ndim == 5:
        output = np.transpose(data_array, [0, 3, 2, 1, 4])
        output = np.rot90(output, axes=(2, 3))
    else:
        raise AttributeError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


def sagittal_slices3d(data_array) -> np.ndarray:  # Work in progress
    if data_array.ndim == 3 or (data_array.ndim == 4 and data_array.shape[-1] <= 3):
        output = data_array
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [0, 3, 2, 1])
        output = np.rot90(output, axes=(2, 3))
    elif data_array.ndim == 5:
        output = np.transpose(data_array, [0, 3, 2, 1, 4])
        output = np.rot90(output, axes=(2, 3))
    else:
        raise AttributeError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


path = 'sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz'

# data = load_mri_scan(path)
# data = sagittal_slices3d(data)
# plot_slice(data, plot_index=110)

fmripath = 'sub-01-ses-01-func-sub-01_ses-01_task-MainExp_run-01_bold.nii.gz'
image = sitk.GetArrayFromImage(sitk.ReadImage(fmripath))
image = np.expand_dims(image, axis=-1)
image = coronal_slices3d(image)
image = image / np.amax(image)

print(image.shape)

image[image < 0.01] = 0.0


plt.figure(figsize=(10, 10))
# for index in range(25):
#     plt.subplot(5, 5, index + 1)
#     plt.imshow(image[(index * 10)][45], cmap='viridis')

plt.imshow(image[50][45])

plt.show()
# print(image[200][30][40])
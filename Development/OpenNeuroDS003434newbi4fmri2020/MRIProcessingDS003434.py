# MRIProcessingDS003434
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI/MRI representations using dataset from https://openneuro.org/datasets/ds003434/versions/1.0.1
#

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


def load_mri_scan(filepath: str, pad: bool = True, pad_shape=None, normalize: bool = True,
                  normalize_range=(0, 1), expand_dims: bool = True, fmri_denoise: bool = False,
                  fmri_denoise_level: int = 0.01) -> np.ndarray:
    output = sitk.GetArrayFromImage(sitk.ReadImage(filepath))

    if not (output.ndim == 3 or output.ndim == 4):
        raise FileExistsError('File must contain a 3d or 4d array representable object')

    if pad:
        if pad_shape is None:
            if output.ndim == 3:
                pad_shape = (176, 256, 256)
            else:
                pad_shape = (280, 52, 84, 84)
        elif len(pad_shape) != output.ndim:
            raise ValueError(f'pad_shape must be a tuple of {output.ndim} int values')

        pad_array = []
        for index in range(len(pad_shape)):
            if pad_shape[index] - output.shape[index] >= 0:
                pad_array.append((0, pad_shape[index] - output.shape[index]))
            else:
                raise ValueError(f'Padding (pad_shape) must be greater than the original image in dimension {index}')

        output = np.pad(output, pad_array, mode='constant', constant_values=0.0)

    if normalize:
        output = output / np.amax(output)

        if output.ndim == 4 and fmri_denoise:
            output[output < fmri_denoise_level] = 0.0

        if normalize_range != (0, 1):
            output *= (normalize_range[1] - normalize_range[0])
            output += normalize_range[0]

    if expand_dims:
        output = np.expand_dims(output, axis=-1)

    return output


def plot_slice(data_array: np.ndarray, slice_index: int, time_index: int = 0, step_size: int = 10, figsize=(10, 10),
               vplots: int = 1, hplots: int = 1, cmap: str = 'gray'):
    plt.figure(figsize=figsize)

    if data_array.ndim == 3 or (data_array.ndim == 4 and data_array.shape[-1] <= 3):
        for position in range(vplots * hplots):
            plt.subplot(vplots, hplots, position + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(data_array[slice_index + (position * step_size)], cmap=cmap)
            plt.title(f'Slice index: {slice_index + (position * step_size)}')
    elif data_array.ndim == 4 or data_array.ndim == 5:
        for position in range(vplots * hplots):
            plt.subplot(vplots, hplots, position + 1)
            plt.xticks([])
            plt.yticks([])
            plt.imshow(data_array[time_index][slice_index + (position * step_size)], cmap=cmap)
            plt.title(f'Time index: {time_index}, Slice index: {slice_index + (position * step_size)}')
    else:
        raise ValueError('Input data must be a 3d, 4d, or 5d numpy array')

    plt.show()


def axial_slices3d(data_array) -> np.ndarray:  # Hiegh [-3] index is superior
    if data_array.ndim == 3:
        output = np.transpose(data_array, [1, 2, 0])
        output = np.flip(output, axis=0)
    elif data_array.ndim == 4 and data_array.shape[-1] <= 3:
        output = np.transpose(data_array, [1, 2, 0, 3])
        output = np.flip(output, axis=0)
    elif data_array.ndim == 4 or data_array.ndim == 5:
        output = data_array
        output = np.flip(output, axis=3)
    else:
        raise ValueError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


def coronal_slices3d(data_array) -> np.ndarray:  # Higher [-3] index is anterior
    if data_array.ndim == 3:
        output = np.transpose(data_array, [2, 1, 0])
        output = np.flip(output, axis=[0])
    elif data_array.ndim == 4 and data_array.shape[-1] <= 3:
        output = np.transpose(data_array, [2, 1, 0, 3])
        output = np.flip(output, axis=[0])
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [0, 2, 3, 1])
        output = np.rot90(output, axes=(2, 3))
        output = np.flip(output, axis=1)
    elif data_array.ndim == 5:
        output = np.transpose(data_array, [0, 2, 3, 1, 4])
        output = np.rot90(output, axes=(2, 3))
        output = np.flip(output, axis=1)
    else:
        raise ValueError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


def sagittal_slices3d(data_array) -> np.ndarray:  # Higher index is MRI left (POV right)
    if data_array.ndim == 3 or (data_array.ndim == 4 and data_array.shape[-1] <= 3):
        output = data_array
    elif data_array.ndim == 4:
        output = np.transpose(data_array, [0, 3, 2, 1])
        output = np.rot90(output, axes=(2, 3))
        output = np.flip(output, axis=1)
    elif data_array.ndim == 5:
        output = np.transpose(data_array, [0, 3, 2, 1, 4])
        output = np.rot90(output, axes=(2, 3))
        output = np.flip(output, axis=1)
    else:
        raise ValueError('Number of dimensions must be 3 or 4 for MRI and 4 or 5 for fMRI')
    return output


path = 'sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz'
fmripath = 'sub-01-ses-01-func-sub-01_ses-01_task-MainExp_run-01_bold.nii.gz'

# data = load_mri_scan(path)
# data = sagittal_slices3d(data)
# plot_slice(data, slice_index=110)

data = load_mri_scan(fmripath)
data = sagittal_slices3d(data)
plot_slice(data, slice_index=50)

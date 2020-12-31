# MRIProcessingDS003434
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI/MRI representations using dataset from https://openneuro.org/datasets/ds003434/versions/1.0.1
# Numpy array of fmri dataset (5 runs per subject) contains 9e9 elements
# Numpy array of mri dataset contains 2e8 elements

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import os
import time

from PIL import Image


def load_mri_scan(filepath: str, use_float64: bool = False, pad: bool = True, pad_shape=None, normalize: bool = True,
                  normalize_range=(0, 1), expand_dims: bool = True, denoise: bool = False, denoise_lower: float = 0.05,
                  denoise_upper: float = 0.45) -> np.ndarray:
    dtype = sitk.sitkFloat64 if use_float64 else sitk.sitkFloat32
    output = sitk.GetArrayFromImage(sitk.ReadImage(filepath, dtype))

    if not (output.ndim == 3 or output.ndim == 4):
        raise FileExistsError('File must contain a 3d or 4d array representable object')

    if pad:
        if pad_shape is None:
            pad_shape = (280, 52, 84, 84) if output.ndim == 4 else (176, 256, 256)
        elif len(pad_shape) != output.ndim:
            raise ValueError(f'pad_shape must be a tuple of {output.ndim} int values')

        pad_array = []
        for index in range(len(pad_shape)):
            if pad_shape[index] - output.shape[index] >= 0:
                pad_array.append((0, pad_shape[index] - output.shape[index]))
            else:
                shape = list(output.shape)
                shape[index] = pad_shape[index]
                if output.ndim == 3:
                    output = output[:shape[0], :shape[1], :shape[2]]
                else:
                    output = output[:shape[0], :shape[1], :shape[2], :shape[3]]
                pad_array.append((0, pad_shape[index] - output.shape[index]))

        output = np.pad(output, pad_array, mode='constant', constant_values=0.0)

    if normalize:
        output /= np.amax(output)

        if denoise:
            output[output < denoise_lower] = 0.0
            output[output > denoise_upper] = 0.0
            output /= np.amax(output)

        if normalize_range != (0, 1):
            output *= (normalize_range[1] - normalize_range[0])
            output += normalize_range[0]

    if expand_dims:
        output = np.expand_dims(output, axis=-1)

    return output


def load_dataset(filepath: str, fmri: bool = False, loading_status: bool = True, use_float64: bool = False,
                 pad: bool = True, pad_shape=None, normalize: bool = True, normalize_range=(0, 1),
                 expand_dims: bool = True, denoise: bool = False, denoise_lower: float = 0.05,
                 denoise_upper: float = 0.45) -> np.ndarray:
    scan_paths = list(map(lambda x: os.path.join(filepath, x, 'ses-01'),
                          [f for f in os.listdir(filepath) if 'sub' in f]))
    output = []
    fmri_list = []

    if fmri:
        scan_paths = (list(map(lambda x: os.path.join(x, 'func'), scan_paths)))
        scan_paths.sort()

        for path in scan_paths:
            temp_list = [os.path.join(path, f) for f in os.listdir(path) if '.nii' in f]
            temp_list.sort()
            fmri_list.append(temp_list[:5])

        for path_lists in fmri_list:
            temp_list = []
            for run_paths in path_lists:
                starttime = time.time()
                temp_list.append(
                    load_mri_scan(run_paths, use_float64=use_float64, pad=pad, pad_shape=pad_shape, normalize=normalize,
                                  normalize_range=normalize_range, expand_dims=expand_dims, denoise=denoise,
                                  denoise_lower=denoise_lower, denoise_upper=denoise_upper))
                if loading_status:
                    print(f'Loaded ({time.time() - starttime:0.3}s): ' + run_paths)
            output.append(temp_list)
    else:
        scan_paths = list(map(lambda x: os.path.join(x, 'anat'), scan_paths))
        scan_paths = list(map(lambda x: os.path.join(x, [f for f in os.listdir(x) if '.nii' in f][0]), scan_paths))
        scan_paths.sort()
        for path in scan_paths:
            starttime = time.time()
            output.append(
                load_mri_scan(path, use_float64=use_float64, pad=pad, pad_shape=pad_shape, normalize=normalize,
                              normalize_range=normalize_range, expand_dims=expand_dims, denoise=denoise,
                              denoise_lower=denoise_lower, denoise_upper=denoise_upper))
            if loading_status:
                print(f'Loaded ({time.time() - starttime:0.3}s): ' + path)

    return np.array(output)


def plot_slice(data_array: np.ndarray, slice_index: int, time_index: int = 0, step_size: int = 10, figsize=(10, 9),
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


def axial_slices(data_array) -> np.ndarray:  # Hiegh [-3] index is superior
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


def coronal_slices(data_array) -> np.ndarray:  # Higher [-3] index is anterior
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


def sagittal_slices(data_array) -> np.ndarray:  # Higher index is MRI left (POV right)
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


def process_png_overlay(filepath: str) -> np.ndarray:
    overlay = np.array(Image.open(filepath))

    overlay = overlay[:, :, :1]
    overlay[overlay < 3] = 0
    overlay[overlay > 239] = 0

    return overlay


def testingoverlay():
    data = load_mri_scan('sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz', denoise=True)
    data = sagittal_slices(data)

    overlay = process_png_overlay('11080.png')

    print(overlay.shape)

    plt.figure(figsize=(10, 9))
    plt.imshow(data[80], cmap='gray')
    plt.imshow(overlay, alpha=0.8)
    plt.show()


def testingloaddataset():
    path = 'ds003434'
    filepath = 'ds003434/sub-14/ses-01/anat/sub-14_ses-01_T1w.nii.gz'

    # data = load_mri_scan(filepath, denoise=True)
    # print(data.shape)
    # plot_slice(data, slice_index=110)

    dataset = load_dataset(path, fmri=True)
    print(dataset.shape)
    # plot_slice(dataset[10], slice_index=110)
    # plt.figure(figsize=(10, 9))
    # for num in range(16):
    #     plt.subplot(4, 4, num + 1)
    #     plt.imshow(dataset[num + 2][83], cmap='gray')
    #     plt.title(num + 2)
    #
    # plt.show()


# testingloaddataset()


def generaltesting():
    path = 'sub-01-ses-01-anat-sub-01_ses-01_T1w.nii.gz'
    fmripath = 'sub-01-ses-01-func-sub-01_ses-01_task-MainExp_run-01_bold.nii.gz'

    data = load_mri_scan(path, denoise=True, normalize=True)
    data = sagittal_slices(data)
    plot_slice(data, slice_index=40, time_index=100, step_size=10)

    print(type(data[0][0][0][0]))

    plt.figure(figsize=(10, 9))
    plt.hist(data.flatten())
    plt.show()
    # data = load_mri_scan(fmripath)
    # data = sagittal_slices(data)
    # plot_slice(data, slice_index=50)

    print(np.std(data[data != 0.0]))
    print(np.mean(data))
    print(data[100][100])


generaltesting()

# MRIProcessingDS003434
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI/MRI representations using dataset from https://openneuro.org/datasets/ds003434/versions/1.0.1
# Numpy array of fmri dataset (5 runs per subject) contains 9e9 elements
# Numpy array of mri dataset contains 2e8 elements

import numpy as np
import matplotlib.pyplot as plt
import os
import time

from PIL import Image
import sys

import niftiProcessing


def load_dataset_ds003434(dir_path: str, fmri: bool = False, loading_status: bool = True, use_float64: bool = False,
                          pad: bool = True, pad_shape=None, normalize: bool = True, normalize_range=(0, 1),
                          expand_dims: bool = True, denoise: bool = False, denoise_lower: float = 0.05,
                          denoise_upper: float = 0.45) -> np.ndarray:
    scan_paths = list(map(lambda x: os.path.join(dir_path, x, 'ses-01'),
                          [f for f in os.listdir(dir_path) if 'sub' in f]))
    output = []

    if fmri:
        scan_paths = (list(map(lambda x: os.path.join(x, 'func'), scan_paths)))
        scan_paths.sort()

        fmri_list = []
        for path in scan_paths:
            temp_list = [os.path.join(path, f) for f in os.listdir(path) if '.nii' in f]
            temp_list.sort()
            fmri_list.append(temp_list[:5])

        for path_lists in fmri_list:
            temp_list = []
            for run_paths in path_lists:
                starttime = time.time()
                temp_list.append(
                    niftiProcessing.load_mri_scan(run_paths, use_float64=use_float64, pad=pad, pad_shape=pad_shape,
                                                  normalize=normalize, normalize_range=normalize_range,
                                                  expand_dims=expand_dims, denoise=denoise, denoise_lower=denoise_lower,
                                                  denoise_upper=denoise_upper))
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
                niftiProcessing.load_mri_scan(path, use_float64=use_float64, pad=pad, pad_shape=pad_shape,
                                              normalize=normalize, normalize_range=normalize_range,
                                              expand_dims=expand_dims, denoise=denoise, denoise_lower=denoise_lower,
                                              denoise_upper=denoise_upper))
            if loading_status:
                print(f'Loaded ({time.time() - starttime:0.3}s): ' + path)

    return np.array(output)


def axial_ds003434(data_array) -> np.ndarray:  # Hiegh [-3] index is superior
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


def coronal_ds003434(data_array) -> np.ndarray:  # Higher [-3] index is anterior
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


def sagittal_ds003434(data_array) -> np.ndarray:  # Higher index is MRI left (POV right)
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


def load_mask(dir_path: str, file_ext: str = '.png', layered: bool = False, layer: int = 1, denoise: bool = True,
              denoise_cutoff: int = 50, expand_dims: bool = True, normalize_value: float = 255.) -> np.ndarray:
    image_paths = list(map(lambda x: os.path.join(dir_path, x), [f for f in os.listdir(dir_path) if file_ext in f]))
    image_paths.sort()

    overlay = []
    for paths in image_paths:
        overlay.append(np.array(Image.open(paths)))

    overlay = np.array(overlay)
    if layered:
        overlay = overlay[:, :, :, layer:layer + 1]
    if not expand_dims:
        overlay = np.squeeze(overlay, axis=-1)
    if denoise:
        overlay[overlay < denoise_cutoff] = 0.
        overlay[overlay >= denoise_cutoff] = normalize_value

    return overlay


def testingoverlay():
    ds_path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz'
    mask_path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/s1s1SagittalSlicesMasks'

    data = niftiProcessing.load_mri_scan(ds_path, denoise=True)
    overlay = load_mask(mask_path, layered=True)

    print(overlay.shape)
    # overlay = axial_ds003434(overlay)
    # data = axial_ds003434(data)

    mask = np.ma.masked_where(overlay == 0, overlay)
    print(overlay.shape)
    print(overlay[110][180])

    plt.figure(figsize=(10, 9))
    plt.imshow(data[85], cmap='gray')
    plt.imshow(mask[85], alpha=0.8)
    plt.show()


def testingloaddataset():
    path = 'OpenNeuroDS003434newbi4fmri2020/ds003434'
    filepath = 'OpenNeuroDS003434newbi4fmri2020/ds003434/sub-14/ses-01/anat/sub-14_ses-01_T1w.nii.gz'

    # data = load_mri_scan(filepath, denoise=True)
    # print(data.shape)
    # plot_slice(data, slice_index=110)

    dataset = load_dataset_ds003434(path, fmri=False)
    print(dataset.shape)
    print(sys.getsizeof(dataset))
    # niftiProcessing.plot_slice(dataset[10], slice_index=110)
    # plt.figure(figsize=(10, 9))
    # for num in range(16):
    #     plt.subplot(4, 4, num + 1)
    #     plt.imshow(dataset[num + 2][83], cmap='gray')
    #     plt.title(num + 2)
    #
    # plt.show()


def generaltesting():
    path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz'
    fmripath = 'sub-01-ses-01-func-sub-01_ses-01_task-MainExp_run-01_bold.nii.gz'

    data = niftiProcessing.load_mri_scan(path, denoise=True, normalize=True)
    data = sagittal_ds003434(data)
    niftiProcessing.plot_slice(data, slice_index=40, time_index=100, step_size=10)

    print(type(data[0][0][0][0]))
    print(sys.getsizeof(data))
    plt.figure(figsize=(10, 9))
    plt.hist(data[data != 0.0].flatten())
    plt.show()
    # data = load_mri_scan(fmripath)
    # data = sagittal_ds003434(data)
    # plot_slice(data, slice_index=50)

    print(np.std(data[data != 0.0]))
    print(np.mean(data))
    print(np.median(data[data != 0.0]))

# testingoverlay()
# generaltesting()

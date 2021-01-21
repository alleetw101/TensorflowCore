# niftiProcessing
# https://github.com/alleetw101/TensorflowCore <2020>


import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk


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
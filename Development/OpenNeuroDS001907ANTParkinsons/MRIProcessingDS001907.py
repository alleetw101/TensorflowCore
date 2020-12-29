# fMRIRepresentation
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI 3D projection using dataset from https://openneuro.org/datasets/ds001907/versions/2.0.3
# On OpenNeuro, slice(110) == x = 23, y = 10, z = -8
# MRI left/right is opposite of personal left/right

import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
import tensorflow as tf

import time

path = 'ds001907/sub-RC4101/ses-1/anat/sub-RC4101_ses-1_T1w.nii.gz'

img = sitk.ReadImage(path)
data = sitk.GetArrayFromImage(img)
data = tf.expand_dims(data, -1)
data /= np.amax(data)


def axial_slices3d(data_array) -> np.ndarray:
    return np.transpose(data_array, [1, 2, 0, 3])


def coronal_slices3d(data_array) -> np.ndarray:
    transposed = np.transpose(data_array, [2, 1, 0, 3])
    return np.flip(transposed, axis=[0, 1])


def sagittal_slices3d(data_array) -> np.ndarray:
    return np.flip(data_array, axis=1)


def upscale_slice(image) -> np.ndarray:
    output_height = (len(image) * 2) - 1
    output_width = (len(image[0]) * 2) - 1
    output_image = np.zeros([output_height, output_width, 1])
    image_height = len(image)
    image_width = len(image[0])

    # for height_index in range(len(image)):
    #     for width_index in range(len(image[0])):
    #         output_image[height_index * 2][width_index * 2] = image[height_index][width_index]

    row_nonzeros = []
    height_nonzeros = list(range(image_height))

    for height_index in range(image_height):
        templist = np.nonzero(image[height_index])[0]
        row_nonzeros.append(templist if len(templist) != 0 and templist[-1] != (image_width - 1) else templist[:-1])

    for height_index in range(image_height):
        if not row_nonzeros[height_index].any():
            height_nonzeros.remove(height_index)

    for x in np.transpose(np.nonzero(image)):
        output_image[x[0] * 2][x[1] * 2] = image[x[0]][x[1]]

    for height_index in height_nonzeros:
        for indices in row_nonzeros[height_index]:
            output_image[height_index * 2][(indices * 2) + 1][0] = (output_image[height_index * 2][
                                                                        (indices * 2)][0] +
                                                                    output_image[height_index * 2][
                                                                        (indices * 2 + 2)][0]) / 2.0

    # for height_index in height_nonzeros:
    #     for indices in np.unique(np.concatenate((row_nonzeros[height_index] * 2, row_nonzeros[height_index + 1] * 2, (row_nonzeros[height_index] * 2) + 1, (row_nonzeros[height_index + 1] * 2) + 1, row_nonzeros[height_index][0] - 1, row_nonzeros[height_index][0] - 1))):
    #         output_image[height_index * 2 + 1][indices][0] = (output_image[height_index * 2][indices][0] +
    #                                                           output_image[(height_index * 2) + 2][indices][
    #                                                               0]) / 2.0

    # for height_index in range(len(image)):
    #     for width_index in range(len(image[0]) - 1):
    #         output_image[height_index * 2][(width_index * 2) + 1][0] = (output_image[height_index * 2][(width_index * 2)][0] + output_image[height_index * 2][(width_index * 2 + 2)][0]) / 2.0

    for height_index in range(len(image) - 1):
        for width_index in range(output_width):
            output_image[height_index * 2 + 1][width_index][0] = (output_image[height_index * 2][width_index][0] + output_image[(height_index * 2) + 2][width_index][0]) / 2.0

    return output_image


def upscale_volume(image) -> np.ndarray:
    output_depth = (len(image) * 2) - 1
    output_height = (len(image[0]) * 2) - 1
    output_width = (len(image[0][0]) * 2) - 1
    output_image = np.zeros([output_depth, output_height, output_width, 1])

    for depth_index in range(len(image)):
        start = time.time()
        output_image[depth_index * 2] = upscale_slice(image[depth_index])
        print(time.time() - start)
        print(depth_index)

    return output_image


# Singular 2D Representation using matplotlib.pyplot
# start = time.time()
# pltimage = upscale_slice(axial_slices3d(data)[130])
# print(time.time() - start)
#
# start = time.time()
# pltimage = upscale_slice(sagittal_slices3d(data)[130])
# print(time.time() - start)
# # pltimage = upscale_volume(sagittal_slices3d(data))[130]
# plt.figure(figsize=(10, 10))
# plt.imshow(pltimage, cmap='gray')
# # plt.suptitle("axial, flip(axis=1) @ slice 110\nHigher index is MRI left")
# # plt.ylabel("Front")
# # plt.xlabel("Bottom")
# plt.show()
# print(type(pltimage))
# print(pltimage.dtype)
# print(pltimage.shape)

# axial = sagittal_slices3d(data)
# plt.figure(figsize=(10, 10))
# for i in range(16):
#     plt.subplot(4, 4, i + 1)
#     plt.yticks([])
#     plt.xticks([])
#     plt.title(f'{i * 10}')
#     plt.imshow(axial[i * 10], cmap='gray')
#
# plt.show()


# 3D Monocolor Representation using matplotlib.pyplot
# representationarray = []
#
# axial = axial_slices3d(mri_image)
# for index in range(len(axial)):
#     for index2 in range(len(axial[0])):
#         for index3 in range(len(axial[0][0])):
#             if axial[index][index2][index3] != 0.0:
#                 representationarray.append([index, index2, index3, axial[index][index2][index3]])
#
# representationarray = np.array(representationarray)
#
# # mri_image=np.array(np.random.random((100,3)))
# x=representationarray[:, 0]
# y=representationarray[:, 1]
# z=representationarray[:, 2]
#
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.scatter(x,y,z)
# plt.show()

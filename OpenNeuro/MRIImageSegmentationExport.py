# MRIImageSegmentationExport
# https://github.com/alleetw101/TensorflowCore <2021>

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from PIL import Image

import niftiProcessing


def plot_test():
    sample_path = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-08/ses-01/anat/sub-08_ses-01_T1w.nii.gz'
    sample = niftiProcessing.load_mri_scan(sample_path, denoise=True)
    sample = np.concatenate((sample,)*3, axis=-1)
    sample_image = sample[85]

    model = tf.keras.models.load_model('/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/SavedModelMRIImageSegmentation200205_1')
    prediction = model.predict(sample_image[tf.newaxis, ...])


    prediction = prediction[0][..., :1]
    prediction[prediction > 0] = 255
    prediction[prediction < 0] = 0
    prediction = np.ma.masked_where(prediction == 0, prediction)

    model1 = tf.keras.models.load_model('/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/SavedModelMRIImageSegmentation20210209_2')
    prediction1 = model1.predict(sample_image[tf.newaxis, ...])
    print(prediction1[0][80])

    prediction1[prediction1 < 0] = 0
    prediction1[prediction1 > 0] = 255
    prediction1 = prediction1[0][..., :1]
    prediction1 = np.ma.masked_where(prediction1 == 0, prediction1)

    plt.figure(figsize=(9, 9))
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 1)
    plt.imshow(sample_image, cmap='gray')
    plt.imshow(prediction, alpha=0.5)
    plt.subplot(1, 2, 2)
    plt.imshow(sample_image, cmap='gray')
    plt.imshow(prediction1, alpha=0.5)
    # plt.show()

    # print(prediction[80])

    # sample_image = sample_image[..., :1]
    # png = np.concatenate((sample_image, prediction), axis=-1)
    # print(prediction.shape)
    # print(sample_image.shape)
    # print(png.shape)
    # print(png[80])


def png_export(filepath: str, save_dir: str):
    data = niftiProcessing.load_mri_scan(filepath, denoise=True)
    sample = np.concatenate((data,)*3, axis=-1)
    model = tf.keras.models.load_model(
        '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/SavedModelMRIImageSegmentation200205_1')

    for index in range(len(sample)):
        prediction = model.predict(sample[index][tf.newaxis, ...])
        prediction = prediction[0][..., :1]
        prediction[prediction > -10] = 255
        prediction[prediction < -10] = 0

        image = data[index]
        image = image * 255

        mask_name = f's03s01-mask-{index:03d}.png'
        export_path = os.path.join(save_dir, 'masks', mask_name)

        mask_array = np.concatenate((image, prediction), axis=-1)
        mask_array = mask_array.astype(np.uint8)
        mask = Image.fromarray(mask_array)
        mask.save(export_path)

        image_name = f's03s01-orig-{index:03d}.png'
        orig_export_path = os.path.join(save_dir, 'orig', image_name)

        image = np.squeeze(image, axis=-1)
        image = Image.fromarray(image)
        image = image.convert('L')
        image.save(orig_export_path, transparency=0.0)

        if index % 10 == 0:
            print(index)


def png_denoise(dir_path: str, save_dir: str):
    image_paths = list(map(lambda x: os.path.join(dir_path, x), [f for f in os.listdir(dir_path) if 'png' in f]))
    image_names = [f for f in os.listdir(dir_path) if 'png' in f]
    image_paths.sort()
    image_names.sort()

    for index in range(len(image_paths)):
        img_array = np.array(Image.open(image_paths[index]))

        img = img_array[:, :, :1]
        mask = img_array[:, :, 1:]

        # img = np.squeeze(img, axis=-1)
        # mask = np.squeeze(mask, axis=-1)

        for index_x in range(len(img)):
            for index_y in range(len(img[index_x])):
                if img[index_x][index_y] <= 40:
                    mask[index_x][index_y] = 0

        export_img = np.concatenate((img, mask), axis=-1)
        export_path = os.path.join(save_dir, image_names[index])
        export_img = export_img.astype(np.uint8)
        export_img = Image.fromarray(export_img)
        export_img.save(export_path)




plot_test()
# file = '/Users/alan/Documents/Programming/Python/TensorflowCore/OpenNeuro/OpenNeuroDS003434newbi4fmri2020/ds003434/sub-03/ses-01/anat/sub-03_ses-01_T1w.nii.gz'
# save_dir = '/Users/alan/Desktop/s3s1'
# png_export(file, save_dir)

# dir_path = '/Users/alan/Desktop/Colab/DS003434/s2s1/masks'
# save_dir = '/Users/alan/Desktop/Colab/DS003434/s2s1/masks-2'
# png_denoise(dir_path, save_dir)
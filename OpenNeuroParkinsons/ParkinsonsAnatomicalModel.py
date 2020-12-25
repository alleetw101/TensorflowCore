# fMRIRepresentation
# https://github.com/alleetw101/TensorflowCore <2020>
#
# fMRI dataset from https://openneuro.org/datasets/ds001907/versions/2.0.3 (excludes /derivative)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import SimpleITK as sitk
import os.path
import matplotlib.pyplot as plt
import time

import fMRIRepresentation

dataset_path = 'ds001907'

train_subject_paths = list(
    map(lambda x: os.path.join(dataset_path, x), [f for f in os.listdir(dataset_path) if 'sub' in f]))
test_subject_paths = ['ds001907/sub-RC4208', 'ds001907/sub-RC4206', 'ds001907/sub-RC4112', 'ds001907/sub-RC4115']
dev_subject_paths = ['ds001907/sub-RC4130', 'ds001907/sub-RC4106', 'ds001907/sub-RC4224', 'ds001907/sub-RC4212']

class_names = ["Healty", "Parkinson's"]

for sample in test_subject_paths + dev_subject_paths:
    train_subject_paths.remove(sample)


def create_dataset(subject_paths):
    labels = []
    for paths in subject_paths:
        if 'C42' in paths:
            labels.append(1)
        else:
            labels.append(0)

    subjects_count = len(subject_paths)
    double_subject_paths = subject_paths * 2
    scan_paths = (
            list(map(lambda x: os.path.join(x, 'ses-1', 'anat'), double_subject_paths[:subjects_count])) +
            list(map(lambda x: os.path.join(x, 'ses-2', 'anat'), double_subject_paths[subjects_count:])))
    scan_paths = list(
        map(lambda x: os.path.join(x, [f for f in os.listdir(x) if '.nii' in f][0]), scan_paths))

    examples_list = []
    for sample_paths in scan_paths:
        mri_image = sitk.GetArrayFromImage(sitk.ReadImage(sample_paths, sitk.sitkFloat64))
        mri_image /= np.amax(mri_image)
        examples_list.append(mri_image)

    labels = []
    for paths in scan_paths:
        if 'C42' in paths:
            labels.append(1)
        else:
            labels.append(0)

    examples = tf.expand_dims(np.array(examples_list), -1)
    labels = np.array(labels)

    return tf.data.Dataset.from_tensor_slices((examples, labels))


train_ds = create_dataset(train_subject_paths)
dev_ds = create_dataset(dev_subject_paths)
test_ds = create_dataset(test_subject_paths)

print(train_ds.cardinality().numpy())
print(train_ds)
print(dev_ds.cardinality().numpy())
print(dev_ds)
print(test_ds.cardinality().numpy())
print(test_ds)

for example, label in train_ds.take(1):
    axial = fMRIRepresentation.axial_slices3d(example)
    plt.figure(figsize=(10, 10))
    for i in range(16):
        plt.subplot(4, 4, i + 1)
        plt.yticks([])
        plt.xticks([])
        plt.title(f'{i * 10} + {class_names[label]}')
        plt.imshow(axial[i * 10], cmap='gray')

    plt.show()

AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.cache().shuffle(32).batch(4).prefetch(buffer_size=AUTOTUNE)
dev_ds = dev_ds.cache().shuffle(8).batch(4).prefetch(buffer_size=AUTOTUNE)

model = keras.models.Sequential([
    layers.Conv3D(16, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001),
                  input_shape=(176, 256, 256, 1)),
    layers.MaxPooling3D(),
    layers.Conv3D(32, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling3D(),
    layers.Conv3D(64, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling3D(),
    layers.Conv3D(128, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.MaxPooling3D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
    layers.Dense(2)
])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
epochs = 1
train_history = model.fit(train_ds, validation_data=dev_ds, epochs=epochs)

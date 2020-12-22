# euroSATModel
# https://github.com/alleetw101/TensorflowCore <2020>
#
# Prediction model trained using the EuroSAT dataset from https://www.tensorflow.org/datasets/catalog/eurosat.
# Characterizes 64x64 RGB satellite images into 10 geographical landmarks.
# AnnualCrop, Forest, HerbaceousVegetation, Highway, Industrial, Pasture, PermanentCrop, Residential, River, Sea/Lake
# TODO: load images into dataset using tf.data (finer control) over keras image_dataset_from_directory()

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os.path
import time

# Load dataset
dataset_path = "euroSAT_Dataset"
batch_size = 128
train_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(dataset_path, "train"), seed=2020,
                                                            image_size=(64, 64), batch_size=batch_size)
dev_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(dataset_path, "dev"), seed=2020,
                                                          image_size=(64, 64), batch_size=batch_size)
test_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(dataset_path, "test"), seed=2020,
                                                           image_size=(64, 64), batch_size=batch_size)
class_names = train_ds.class_names
num_classes = len(class_names)
print(class_names)

# Visualize dataset
# plt.figure(figsize=(10, 10))
# for sample, sample_label in train_ds.take(1):
#     for i in range(9):
#         _ = plt.subplot(3, 3, i + 1)
#         plt.imshow(sample[i].numpy().astype("uint8"))
#         plt.title(class_names[sample_label[i]])

# plt.show()
for sample, sample_label in train_ds.take(1):
    print(sample.shape)

# Optimize performance
AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
dev_ds = dev_ds.cache().prefetch(buffer_size=AUTOTUNE)


regularizer = keras.regularizers.l2(0.001)
# Create model
model = keras.models.Sequential([
    layers.experimental.preprocessing.Rescaling(1./255, input_shape=(64, 64, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=regularizer),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=regularizer),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=regularizer),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=regularizer),
    layers.MaxPooling2D(),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(256, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(128, activation='relu', kernel_regularizer=regularizer),
    layers.Dense(num_classes)
])

learning_rate = 0.001
optimizer = "Adam"
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

epochs = 100
history = model.fit(train_ds, validation_data=dev_ds, epochs=epochs)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='lower left')
plt.title('Loss')

time = time.strftime('%y%m%d_%H%M', time.localtime())

plt.suptitle(f'euroSATModel<{time}>\nBatch size: {batch_size}, Optimizer: {optimizer}(lr: {learning_rate}), ' +
             f'\nRegularizer: l2(0.001), Epochs: {epochs}')
plt.savefig(os.path.join("euroSAT_Training_Graphs", time))

plt.show()

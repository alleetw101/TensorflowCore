# MRIImageSegmentation
# https://github.com/alleetw101/TensorflowCore <2021>

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

import niftiProcessing
import MRIProcessingDS003434

# Preprocessing
ds_path ='OpenNeuroDS003434newbi4fmri2020/ds003434/sub-01/ses-01/anat/sub-01_ses-01_T1w.nii.gz'
mask_path = 'OpenNeuroDS003434newbi4fmri2020/s1s1SagittalSlicesMasks'
scans = niftiProcessing.load_mri_scan(ds_path, use_float64=False, denoise=True)
masks = MRIProcessingDS003434.load_mask(mask_path, layered=True)


def conversion(image, mask):
    return tf.image.grayscale_to_rgb(image), tf.image.grayscale_to_rgb(tf.cast(mask, tf.float32))


ds = tf.data.Dataset.from_tensor_slices((scans, masks)).shuffle(20).map(conversion)

batch_size = 4
val_size = int(0.1 * len(scans))
test_size = int(0.1 * len(scans))
train_size = int(len(scans) - val_size - test_size)

train_ds = ds.take(train_size).cache().batch(batch_size).repeat()
val_ds = ds.skip(train_size).take(val_size).batch(batch_size)
test_ds = ds.skip(train_size + val_size).batch(batch_size)


def display(display_list):
    plt.figure(figsize=(9, 9))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]))
        plt.axis('off')
    plt.show()


for image, mask in train_ds.skip(7).take(1):
    display([image[0], mask[0]])
    sample_image, sample_mask = image[0], mask[0]

STEPS_PER_EPOCH = len(scans) // batch_size
OUTPUT_CHANNELS = 3

base_model = tf.keras.applications.MobileNetV2(input_shape=[256, 256, 3], include_top=False)

layer_names = [
    'block_1_expand_relu',  # 64x64
    'block_3_expand_relu',  # 32x32
    'block_6_expand_relu',  # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project'
]
layers = [base_model.get_layer(name).output for name in layer_names]

down_stack = tf.keras.Model(inputs=base_model.input, outputs=layers)
down_stack.trainable = False


def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2DTranspose(filters, size, strides=2, padding='same', kernel_initializer=initializer,
                                        use_bias=False))
    result.add(tf.keras.layers.BatchNormalization())

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))
    result.add(tf.keras.layers.ReLU())

    return result


up_stack = [
    upsample(512, 3),
    upsample(256, 3),
    upsample(128, 3),
    upsample(64, 3),
]


def unet_model(output_channels):
    inputs = tf.keras.layers.Input(shape=[256, 256, 3])
    x = inputs

    # Downsampling
    skips = down_stack(x)
    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establing skip connections
    for up, skip in zip(up_stack, skips):
        x = up(x)
        concat = tf.keras.layers.Concatenate()
        x = concat([x, skip])

    # Last layer
    last = tf.keras.layers.Conv2DTranspose(output_channels, 3, strides=2, padding='same')

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


model = unet_model(OUTPUT_CHANNELS)
model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])



class DisplayCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        show_predictions()
        print(f'\nSample Prediction after epoch {epoch + 1}\n')


EPOCHS = 50
VAL_SUBSPLITS = 5
VALIDATION_STEPS = len(scans) // batch_size // VAL_SUBSPLITS

model_history = model.fit(train_ds, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_data=val_ds,
                          callbacks=[DisplayCallback()])
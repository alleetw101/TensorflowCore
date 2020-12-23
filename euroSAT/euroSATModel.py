# euroSATModel
# https://github.com/alleetw101/TensorflowCore <2020>
#
# Prediction model trained using the EuroSAT dataset from https://www.tensorflow.org/datasets/catalog/eurosat.

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as plt
import os.path


class EuroSAT:
    """A prediction model to classify geogrphical landmarks in images

        A prediction model trained using the EuroSAT dataset from https://www.tensorflow.org/datasets/catalog/eurosat.
        Characterizes 64x64 RGB satellite images into 10 geographical landmarks.
        ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture', 'PermanentCrop',
        'Residential','River', 'SeaLake']

    Attributes:
        model: A keras Sequential model
        batch_size: An integer specifying dataset batch size
        train_ds: A tensorflow Dataset for model training
        dev_ds: A tensorflow Dataset for model training validation
        test_ds: A tensorflow Dataset for model evaluation
    """
    model = keras.models.Sequential([
        layers.experimental.preprocessing.Rescaling(1. / 255, input_shape=(64, 64, 3)),
        layers.Conv2D(16, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, padding='same', activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dense(128, activation='relu', kernel_regularizer=keras.regularizers.l2(0.001)),
        layers.Dense(10)
    ])

    def __init__(self, usetfds: bool = True, datasetpath: str = "euroSAT_Dataset", batchsize: int = 64):
        """Initialization with dataset processing

        Initializes EuroSAT class and splits dataset for training, validation, and evaluation. Obtains euroSAT dataset
        from tensorflow_dataset or file system.

        Args:
            usetfds: Optional; A boolean. If usetfds is true, class will obtain and use dataset through
                tensorflow_dataset
            datasetpath: Optional; A string directory path for euroSAT dataset if tensorflow_dataset is not used
            batchsize: Optional; An integer specifying dataset batch size
        """
        self.batch_size = batchsize

        if usetfds:
            ds = tfds.load("eurosat", shuffle_files=False, split='train', as_supervised=True, with_info=False)

            num_images = ds.cardinality().numpy()
            train_size = int(num_images * 0.7)
            dev_size = int(num_images * 0.2)
            test_cutoff = int(num_images * 0.9)

            train_dev_ds = ds.take(test_cutoff).shuffle(test_cutoff)
            self.train_ds = train_dev_ds.take(train_size).batch(self.batch_size)
            self.dev_ds = train_dev_ds.skip(train_size).take(dev_size).batch(self.batch_size)
            self.test_ds = ds.skip(test_cutoff).batch(self.batch_size)
        else:
            self.train_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(datasetpath, "train"),
                                                                             seed=2020, image_size=(64, 64),
                                                                             batch_size=self.batch_size)
            self.dev_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(datasetpath, "dev"),
                                                                           seed=2020, image_size=(64, 64),
                                                                           batch_size=self.batch_size)
            self.test_ds = keras.preprocessing.image_dataset_from_directory(os.path.join(datasetpath, "test"),
                                                                            seed=2020, image_size=(64, 64),
                                                                            batch_size=self.batch_size)

        AUTOTUNE = tf.data.experimental.AUTOTUNE
        self.train_ds = self.train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
        self.dev_ds = self.dev_ds.cache().prefetch(buffer_size=AUTOTUNE)

    def trainmodel(self, epochs: int = 120, graph: bool = True, savegraph: bool = False,
                   savegraphpath: str = 'euroSATModelTraining', savemodel: bool = False,
                   savemodeloverwrite: bool = False, savemodelpath: str = 'euroSATSavedModel'):
        """Train EuroSAT model

        Train tensorflow Sequential model with euroSAT dataset. Training can be visualized using matplotlib graph
        and saved in a SavedModel.

        Args:
            epochs: Optional; An integer specifying number of times to iterate over dataset during training
            graph: Optional; If graph is true, a matplotlib graph with training loss and accuracy will be shown
            savegraph: Optional; If savegraph is true, a training graph will be saved as a png
            savegraphpath: Optional; A string specifying where to save the training graph
            savemodel: Optional; If savemodel is true, a SavedModel will be generated from the trained model
            savemodeloverwrite: Optional; If savemodeloverwrite is true, previous SavedModels will be replaced
            savemodelpath: Optional; A string specifying where to save the SavedModel
        """
        self.model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

        train_history = self.model.fit(self.train_ds, validation_data=self.dev_ds, epochs=epochs)

        if savemodel:
            self.model.save(savemodelpath, overwrite=savemodeloverwrite)

        if graph:
            self.__graphtraining(train_history, epochs, savegraph, savegraphpath)

    def __graphtraining(self, history, epochs, savegraph, savegraphpath):
        """Graph training loss and accuracy

        Graphs training loss and accuracy via matplotlib

        Args:
            history: A tensorflow history object generated during model training from tf.keras.Models.fit()
            epochs: An integer specifying number of times to iterate over dataset during training
            savegraph: If savegraph is true, a training graph will be saved as a png
            savegraphpath: A string specifying where to save the training graph
        """
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

        plt.suptitle(f'euroSATModel\nBatch size: {self.batch_size}, Epochs: {epochs}')

        if savegraph:
            plt.savefig(savegraphpath)

        plt.show()


def eurosatclassification(savedmodelpath: str, imagepath: str):
    """Landmark classification of image

    Classifies image using a EuroSAT SavedModel

    Args:
        savedmodelpath: A string specifying lcoation of EuroSAT SavedModel
        imagepath: A string specifying lcoation of EuroSAT SavedModel
    """
    newmodel = tf.keras.models.load_model(savedmodelpath)
    class_names = ['AnnualCrop', 'Forest', 'HerbaceousVegetation', 'Highway', 'Industrial', 'Pasture',
                   'PermanentCrop', 'Residential', 'River', 'SeaLake']

    image = keras.preprocessing.image.load_img(imagepath, target_size=(64, 64))
    image_array = keras.preprocessing.image.img_to_array(image)
    image_array = tf.expand_dims(image_array, 0)

    predictions = newmodel.predict(image_array)
    score = tf.nn.softmax(predictions[0])
    print(f"This image most likely belongs to {class_names[np.argmax(score)]} with a {100 * np.max(score)} " +
          f"percent confidence.")

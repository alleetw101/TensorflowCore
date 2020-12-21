# euroSATRawDatasetProcessing
# https://github.com/alleetw101/TensorflowCore <2020>
#
# Splits EuroSAT dataset from https://www.tensorflow.org/datasets/catalog/eurosat into train, dev, and test directories.

import os.path
import shutil


def splitDownloadedEuroSATDataset(train: float = 0.8, dev: float = 0.1, test: float = 0.1):
    # Split ratios
    train_split = train
    dev_split = dev
    test_split = test

    folder_path = "euroSAT_Dataset"
    folders = [f for f in os.listdir(folder_path) if f != '.DS_Store']

    # Train, dev, and test directory paths
    train_folder_path = os.path.join(folder_path, "train")
    dev_folder_path = os.path.join(folder_path, "dev")
    test_folder_path = os.path.join(folder_path, "test")

    if not os.path.exists(test_folder_path):  # Check to ensure dataset has not already been split

        # Creates train, dev, and test directories
        for new_folder_paths in [train_folder_path, dev_folder_path, test_folder_path]:
            if not os.path.exists(new_folder_paths):
                os.makedirs(new_folder_paths)

        # Moves images into correct split directories
        for foldername in folders:

            # Note: Images are shuffled within images list and do not correspond with image integer-naming
            images = [f for f in os.listdir(os.path.join(folder_path, foldername)) if f != '.DS_Store']

            # Creates laballed directory from image name "Forest_123"
            new_folder_name = images[0].split("_")[0]
            for new_folder_paths in [train_folder_path, dev_folder_path, test_folder_path]:
                if not os.path.exists(os.path.join(new_folder_paths, new_folder_name)):
                    os.makedirs(os.path.join(new_folder_paths, new_folder_name))

            # Integer split of images list within characterized directories
            train_dev_split = int(len(images) * train_split)
            dev_test_split = train_dev_split + int(len(images) * dev_split)

            for image in images[:train_dev_split]:  # Train
                old_path = os.path.join(folder_path, foldername, image)
                new_path = os.path.join(train_folder_path, new_folder_name, image)
                shutil.move(old_path, new_path)

            for image in images[train_dev_split:dev_test_split]:  # Dev
                old_path = os.path.join(folder_path, foldername, image)
                new_path = os.path.join(dev_folder_path, new_folder_name, image)
                shutil.move(old_path, new_path)

            for image in images[dev_test_split:]:  # Test
                old_path = os.path.join(folder_path, foldername, image)
                new_path = os.path.join(test_folder_path, new_folder_name, image)
                shutil.move(old_path, new_path)

            # Removes old, empty directory
            shutil.rmtree(os.path.join(folder_path, foldername))

    print(f'Dataset split complete. Train: {train_split}, Dev: {dev_split}, Test: {test_split}.')

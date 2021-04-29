"""Main module."""

import os
import sys
import SimpleITK as sitk
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader:
    """[summary]"""

    def __init__(
        self,
        imgA_label,
        imgB_label,
        data_dir="./Data",
        imgs_subdir="Images",
        cache_mode="prod",
        extract_only=None,
    ):
        """[summary]

        Args:
            imgA_label (str): Identifier for class A. It's the name of the
                folder inside :py:attr:`imgs_subdir` that contains images
                labeled as class A.
            imgB_label (str): Identifier for class B. It's the name of the
                folder inside :py:attr:`imgs_subdir` that contains images
                labeled as class B.
            data_dir (str, optional): Path to directory that contains the
                Dataset. This folder **must** contain a subfolder named like
                :py:attr:`imgs_subdir`. Defaults to './Data'.
            imgs_subdir (str, optional): Name (**not** the entire path) of the
                folder that actually contains :py:attr:`imgA_label` and
                :py:attr:`imgB_label` subfolders. It's a subfolder of Defaults
                to 'Images'.
            cache_mode (str, optional): One between "prod" or "test". "test" is
                used as a debug modality only in case you want to create Cache
                directory at the current folder location. "prod" creates a
                subfolder named "Cache" inside :py:attr:`data_dir`. Defaults to
                "prod".
            extract_only (int, optional): Indicate wheter to partially cache a
                certain amount of elements in the dataset. Please remember that
                if "Cache" folder is already populated, you need to clean this
                folder content to recreate a partial cache file. When it is
                set to None, the entire Dataset is cached. Defaults to None.

        Raises:
            FileNotFoundError: :py:attr:`data_dir` doesn't exists.
            FileNotFoundError: :py:attr:`imgs_subdir` doesn't exists.
            FileNotFoundError: :py:attr:`imgA_label` doesn't exists.
            FileNotFoundError: :py:attr:`imgB_label` doesn't exists.
            ValueError: :py:attr:`cache_mode` is not "prod" nor "test".
        """

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"{data_dir} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir)):
            raise FileNotFoundError(f"{imgs_subdir} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir, imgA_label)):
            raise FileNotFoundError(f"{imgA_label} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir, imgB_label)):
            raise FileNotFoundError(f"{imgB_label} does not exist")

        self.data_dir = data_dir
        self.imgs_subdir = imgs_subdir
        self.imgA_label = imgA_label
        self.imgB_label = imgB_label

        if cache_mode == "prod":
            self.cache_dir = os.path.join(self.data_dir, "Cache")
        elif cache_mode == "test":
            self.cache_dir = "./Test_Cache"
        else:
            raise ValueError("Cache mode can be only 'prod' or 'test'")
        os.makedirs(self.cache_dir, exist_ok=True)

        self.imgA_paths, self.imgB_paths = self.get_imgs_paths()

        if extract_only is not None:
            self.imgA_paths = self.imgA_paths[:extract_only]
            self.imgB_paths = self.imgB_paths[:extract_only]

    def get_dataset(self,
                    input_size,
                    batch_size=32,
                    norm_bounds=None,
                    augmentation=False,
                    random_crop_size=None,
                    random_rotate=False,
                    random_flip=False):

        ds = tf.data.Dataset.zip((self.get_imgs(img_paths=self.imgA_paths,
                                                norm_bounds=norm_bounds),
                                  self.get_imgs(img_paths=self.imgB_paths,
                                                norm_bounds=norm_bounds)
                                  ))

        ds = ds.map(lambda imgA, imgB: self.check_dims(imgA,
                                                       imgB,
                                                       input_size),
                    num_parallel_calls=AUTOTUNE)

        if augmentation:
            if random_crop_size:
                ds = ds.map(
                    lambda imgA, imgB: self.random_crop(imgA,
                                                        imgB,
                                                        random_crop_size),
                    num_parallel_calls=AUTOTUNE)
            if random_rotate:
                ds = ds.map(self.random_rotate, num_parallel_calls=AUTOTUNE)
            if random_flip:
                ds = ds.map(self.random_flip, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def get_imgs(self, img_paths, norm_bounds=None):
        """Open image files for one class and store it inside cache.

        This function performs all the (usually) slow reading operations that
        is necessary to execute at least the first time. After the first
        execution information are saved inside some cache file inside Cache
        folder (typically created in your Dataset folder, at the same level of
        Images folder). This function detects if cache files are already
        present, and in that case it skips the definition of these files.
        Please take into account that cache files will be as big as your
        Dataset overall size. First execution may result in a considerably
        bigger amount of time.

        Args:
            img_paths (str): Path to single class images.

        Returns:
            tf.Data.Dataset: Tensorflow dataset object containing images of one
                classes converted in Tensor format, without any other
                computations.
        """
        # Get parent folder name from first element in img_paths
        img_label = os.path.basename(os.path.split(img_paths[0])[0])

        cache_file = os.path.join(self.cache_dir, f"{img_label}.cache")
        index_file = f"{cache_file}.index"

        ds = tf.data.Dataset.from_tensor_slices(img_paths)
        ds = ds.map(lambda path: tf.py_function(self.open_img,
                                                [path],
                                                [tf.float32]),
                    num_parallel_calls=AUTOTUNE)

        if norm_bounds is not None:
            ds = ds.map(lambda imgA: self.norm_with_bounds(imgA,
                                                           norm_bounds),
                        num_parallel_calls=AUTOTUNE
                        )
        ds = ds.cache(cache_file)

        if not os.path.exists(index_file):
            self._populate_cache(ds, cache_file, len(img_paths))

        return ds

    def get_imgs_paths(self):
        """Get paths of every single image divided by classes.

        Returns:
            list, list: two list containing the paths of every images for both
                classes. The list is sorted alphabetically, this can be usefull
                when images are named with a progressive number inside a folder
                (e.g.: 001.xxx, 002.xxx, ..., 999.xxx)
        """
        # print("Fetching images paths...")
        subset_dir = os.path.join(self.data_dir, self.imgs_subdir)

        subset_dir_imgA = os.path.join(subset_dir, self.imgA_label)
        subset_dir_imgB = os.path.join(subset_dir, self.imgB_label)

        filenames_imgA = os.listdir(subset_dir_imgA)
        filenames_imgB = os.listdir(subset_dir_imgB)

        paths_imgA = [os.path.join(subset_dir_imgA, img)
                      for img in filenames_imgA]
        paths_imgB = [os.path.join(subset_dir_imgB, img)
                      for img in filenames_imgB]

        # print("Images paths collected.")

        # Sort paths alphabetically
        paths_imgA.sort()
        paths_imgB.sort()

        return paths_imgA, paths_imgB

    def open_img(self, path):
        """Open an image file and convert it to a tensor.

        Args:
            path (tf.Tensor): Tensor containing the path to the file to be
                opened.

        Returns:
            tf.Tensor: Tensor containing the actual image content.
        """

        path = path.numpy().decode("utf-8")
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        tensor = tf.convert_to_tensor(image)
        return tf.expand_dims(tensor, axis=-1)

    def _populate_cache(self, ds, cache_file, num_tot):
        print(f"Caching decoded images in {cache_file}...")
        i = 0
        for _ in ds:
            i += 1
            sys.stdout.write("\r")
            sys.stdout.write(f"{i}/{num_tot}")
            sys.stdout.flush()
        print(f"\nCached decoded images in {cache_file}.")

    # -----------------------------------------------------------
    #  Transformations
    # -----------------------------------------------------------
    @staticmethod
    def check_dims(imgA, imgB, size):
        imgA = tf.expand_dims(tf.squeeze(imgA), axis=-1)
        imgA = tf.image.resize_with_pad(imgA, size, size)

        imgB = tf.expand_dims(tf.squeeze(imgB), axis=-1)
        imgB = tf.image.resize_with_pad(imgB, size, size)

        return imgA, imgB

    @staticmethod
    def norm_with_bounds(image, bounds):
        """Image normalisation. Normalises image in the range defined by lb and
        ub to fit [0, 1] range."""
        epsilon = 1e-8
        lb = bounds[0]
        ub = bounds[1]

        tf.where(image < lb, lb, image)
        tf.where(image > ub, ub, image)

        image = image - lb
        image /= (ub - lb) + epsilon

        return image

    @staticmethod
    def resize(imgA, imgB, size=256):
        imgA = tf.image.resize(imgA, (size, size))
        imgB = tf.image.resize(imgB, (size, size))

        return imgA, imgB

    @staticmethod
    def random_crop(imgA, imgB, crop_size=256):
        stacked_img = tf.stack([imgA, imgB], axis=0)
        cropped_img = tf.image.random_crop(
            stacked_img, size=[2, crop_size, crop_size, 1]
        )
        cropped_img = tf.split(cropped_img, 2)
        imgA = tf.expand_dims(tf.squeeze(cropped_img[0]), axis=-1)
        imgB = tf.expand_dims(tf.squeeze(cropped_img[1]), axis=-1)
        return imgA, imgB

    @staticmethod
    def random_flip(imgA, imgB):
        if tf.random.uniform(()) > 0.5:
            imgA = tf.image.flip_left_right(imgA)
            imgB = tf.image.flip_left_right(imgB)
        return imgA, imgB

    @staticmethod
    def random_rotate(imgA, imgB):
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        imgA = tf.image.rot90(imgA, k=rn)
        imgB = tf.image.rot90(imgB, k=rn)
        return imgA, imgB


def generate_dataset(data_loader,
                     input_size,
                     percentages,
                     batch_size,
                     norm_bounds=None,
                     train_augmentation=True,
                     random_crop_size=None,
                     random_rotate=True,
                     random_flip=True,
                     ):

    if len(percentages) != 3:
        raise ValueError("Percentages has to be a list of 3 elements")

    if percentages[0] + percentages[1] + percentages[2] != 1.0:
        raise ValueError("Sum of percentages has to be 1")

    num_imgsA = len(data_loader.imgA_paths)
    num_imgsB = len(data_loader.imgB_paths)
    if num_imgsA != num_imgsB:
        raise ValueError("The CBCT and CT subsets have different dimension!")

    train_ends = int(num_imgsA * percentages[0])

    valid_begins = train_ends
    valid_ends = valid_begins + int(num_imgsA * percentages[1])

    complete_ds = data_loader.get_dataset(batch_size=batch_size,
                                          norm_bounds=norm_bounds,
                                          input_size=input_size,
                                          augmentation=train_augmentation,
                                          random_crop_size=random_crop_size,
                                          random_rotate=random_rotate,
                                          random_flip=random_flip)

    complete_ds = complete_ds.unbatch()

    # Train Datasets
    train_ds = complete_ds.take(train_ends)
    train_ds = train_ds.batch(batch_size)

    complete_ds = data_loader.get_dataset(batch_size=batch_size,
                                          norm_bounds=norm_bounds,
                                          input_size=input_size,
                                          augmentation=False)

    complete_ds = complete_ds.unbatch()

    # Validation Datasets
    valid_ds = complete_ds.take(valid_ends)
    valid_ds = valid_ds.skip(valid_begins)
    valid_ds = valid_ds.batch(batch_size)

    # Test Datasets
    test_ds = complete_ds.skip(valid_ends)
    test_ds = test_ds.batch(batch_size)

    return train_ds, valid_ds, test_ds

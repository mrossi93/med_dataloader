"""Main module."""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
import SimpleITK as sitk
import json
import numpy as np
import sys

AUTOTUNE = tf.data.experimental.AUTOTUNE

__dataloader_modality__ = ["get", "gen"]
__dict_dtype__ = {"int8": tf.int8,
                  "int16": tf.int16,
                  "int32": tf.int32,
                  "int64": tf.int64,
                  "uint8": tf.uint8,
                  "uint16": tf.uint16,
                  "uint32": tf.uint32,
                  "uint64": tf.uint64,
                  "float32": tf.float32,
                  "float64": tf.float64,
                  }


class DataLoader:
    """[summary]"""

    def __init__(
        self,
        mode,
        data_path,
        imgA_label=None,
        imgB_label=None,
        img_size=None,
        output_dir=None,
        is_B_categorical=False,
        num_classes=None,
        norm_boundsA=None,
        norm_boundsB=None,
        extract_only=None,
        use_3D=False,
    ):
        """[summary]

        Args:
            mode ([type]): [description]
            imgA_label (str): Identifier for class A. It's the name of the
                folder inside :py:attr:`data_dir` that contains images
                labeled as class A.
            imgB_label (str): Identifier for class B. It's the name of the
                folder inside :py:attr:`data_dir` that contains images
                labeled as class B.
            img_size (int): Dimension of a single image, defined as
                img_size x img_size. Currently, it supports only squared
                images.
            data_dir (str, optional): Path to directory that contains the
                Dataset. This folder **must** contain two subfolders named like
                :py:attr:`imgA_label` and :py:attr:`imgB_label`. Defaults to
                './Data'.
            output_dir ([type], optional): [description]. Defaults to None.
            is_B_categorical (bool, optional): [description]. Defaults to False.
            num_classes ([type], optional): [description]. Defaults to None.
            norm_boundsA ([type], optional): [description]. Defaults to None.
            norm_boundsB ([type], optional): [description]. Defaults to None.
            extract_only (int, optional): Indicate wheter to partially cache a
                certain amount of elements in the dataset. Please remember that
                if :py:attr:`output_dir` folder is already populated, you need
                to clean this folder content to recreate a partial cache file.
                When it is set to None, the entire Dataset is cached. Defaults
                to None.
            use_3D: Indicate whether to use three-dimensional data in the cache
                (if True) or to extract two-dimensional slices from the 3D
                volumes (if False). Defaults to False.

        Raises:
            ValueError: [description]
            FileNotFoundError: [description]
            ValueError: [description]
            FileNotFoundError: [description]
            FileNotFoundError: [description]
            ValueError: [description]
            ValueError: [description]
            ValueError: [description]
            FileNotFoundError: [description]
        """

        if mode not in __dataloader_modality__:
            raise ValueError(f"{mode} modality not recognized. Choose between 'gen' or 'get'")  # noqa

        self.mode = mode

        if mode == "gen":
            dir_mode = False
            file_mode = False

            if os.path.isdir(data_path):
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"{data_path} does not exist")
                else:
                    dir_mode = True
            else:
                if not os.path.exists(data_path):
                    raise FileNotFoundError(f"{data_path} does not exist")
                else:
                    file_mode = True

            self.data_path = data_path

            if output_dir is None:
                if dir_mode:
                    self.output_dir = os.path.join(os.path.dirname(self.data_path),
                                                   f"{os.path.basename(self.data_path)}_TF")
                elif file_mode:
                    base_dirname = os.path.dirname(self.data_path)
                    output_dirname = os.path.basename(
                        self.data_path).replace(".json", "_TF")
                    self.output_dir = os.path.join(
                        base_dirname, output_dirname)
            else:
                self.output_dir = output_dir
            os.makedirs(self.output_dir, exist_ok=True)

            if imgA_label is None or imgB_label is None:
                raise ValueError("imgA_label or imgB_label is None.")

            self.imgA_label = imgA_label
            self.imgB_label = imgB_label

            if dir_mode:
                if not os.path.exists(os.path.join(data_path,
                                                   imgA_label)):
                    raise FileNotFoundError(f"{imgA_label} does not exist")

                if not os.path.exists(os.path.join(data_path,
                                                   imgB_label)):
                    raise FileNotFoundError(f"{imgB_label} does not exist")

                self.imgA_paths, self.imgB_paths = self.get_imgs_paths()
            elif file_mode:
                self.imgA_paths, self.imgB_paths = self.read_imgs_paths()

            if extract_only is not None:
                self.imgA_paths = self.imgA_paths[:extract_only]
                self.imgB_paths = self.imgB_paths[:extract_only]

            if img_size is None:
                raise ValueError("img_size is None")

            self.img_size = img_size

            if (not isinstance(use_3D, bool)):
                raise ValueError("use_3D is not a Boolean value")
            self.use_3D = use_3D

            self.is_3D = self.is_3D_data(self.imgA_paths[0])
            if self.is_3D:
                self.is_A_RGB = self.is_RGB_data(self.imgA_paths[0])
                self.is_B_RGB = self.is_RGB_data(self.imgB_paths[0])
                if self.is_A_RGB or self.is_B_RGB:
                    self.is_3D = False
            else:
                self.is_A_RGB = False
                self.is_B_RGB = False

            if ((not self.is_3D) and (self.use_3D)):
                raise ValueError(
                    "Image files are not 3D but use_3D was set to True")

            self.imgA_type = self.check_type(self.imgA_paths[0])
            self.imgB_type = self.check_type(self.imgB_paths[0])
            self.is_B_categorical = is_B_categorical
            self.num_classes = num_classes
            if norm_boundsA is not None:
                if norm_boundsA[0] >= norm_boundsA[1]:
                    raise ValueError(
                        f"Lower lim for normalization ({norm_boundsA[0]}) must be lower than upper lim ({norm_boundsA[1]})")  # noqa
                self.norm_boundsA = norm_boundsA
            else:
                self.norm_boundsA = None
            if norm_boundsB is not None:
                if norm_boundsB[0] >= norm_boundsB[1]:
                    raise ValueError(
                        f"\rLower lim for normalization ({norm_boundsB[0]}) must be lower than upper lim ({norm_boundsB[1]})")  # noqa
                self.norm_boundsB = norm_boundsB
            else:
                self.norm_boundsB = norm_boundsB

            dataset_property = {"is_3D": self.is_3D,
                                "img_size": self.img_size,
                                "imgA_label": self.imgA_label,
                                "imgB_label": self.imgB_label,
                                "imgA_type": self.imgA_type,
                                "imgB_type": self.imgB_type,
                                "is_A_RGB": self.is_A_RGB,
                                "is_B_RGB": self.is_B_RGB,
                                "is_B_categorical": self.is_B_categorical,
                                "num_classes": self.num_classes,
                                "norm_boundsA": self.norm_boundsA,
                                "norm_boundsB": self.norm_boundsB,
                                "use_3D": self.use_3D
                                }

            output_dir_content = os.listdir(self.output_dir)
            if output_dir_content is not None:
                if "ds_property.json" not in output_dir_content:
                    # folder is not empty, but property file is missing,
                    # we need to write it
                    write_property = True
                elif len(output_dir_content) == 1 and "ds_property.json" in output_dir_content:  # noqa
                    # folder contains only an old version of property file,
                    # we need to overwrite it
                    write_property = True
                else:
                    # every necessary file already exist
                    write_property = False
            else:
                write_property = False

            if write_property:
                with open(os.path.join(self.output_dir,
                                           "ds_property.json"), 'w') as property_file:  # noqa
                    json.dump(dataset_property, property_file, indent=2)

        elif mode == "get":
            if not os.path.exists(data_path) or (not os.path.isdir(data_path)):
                raise FileNotFoundError(f"{data_path} does not exist")
            self.output_dir = data_path

            # dummy variables for images path
            self.imgA_paths = []
            self.imgB_paths = []
            with open(os.path.join(self.output_dir,
                                   "ds_property.json"), 'r') as property_file:
                dataset_property = json.load(property_file)
                self.is_3D = dataset_property["is_3D"]
                self.img_size = dataset_property["img_size"]
                self.imgA_label = dataset_property["imgA_label"]
                self.imgB_label = dataset_property["imgB_label"]
                self.imgA_type = dataset_property["imgA_type"]
                self.imgB_type = dataset_property["imgB_type"]
                self.is_A_RGB = dataset_property["is_A_RGB"]
                self.is_B_RGB = dataset_property["is_B_RGB"]
                self.is_B_categorical = dataset_property["is_B_categorical"]
                self.num_classes = dataset_property["num_classes"]
                self.norm_boundsA = dataset_property["norm_boundsA"]
                self.norm_boundsB = dataset_property["norm_boundsB"]
                self.use_3D = dataset_property["use_3D"]

    def get_dataset(self,
                    batch_size=32,
                    augmentation=False,
                    random_crop_size=None,
                    random_rotate=False,
                    random_flip=False):

        ds = tf.data.Dataset.zip((self.get_imgs(img_paths=self.imgA_paths,
                                                img_label=self.imgA_label,
                                                img_type=self.imgA_type,
                                                is_RGB=self.is_A_RGB,
                                                norm_bounds=self.norm_boundsA),
                                  self.get_imgs(img_paths=self.imgB_paths,
                                                img_label=self.imgB_label,
                                                img_type=self.imgB_type,
                                                is_RGB=self.is_B_RGB,
                                                is_categorical=self.is_B_categorical,  # noqa
                                                num_classes=self.num_classes,
                                                norm_bounds=self.norm_boundsB)
                                  ))

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

    def get_imgs(self,
                 img_paths,
                 img_label,
                 img_type,
                 is_RGB,
                 is_categorical=False,
                 num_classes=None,
                 norm_bounds=None):
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
            img_paths(str): Path to single class images.

        Returns:
            tf.Data.Dataset: Tensorflow dataset object containing images of one
                classes converted in Tensor format, without any other
                computations.
        """
        cache_file = os.path.join(self.output_dir, f"{img_label}.cache")
        index_file = f"{cache_file}.index"

        ds = tf.data.Dataset.from_tensor_slices(img_paths)

        ds = ds.map(lambda path: tf.py_function(self.open_img,
                                                [path],
                                                [__dict_dtype__[img_type]],
                                                ),
                    num_parallel_calls=AUTOTUNE)

        if self.is_3D and (not self.use_3D):
            ds = ds.unbatch()

        if is_RGB:
            ds = ds.map(lambda img: tf.image.rgb_to_grayscale(img),
                        num_parallel_calls=AUTOTUNE)

        # ds = ds.map(lambda img: self.check_dims(img,
        #                                        self.img_size),
        #            num_parallel_calls=AUTOTUNE)
        ds = ds.map(lambda img: self.fix_image_dims(img,
                                                    self.img_size),
                    num_parallel_calls=AUTOTUNE)

        # TODO: add patches

        if is_categorical:
            ds = ds.map(lambda img: tf.one_hot(tf.squeeze(tf.cast(img,
                                                                  img_type)),
                                               depth=int(num_classes)),
                        num_parallel_calls=AUTOTUNE)

        ds = ds.map(lambda img: tf.cast(img, img_type),
                    num_parallel_calls=AUTOTUNE)

        if norm_bounds is not None:
            ds = ds.map(lambda img: self.norm_with_bounds(img,
                                                          norm_bounds),
                        num_parallel_calls=AUTOTUNE)

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
        subset_dir_imgA = os.path.join(self.data_path, self.imgA_label)
        subset_dir_imgB = os.path.join(self.data_path, self.imgB_label)

        filenames_imgA = os.listdir(subset_dir_imgA)
        filenames_imgB = os.listdir(subset_dir_imgB)

        paths_imgA = [os.path.join(subset_dir_imgA, img)
                      for img in filenames_imgA]
        paths_imgB = [os.path.join(subset_dir_imgB, img)
                      for img in filenames_imgB]

        # Sort paths alphabetically
        paths_imgA.sort()
        paths_imgB.sort()

        if len(paths_imgA) != len(paths_imgB):
            raise ValueError(
                f"Dimension mismatch: {len(paths_imgA)} != {len(paths_imgB)}")

        return paths_imgA, paths_imgB

    def read_imgs_paths(self):
        """Read paths of every single image divided by classes from a json file.

        Returns:
            list, list: two list containing the paths of every images for both
                classes. The list is sorted alphabetically, this can be usefull
                when images are named with a progressive number inside a folder
                (e.g.: 001.xxx, 002.xxx, ..., 999.xxx)
        """
        with open(self.data_path, 'r') as f:
            dataset_paths = json.load(f)

        paths_imgA = dataset_paths[self.imgA_label]
        paths_imgB = dataset_paths[self.imgB_label]

        if len(paths_imgA) != len(paths_imgB):
            raise ValueError(
                f"Dimension mismatch: {len(paths_imgA)} != {len(paths_imgB)}")

        return paths_imgA, paths_imgB

    def open_img(self, path):
        """Open an image file and convert it to a tensor.

        Args:
            path(tf.Tensor): Tensor containing the path to the file to be
                opened.

        Returns:
            tf.Tensor: Tensor containing the actual image content.
        """

        path = path.numpy().decode("utf-8")
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        if (self.use_3D):
            image = np.transpose(image, axes=(2, 1, 0))
            # TODO check the correct orientation for axis

        tensor = tf.convert_to_tensor(image)

        return tensor

    def _populate_cache(self, ds, cache_file, num_tot):
        print(f"Caching decoded images in {cache_file}...")
        i = 0
        for _ in ds:
            i += 1
            sys.stdout.write("\r")
            sys.stdout.write(f"{i}/{num_tot}")
            sys.stdout.flush()
        print(f"\nCached decoded images in {cache_file}.")

    @ staticmethod
    def is_3D_data(path):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        if len(image.shape) == 3:
            return True
        elif len(image.shape) == 2:
            return False
        else:
            raise ValueError("Work only with 2D or 3D files.")

    @ staticmethod
    def is_RGB_data(path):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        if image.shape[-1] == 3:
            return True
        else:
            return False

    @ staticmethod
    def check_type(path):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        img_type = image.dtype.name

        return img_type

    def fix_image_dims(self, img, size):
        """Fix tensor dimensions so that they are of the
        proper size to carry out Tensorflow operations.

        This function performs three steps:

        # . `Squeeze <https://www.tensorflow.org/api_docs/python/tf/squeeze>`_ to remove axis with dimension of 1
        # . `Expand <https://www.tensorflow.org/api_docs/python/tf/expand_dims>`_ the dimensions of the tensor by adding one axis
        # . `Resize and pad <https://www.tensorflow.org/api_docs/python/tf/image/resize_with_pad>`_ the tensor to a target width and height

        If `use_3D` was enabled, volume is not resized and padded.

        Args:
            img: image or volume to be processed
            size: desired size of image or volume in the two/three axis.

        """
        # Pad image
        current_size = tf.shape(img)
        diff = (size - current_size) // 2
        pad_amount = tf.where(diff > 0, diff, 0)
        pad_amount = tf.expand_dims(pad_amount, axis=-1)
        paddings = tf.repeat(pad_amount, 2, axis=1)
        img = tf.pad(img, paddings=paddings)

        # Crop image
        current_size = tf.shape(img)
        diff = (size - current_size) // 2
        crop_begins = tf.where(diff < 0, -diff, 0)
        crop_ends = tf.repeat(size, tf.shape(crop_begins))
        img = tf.slice(img, crop_begins, crop_ends)

        img = tf.expand_dims(img, axis=-1)
        return img

    # -------------------------------------------------------------------------
    #  Transformations
    # -------------------------------------------------------------------------

    @ staticmethod
    def norm_with_bounds(image, bounds):
        """Image normalisation. Normalises image in the range defined by lb and
        ub to fit[0, 1] range."""
        lb = tf.cast(bounds[0], dtype=image.dtype)
        ub = tf.cast(bounds[1], dtype=image.dtype)

        image = tf.where(image < lb, lb, image)
        image = tf.where(image > ub, ub, image)

        image = image - lb
        image /= (ub - lb)

        return image

    @ staticmethod
    def random_crop(imgA, imgB, crop_size=256):
        stacked_img = tf.stack([imgA, imgB], axis=0)
        cropped_img = tf.image.random_crop(
            stacked_img, size=[2, crop_size, crop_size, 1]
        )
        cropped_img = tf.split(cropped_img, 2)
        imgA = tf.expand_dims(tf.squeeze(cropped_img[0]), axis=-1)
        imgB = tf.expand_dims(tf.squeeze(cropped_img[1]), axis=-1)
        return imgA, imgB

    @ staticmethod
    def random_flip(imgA, imgB):
        if tf.random.uniform(()) > 0.5:
            imgA = tf.image.flip_left_right(imgA)
            imgB = tf.image.flip_left_right(imgB)
        return imgA, imgB

    @ staticmethod
    def random_rotate(imgA, imgB):
        rn = tf.random.uniform(shape=(), maxval=4, dtype=tf.int32)
        imgA = tf.image.rot90(imgA, k=rn)
        imgB = tf.image.rot90(imgB, k=rn)
        return imgA, imgB


def generate_dataset(data_path,
                     imgA_label,
                     imgB_label,
                     img_size,
                     output_dir=None,
                     extract_only=None,
                     norm_boundsA=None,
                     norm_boundsB=None,
                     is_B_categorical=False,
                     num_classes=None,
                     use_3D=False,
                     ):

    data_loader = DataLoader(mode="gen",
                             data_path=data_path,
                             img_size=img_size,
                             imgA_label=imgA_label,
                             imgB_label=imgB_label,
                             output_dir=output_dir,
                             is_B_categorical=is_B_categorical,
                             num_classes=num_classes,
                             norm_boundsA=norm_boundsA,
                             norm_boundsB=norm_boundsB,
                             extract_only=extract_only,
                             use_3D=use_3D
                             )

    data_loader.get_dataset()

    return


def get_dataset(data_dir,
                percentages,
                batch_size,
                train_augmentation=True,
                random_crop_size=None,
                random_rotate=True,
                random_flip=True
                ):

    if len(percentages) != 3:
        raise ValueError("Percentages has to be a list of 3 elements")

    if round((percentages[0] + percentages[1] + percentages[2]), 1) != 1.0:
        raise ValueError("Sum of percentages has to be 1")

    data_loader = DataLoader(mode="get",
                             data_path=data_dir,
                             )

    complete_ds = data_loader.get_dataset(batch_size=batch_size,
                                          augmentation=train_augmentation,
                                          random_crop_size=random_crop_size,
                                          random_rotate=random_rotate,
                                          random_flip=random_flip)

    complete_ds = complete_ds.unbatch()

    # Compute length of dataset
    num_imgs = 0
    for _ in complete_ds:
        num_imgs += 1

    train_ends = int(num_imgs * percentages[0])

    valid_begins = train_ends
    valid_ends = valid_begins + int(num_imgs * percentages[1])

    # Train Datasets
    train_ds = complete_ds.take(train_ends)
    train_ds = train_ds.batch(batch_size)

    # Same as before, but without augmentation since now we want to obtain
    # validation and test set
    complete_ds = data_loader.get_dataset(batch_size=batch_size,
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

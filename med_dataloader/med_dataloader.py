"""Main module."""

import os
import sys
import SimpleITK as sitk
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE


class DataLoader:
    def __init__(self,
                 imgA_label,
                 imgB_label,
                 data_dir='./Data',
                 imgs_subdir='Images',
                 cache_mode="prod",
                 extract_only=None,
                 ):

        if not os.path.exists(data_dir):
            raise ValueError(f"{data_dir} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir)):
            raise ValueError(f"{imgs_subdir} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir, imgA_label)):
            raise ValueError(f"{imgA_label} does not exist")

        if not os.path.exists(os.path.join(data_dir, imgs_subdir, imgB_label)):
            raise ValueError(f"{imgB_label} does not exist")

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
                    batch_size=32,
                    augmentation=False,
                    crop_size=128):

        ds = tf.data.Dataset.zip((self.get_imgs(img_paths=self.imgA_paths,
                                                img_label=self.imgA_label),
                                  self.get_imgs(img_paths=self.imgB_paths,
                                                img_label=self.imgB_label)
                                  ))

        ds = ds.map(self.check_dims, num_parallel_calls=AUTOTUNE)

        if augmentation:
            # TODO selectively choose which type of data augmentation to apply
            # ds = ds.map(lambda imgA, imgB: self.random_crop(imgA,
            #                                                imgB,
            #                                                crop_size),
            #            num_parallel_calls=AUTOTUNE)
            ds = ds.map(lambda imgA, imgB: self.resize(imgA,
                                                       imgB,
                                                       crop_size),
                        num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.random_rotate, num_parallel_calls=AUTOTUNE)
            ds = ds.map(self.random_flip, num_parallel_calls=AUTOTUNE)

        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)

        return ds

    def get_imgs(self, img_paths, img_label):
        cache_file = os.path.join(self.cache_dir, f'{img_label}.cache')
        index_file = f'{cache_file}.index'

        ds = tf.data.Dataset.from_tensor_slices(img_paths)
        ds = ds.map(lambda path: tf.py_function(self.parse_img,
                                                [path],
                                                [tf.float32]),
                    num_parallel_calls=AUTOTUNE)
        ds = ds.cache(cache_file)

        if not os.path.exists(index_file):
            self._populate_cache(ds, cache_file, len(img_paths))

        return ds

    def get_imgs_paths(self):
        print("Fetching images paths...")
        subset_dir = os.path.join(self.data_dir, self.imgs_subdir)

        subset_dir_imgA = os.path.join(subset_dir, self.imgA_label)
        subset_dir_imgB = os.path.join(subset_dir, self.imgB_label)

        filenames_imgA = os.listdir(subset_dir_imgA)
        filenames_imgB = os.listdir(subset_dir_imgB)

        paths_imgA = [os.path.join(subset_dir_imgA, img)
                      for img in filenames_imgA]
        paths_imgB = [os.path.join(subset_dir_imgB, img)
                      for img in filenames_imgB]

        print("Images paths collected.")

        # Sort paths alphabetically
        paths_imgA.sort()
        paths_imgB.sort()

        return paths_imgA, paths_imgB

    def parse_img(self, path):
        path = path.numpy().decode("utf-8")
        return self.open_file(path)

    def open_file(self, path):
        image = sitk.GetArrayFromImage(sitk.ReadImage(path))

        # Scale image in 0-1 with a predefined range
        # TODO: remove hardcoding in this function
        image = self.normalise_with_boundaries(image, lb=-1024, ub=3200)
        tensor = tf.convert_to_tensor(image)
        return tf.expand_dims(tensor, axis=-1)

    def _populate_cache(self, ds, cache_file, num_tot):
        print(f'Caching decoded images in {cache_file}...')
        i = 0
        for _ in ds:
            i += 1
            sys.stdout.write('\r')
            sys.stdout.write(f"{i}/{num_tot}")
            sys.stdout.flush()
        print(f'\nCached decoded images in {cache_file}.')

    # -----------------------------------------------------------
    #  Transformations
    # -----------------------------------------------------------
    @staticmethod
    def check_dims(imgA, imgB):
        # TODO: remove hardcoding
        imgA = tf.expand_dims(tf.squeeze(imgA), axis=-1)
        imgA = tf.image.resize_with_pad(imgA, 256, 256)

        imgB = tf.expand_dims(tf.squeeze(imgB), axis=-1)
        imgB = tf.image.resize_with_pad(imgB, 256, 256)

        return imgA, imgB

    @staticmethod
    def normalise_with_boundaries(image, lb, ub):
        """Image normalisation. Normalises image in the range defined by lb and
        ub to fit [0, 1] range."""
        epsilon = 1e-8

        image[image < lb] = lb
        image[image > ub] = ub

        # image = image.astype(np.float32)
        ret = image - lb
        ret /= ((ub - lb) + epsilon)

        return ret

    @staticmethod
    def resize(imgA, imgB, size=256):
        imgA = tf.image.resize(imgA, (size, size))
        imgB = tf.image.resize(imgB, (size, size))

        return imgA, imgB

    @staticmethod
    def random_crop(imgA, imgB, crop_size=256):
        stacked_img = tf.stack([imgA, imgB], axis=0)
        cropped_img = tf.image.random_crop(
            stacked_img, size=[2, crop_size, crop_size, 1])
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
                     percentages,
                     batch_size,
                     input_size
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
                                          augmentation=True,
                                          crop_size=input_size
                                          )

    complete_ds = complete_ds.unbatch()

    # Train Datasets
    train_ds = complete_ds.take(train_ends)
    train_ds = train_ds.batch(batch_size)

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

=======
History
=======

0.1.15 (2022-05-17)
-------------------

* img_size support also None value.

0.1.14 (2022-05-16)
-------------------

* Improved handling of img_size parameters for both 2D and 3D images.
  
  * User can now declare img_size as a 2- or 3-elements list.
  
  * Automatic zero-padding or center-based cropping is performed to adapt the size of the image to the declare img_size parameter.

* New features: 3D volumes can be patched into smaller cubic patches. Overlapping between patches is also supported.

0.1.13 (2022-03-10)
-------------------

* New function: now dataset can be generated also by reading a json file containing list of file paths.

0.1.12 (2021-10-29)
-------------------

* Fixed minor bug in function norm_with_bounds

0.1.11 (2021-09-10)
-------------------

* Fixed support for 3D Images
* Fixed minor bugs

0.1.10 (2021-05-11)
-------------------

* Added support one-hot encoding in case of multi-class label

0.1.9 (2021-05-11)
------------------

* Added support for RGB Images
* Fixed some bugs related to norm_bounds types

0.1.8 (2021-05-09)
------------------

* Main Changes in the package structure. Now there are two main functions: 
  generate_dataset and get_dataset, both leveraging on DataLoader class.
* The generation of the dataset can be handled also by CLI, to simplify usage.
* Processed data can live by themself. No more need to transfer also original
  file (e.g. to Drive to make use of them on Colab)

0.1.6 (2021-05-06)
------------------

* Improved flexibility for image data types. Now cache dimension reflects the
  actual dataset dimension.

0.1.5 (2021-04-30)
------------------

* Added support for 3D files: now Dataloader automatically detects whether a
  file is 2D or 3D and returns the properly sized dataset. Please remember that
  med_dataloader returns tf.data.Dataset object for 2D tasks, 3D is not yet
  supported.
* Added new notebook in examples folder.

0.1.4 (2021-04-29)
------------------

* Improved code flexibility:
    * It is possibile to choose which type of data augmentation is performed
    * Boundaries for data normalization can be set by the user
    * Images can be resized automatically by the user
* Added basic_usage example also as a notebook

0.1.1 (2021-04-20)
------------------

* Added code for package
* Basic example of usage inside folder "examples"
* Partial documentation

0.1.0 (2021-04-16)
------------------

* First release on PyPI.

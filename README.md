# CAM-tensorflow
This is an Tensorflow implementation of ["Learning Deep Features for Discriminative Localization"](https://arxiv.org/pdf/1512.04150.pdf).

The study proposed a method to enable the convolutional neural network to have localization ability despite being trained on image-level labels.
It was presented in Conference on Computer Vision and Pattern Recognition (CVPR) 2016 by B. Zhou, A. Khosla, A. Lapedriza, A. Oliva, and A. Torralba from MIT.

(This repository is on working...)

- See the summary of the paper in [this blog](https://kobiso.github.io//research/research-learning-deep-features/)

- **The framework of the Class Activation Mapping**

![Example](/images/cam.jpg)

## Prerequisites
- Python 3.4+
- TensorFlow 1.5+
- Jupyter Notebook
- Python packages: requirements.txt
  - Simply install it by running : `pip install -r /path/to/requirements.txt` in the shell

## Prepare Data set
1. Download the [Tiny ImageNet](https://tiny-imagenet.herokuapp.com/) in this link and unzip it.
2. Set the path of the dataset on variable `TINY_IMAGENET_DIRECTORY` in `build_tfrecords.ipynb` file.
3. To convert Tiny ImageNet to TFRecords, set each requiring path in `build_tfrecords.ipynb` and run all cell.
    - As test set does not include class labels and bounding boxes, validation set will be used as test set in this implementation.
    - And training set will be divided with certain percentage (as you defined) into training set and validation set.
    - Each data set (training, validation and test) will have iamges, labels and bounding box information.
    - Note that you can set the validation ratio in the variable `VALIDATION_RATIO`.
4. You can check and visualize TFRecords file with `check_tfrecords.ipynb`.
    - After reading the TFRecords, the data will be saved in `read_data_dict` dictionary.

## Train the Model
1. Download the [pretrained inception_v3 checkpoint](https://github.com/tensorflow/models/tree/master/research/slim).
2. Set the necessary path and parameters and train the network in `train_and_test.ipynb` file.
    - As we use pretrained model, we first train added layer and then train the entire model for training efficiency.

## Test the Model
  
## Reference
- [Learning Deep Features for Discriminative Localization](https://arxiv.org/pdf/1512.04150.pdf)
- [Official implementation of the paper](https://github.com/metalbubble/CAM)
  
## Author
Byung Soo Ko / kobiso62@gmail.com

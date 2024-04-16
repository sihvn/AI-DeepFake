# AI-DeepFake

This project aims to detect videos that have been manipulated using methods such as Deepfakes, Face2Face, FaceSwap and NeuralTextures. We explored various Convolutional Neural Network (CNN) models such as Residual Network (ResNet) and EfficientNet.

# Dataset

The project uses dataset from [FaceForensics++](https://github.com/ondyari/FaceForensics).

Download the dataset by filling in this [form](https://docs.google.com/forms/d/e/1FAIpQLSdRRR3L5zAv6tQ_CKxmK4W96tAab_pfBu2EKAgQbeDVhmXagg/viewform) and save it under the folder **dataset**.

# To setup

Use Python 3.10 (developed using 3.10.11).\
TensorFlow 2.1 used in this project requires CUDA 10.1.

Set up Python 3.10 virtual environment:

```
> py -3.10 -m pip install virtualenv \
> py -3.10 -m venv \<name-of-virtualenv>
```

Activate Python virtual environment:

```
> \<path-to-virtualenv>\Scripts\activate
```

Install poetry:

```
> pip install poetry
```

Install the dependencies:

```
> poetry install
```

Add dependencies:

```
> poetry add \<name-of-package>
```

Add dependency of specific version:

```
> poetry add \<name-of-package>@\<version>
```

Write/resolve poetry lock file (updates dependencies list by default):

```
> poetry lock [--no-update]
```


Note: due to error with package handling in poetry, in order to install tensorflow, python version 10 must be used, and tensorflow-io-gcs-filesystem version 0.27 must first be added before adding tensorflow version 2.10.

Tensorflow compatibility: https://pypi.org/project/tensorflow-io-gcs-filesystem/



# Run the Project

Run **main.py** to start the project

```
py main.py <train_test_predict> <dataset_root_dir> <require_pre_process> <weights_path>[ <initial_weights_path>][ <seed>]

```
### valid inputs for train_test_predict argument

valid_train_inputs = ["", "train", "Train"]

valid_test_inputs = ["test", "Test"]

valid_predict_inputs = ["predict", "Predict"]

### valid inputs for require_pre_process argument

valid_is_require_pre_process_inputs = ["","y","Y","yes","Yes","t","T","true","True",]

valid_not_require_pre_process_inputs = ["n","N","no","No","f","F","false","False",]

# Project Report

[Report](https://github.com/sihvn/AI-DeepFake/blob/main/Project%20Report.pdf)

# Project Slide

[Slide](https://github.com/sihvn/AI-DeepFake/blob/main/Project%20Slide.pdf)

# GUI

To run GUI
```
py GUI.py
```

To start Tensorboard

```
cd notebooks
tensorboard --logdir=runs
```

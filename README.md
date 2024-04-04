# AI-DeepFake

Use Python 3.10 (developed using 3.10.11).\
TensorFlow 2.1 used in this project requires CUDA 10.1.

Set up Python 3.10 virtual environment:

> py -3.10 -m pip install virtualenv \
> py -3.10 -m venv \<name-of-virtualenv>

Activate Python virtual environment:

> \<path-to-virtualenv>\Scripts\activate

Install poetry:

> pip install poetry

Install the dependencies:

> poetry install

Add dependencies:

> poetry add \<name-of-package>

Add dependency of specific version:

> poetry add \<name-of-package>@\<version>

Write/resolve poetry lock file:

> poetry lock [--no-update]

\
Note: due to error with package handling in poetry, in order to install tensorflow, python version 10 must be used, and tensorflow-io-gcs-filesystem version 0.27 must first be added before adding tensorflow version 2.10.

Tensorflow compatibility: https://pypi.org/project/tensorflow-io-gcs-filesystem/

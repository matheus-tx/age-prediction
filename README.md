# Age Predictor
Welcome to Age Predictor. This project features a deep learning model that predicts someone's age based on a photo of their face. 

Models used here were trained on UTKFace dataset. This dataset contains images of more than 20.000 people ranging from 0 to 116 years old and are labelled by age, gender and ethnicity. Only "Aligned&CroppedFaces" were used here, instead of "In-the-wild" images. More information on UTKFace, as well as its data can be found [here](https://susanqq.github.io/UTKFace/). In this project, we primarily focus on age labels, but nothing prevents us from making models that also predict gender or ethniticy in the future.

The programming language used here is Python 3.10 and the deep learning framework is [Tensorflow](https://github.com/tensorflow/tensorflow), together with [Keras](https://github.com/keras-team/keras). Some models are currently being tested with [Tensorflow Probability](https://github.com/tensorflow/probability), a Tensorflow module that handles uncertainty modelling. This is necessary for interval estimation. All models are currently built upon [Keras VGGFace](https://github.com/rcmalli/keras-vggface), but that may change in the future.

Tensorflow and Python were chosen primarily because they are the most used language-deep learning framework duet. Secondly, because Keras VGGFace package was built upon, well, Keras. The fact that Tensorflow can model uncertainty through Tensorflow Probability module also had some weight at this decision.

This project uses [Poetry](https://python-poetry.org/) for packaging and dependency management. It takes care of all dependency versioning so you do not have to bother with conflicting versions.

The Python Keras VGGFace package was tested on Tensorflow 1.14.0. There is a problem with it, though: the Tensorflow Probability version used in this project is currently XX.XX, which relies on Tensorflow XX.XX. So there were some dependency problems, mostly concerning imports.
Thus some hacks had to be done to make things work properly.

There are some features and changes we hope to implement in the future. Here are some of them:
* Of course, continuously improving models.
* Adding algorithms that model uncertainty and are able to predict age intervals. This makes sense, since you cannot usually tell someone's exact age. Instead, we often use intervals, like "this person must be between 27 and 34 years old". So why not to let an AI do the same?
* Gender and ethnicity prediction could also be thought of once we have got models that perform relatively close to state of the art papers.

## How to install
The following steps should work for Ubuntu 20.04 LTS, Ubuntu 22.04 LTS and Linux distros based on one of these, like Linux Mint 20.3. They might also work for other Debian-based Linux distributions, but they were not tested. Steps should be similar for installing it on Windows or Mac, but will not be covered in this document.

Here are the steps you must follow:
1. **Choose a directory to install this project**. Here, we will call it `~/path/to/project`.
2. **Go to project path** by running `cd ~/path/to/project` in bash and then clone this repository by running the following code:
```bash
git clone https://github.com/matheus-tx/age-predictor
```
3. **Go to this project's directory** on your machine by running `cd age-predictor`.
4. **Make sure you have Python 3.10 installed on your machine**. This project was only tested with this version. It might work with other Python versions, specially with more recent ones, but there is no guarantee. So you are highly advised to have it installed on your machine. In case you do not have it installed, you can find a tutorial on how to install it on Ubuntu 20.04 and 22.04 on this [link](https://computingforgeeks.com/how-to-install-python-on-ubuntu-linux-system/).
5. **Make sure you have Poetry installed**. If you do not, run the following commands on bash:
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
curl -sSL https://install.python-poetry.org | python3 - -y
```
6. **Install the project**. Note that this step may take quite some time depending on your machine or your internet. This happens because it installs all the needed packages. These include some heavy ones, like Tensorflow.
```bash
make install
```
7. **Alternatively, you can do installation manually**. This involves some steps:
* configuring Poetry environment to be local. This is useful for the hacks we will make to get Keras VGGFace work with the required version of Tensorflow. To do this, run
```bash
poetry config --local virtualenvs.in-project true
```
* initializing poetry virtual environment. You can do this by running
```bash
poetry init
```
* installing packages. This is the long part. Run the following code and wait some minutes until it finishes:
```bash
poetry install
```
8. **Change Keras VGGFace dependencies**. You have to manually change a line in Keras VGGFace source code. Open file `.venv/lib/keras_vggface/models.py`. Then replace line 20 by
```python
from keras.src.utils.layer_utils import get_source_inputs
```

[//] # (TODO: Add step to test if everything works properly)

9. **Be happy**. If you have gotten to this point without any errors, then this project is installed on your machine and ready to run.

### Optional step: permanently set Python keyring backend to `null`
There is one note regarding Poetry. At least on machines I have tested, every time you install it or install a new package using it, you first have to run
```bash
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
If you do not, you get an error: `Failed to unlock the collection!`. To avoid this inconvenience, you may want to permanently set Python keyring backend to `null`. I prefer (and find more secure) to set it to `null` only when needed, but if you wish to do so for whatever reason (and understand the risks), do the following:
1. **Open the the Bash shell configuration file**. You can do it in several different ways, but we will use `nano` here. Run
```bash
nano ~/.bashrc
```
2. **Add the new environment variable**. Go to the end of the file and add the following to it:
```
# Set Python keyring backend to null
export PYTHON_KEYRING_BACKEND=keyring.backends.null.Keyring
```
3. **Save the new configuration**. Just press `Ctrl` + `O` and then `Ctrl` + `X`.
4. **Load the new configuration**. Run
```bash
source ~/.bashrc
```

## How to use

This project's main goal is to let you to be able to predict someone's age. There are two ways you can do this: by running a prediction script directly or using the Python API.

### Running on main script directly
You can use prediction script by running
```bash
python3.10 src
```

The main script does several things besides of making predictions. 

### Using API
The recommended (and most elegant) way of making predictions is to use Python API. It is still under development.
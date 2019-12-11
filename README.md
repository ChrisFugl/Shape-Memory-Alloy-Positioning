# Shape Memory Alloy Positioning
Purpose of this work is to train a controller that stabilises a shape memory alloy in a real environment.

## Requirements
Please make sure that you have the following requirements installed on your system:

* Python (>= 3.6)

## Installation
**Important:** These installation steps are only tested on a Unix system. Please add necessary additions for Windows users.

First clone the project.

``` sh
git clone https://github.com/ChrisFugl/Shape-Memory-Alloy-Positioning
cd Shape-Memory-Alloy-Positioning
```

**Note:** We recommend that you install the Python packages in a virtual environment. See the next section for how to do this, and then proceed with the rest of this section afterwards.

``` sh
pip install -r requirements.txt
```

### Virtual Environment (optional)
A virtual environment helps you to avoid that Python packages in this project does not conflict with other Python packages in your system. Follow the instructions [on this site](https://virtualenv.pypa.io/en/stable/installation/) to install the virtualenv package, which enables you to create virtual environments.

Once virtualenv is installed, you will need to run the following commands to setup a virtual environment for this project.

``` sh
virtualenv env
```

You may want to add the flag "--python python3" in case your default Python interpreter is not at version 3 (run ```python --version``` to check the Python version):

``` sh
virtualenv --python python3 env
```

Either of the previous two commands will create a directory called *env* in the project directory. You need to run the following command to make use of the virtual environment.

``` sh
source env/bin/activate
```

You are now up an running with the virtual environment. Run the following command when you want to exit this environment.

``` sh
deactivate
```

## Usage
Two main scripts can be used to train and test the model. Use the configuration files in *config/* to specify how to configure the model, environment, and more. For example:

``` sh
python train.py configs/debug.yaml
```

Note that the training can continue infinitely (or until manually stopped) by setting iterations to a negative number.

## Acknowledgement
The implementation of Soft Actor-Critic is based of [the implementation by Vitchyr Pong](https://github.com/vitchyr/rlkit).

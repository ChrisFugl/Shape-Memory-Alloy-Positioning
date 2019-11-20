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

## Tests
All tests are located in the *tests* directory. Use the following command to run all tests:

``` sh
python -m unittest discover -s tests
```

## Structure
The code is structured as follows:

**train.py**: A script that should be called from the command line to train the model. It should accept some arguments (TBD). For example the following command could be used to run a simulation:

``` sh
python train.py simulated
```

The following command could be used to train on real time date.

``` sh
python train.py real_time
```

**test.py**: A script that evaluates how well a pretrained model performs. Similarly to *train.py*, it should be possible to configure this script using command line arguments.

**app/**: All code related to the model and environments.

**app/model.py**: The Soft Actor-Critic model should be implemented in this file.

**app/network.py**: The neural network used by the model should be implemented here.

**app/replay_buffer.py**: Store used to store samples.

**app/environments/**: The purpose of this module is to provide an interface that *train.py* and *test.py* can use to interact with the environment. In other words: The environments should act as adapters, such that the training and test script does not have to bother with whether the environment is simulated or a real time system.

**app/environment/real_time**: An environment for a real time system.

**app/environment/simulated.py**: An environment for a simulated system. The simulation can be fairly simple. The purpose of it is to debug and test the implementation of the model without having to access the real time system.

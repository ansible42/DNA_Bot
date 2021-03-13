# January 13 2021 – Getting a working GPT2 model running on Raspberry Pi 4 with Python
# Details of my setup: Raspberry Pi OS on Raspberry Pi 4 (4GB RAM) + 128GB Samsung EVO+ MicroSD card
# This is under the assumption you are NOT SSH/remote into Raspberry Pi OS

# 1) Open terminal window on Raspberry Pi OS

# 2) You may want to update Python and IDLE: 
sudo apt update
# (As of today I have Python 3.7.3)

sudo apt install python3 idle3
# (Updating IDLE is optional since everything is happening inside terminal)

# 3) Install/update pip:
sudo apt install python3-pip

# 4) Install/update virtualenv:
sudo apt install virtualenv python3-virtualenv –y

# 5) Create a virtual environment (env) called 'testpip' (you can name it whatever):
virtualenv -p /usr/bin/python3 testpip
source testpip/bin/activate

# 6) Your prompt should go from "pi@raspberrypi" --> "(testpip) pi@raspberrypi"
# (Indicates you are inside the virtualenv 'testpip' that you created)

# 7) Inside the virtualenv 'testpip' we install the various packages and libraries to use within the 'testpip' virtualenv (only)

# 8) For instance if you want scipy (which will also install numpy):
pip install scipy

# 9) The following commands install Tensorflow and various other dependencies to make it run:
sudo apt-get install -y libhdf5-dev libc-ares-dev libeigen3-dev
python3 -m pip install keras_applications==1.0.8 --no-deps
python3 -m pip install keras_preprocessing==1.1.0 --no-deps
python3 -m pip install h5py==2.9.0
sudo apt-get install -y openmpi-bin libopenmpi-dev
sudo apt-get install -y libatlas-base-dev
python3 -m pip install -U six wheel mock

# 10) These steps are specific to getting Tensorflow 1.13.1
# (which works with gpt2client, unsure if any slightly newer versions of Tensorflow will work.
# The gpt2client package states that 2.0 and newer don’t.)

# a) Go to this website: https://www.piwheels.org/project/tensorflow/#install
# b) Scroll down to find the 1.13.1 version of Tensorflow and click the blue box
# c) Copy the link for the "tensorflow-1.13.1-cp37-none-linux_armv7l.whl" file

# d) In terminal type: 
wget https://www.piwheels.org/simple/tensorflow/tensorflow-1.13.1-cp37-none-linux_armv7l.whl#sha256=25f4ff027beec1e568baf8e90a07bad59d354560533d6b37318b9efeb70beeb1

# e) Uninstall previous versions of Tensorflow (if you have any):
python3 -m pip uninstall tensorflow

# f) Since we are in a virtual environment, type:
pip install tensorflow*
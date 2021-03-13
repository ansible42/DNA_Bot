# Running GPT2-Client on Raspbery Pi 4
Not as straitforward as one might think 

##  TensorFlow dependency 

Some directions are included [here](https://www.tensorflow.org/lite/guide/build_rpi) I chose to compile natively on the Raspberry Pi 4, running Ubuntu 20.

```bash
sudo apt-get install build-essential
sudo apt-get install zip unzip
sudo apt-get install g++-arm-linux-gnueabihf
git clone https://github.com/tensorflow/tensorflow.git tensorflow_src
cd tensorflow_src
./tensorflow/lite/tools/make/download_dependencies.sh
./tensorflow/lite/tools/make/build_rpi_lib.sh
## Buiding the pip package 
sudo apt install swig libjpeg-dev zlib1g-dev python3-dev python3-numpy
pip3 install numpy pybind11
tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/lite/tools/pip_package/build_pip_package.sh
## At this point watch for smoke from the pi and go get some coffee, or beer, or a coffee than a beer 

``` 
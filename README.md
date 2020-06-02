# Wally Finder

A simple solver for ["Where's Wally"](https://en.wikipedia.org/wiki/Where%27s_Wally%3F) images.

## Introduction

This repository contains a deep convolutional neural network trained to find Wally (aka Waldo).

The training code is based upon the [dlib](http://dlib.net) [example for training a car detector](http://dlib.net/dnn_mmod_train_find_cars_ex.cpp.html).
The only thing I changed was the network architecture to be able to detect objects as small as 24x24 pixels.

The definition of the network architecture can be found in [`src/detector.h`](src/detector.h).
The trained model has 7 convolutional layers and is only 350 kB, so I decided to embed it into the code directly.
To that end, I used the powerful [`serialize`](http://dlib.net/other.html#serialize) function family from [dlib](http://dlib.net).
The model has been serialized into a bytestring, then compressed and finally converted to base 64.

As a result, instantiating the `WallyFinder` class from python code is enough to get a fully working model that, when run on an image, will return a list of dictonaries with `xmin`, `ymin`, `xmax`, `ymax` that describe each bounding box.

## Dependencies

These dependencies are only needed at build time, not at run-time:
- `CMake` `>=3.14`
- `Ninja`
- `gcc` `>=7`
- `g++` `>=7`

These dependencies are needed at both build-time and run-time:
- `python` `>=3.6`
- `CUDA` `>=7.5` (optionaL: it will use CPU instead if not found)
- `CUDNN` `>=5` (optional: it will use CPU instead if not found)

## Building

A `build.sh` script is provided to simplify the build process, just run:

``` bash
./build.sh
```

## Building and installing the python module

### Creating a wheel file

An installable wheel file can be created by running

```
python setup.py bdist_wheel
```

The compiled .whl file will be placed in `dist` directory, which can be now distributed or installed using `pip install`

### Installing the python module

The steps described above can be combined by running

```bash
python setup.py install --user
```

This will install the module to `~/.local/lib/`.

You can also opt for a system-wide installation by removing the `--user` flag.

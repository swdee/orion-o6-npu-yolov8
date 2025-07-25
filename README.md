
# Orion O6 NPU YOLOv8 C++ Example

## Overview

This is a C++ example showing how to use the Orion O6 NPU with a YOLOv8 model.

It shows you step by step the process of taking the source Ultralytics YOLOv8 pytorch model through to running inference.

This example uses Debian 12 (Bookworm) with KDE desktop environment installed for the x86 workstation,  
however other operating systems can be setup by installing equivalent packages.

## OS Tools

The following packages are required to be installed on your Debian OS.
```
sudo apt install build-essential libtinfo5 libncurses5
```

Install Conda for managing python versions and environments.
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

After following the install instructions, close and re-open your shell.


## Python Environment

We require two python environments to run the various projects.
1. Python 3.11 for Ultralytics and misc tools
2. Python 3.8 for Cix Builder

Create these environments using conda.
```
conda create -n ultra python=3.11
conda create -n cixbuild python=3.8
```


### Ultralytics


Activate the virtual environment for Ultralytics.
```
conda activate ultra
```

Install Ultralytics and onnx packages
```
pip install ultralytics onnx onnxslim
```


### CIX Builder

Activate the virtual environment for CIX Builder
```
conda activate cixbuild
```

You need to register with CIX to download their NOE SDK release (25 Q1).   Within their downloadable archive
install CIX compiler wheel file.
```
pip install CixBuilder-6.1.3119.2-py3-none-linux_x86_64.whl
```



## Preparation

Check out this git repository to your workstation.  This must be an x86 workstation to compile the 
needed CIX model file for use on the Orion O6.
```
git clone https://github.com/swdee/orion-o6-npu-yolov8.git
```

Enter the project `build` directory.
```
cd orion-o6-npu-yolov8/build
```


## Pytorch Model

Using the Ultralytics YOLOv8 [branch here](https://github.com/ultralytics/ultralytics/tree/v8.1.43) download the
Detection (COCO) model file of the desired size.  In this example we will use the `YOLOv8s` (small) variant.  Download
this file into the `build` directory.
```
wget https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt
```

You can train this model with your own data if needed, but for this example we will stick with the 
default trained COCO-80 dataset.



## Convert to Onnx

Activate the Ultralytics python environment.
```
conda activate ultra
```

Convert the pytorch `yolov8s.pt` model to ONNX.
```
yolo export model=yolov8s.pt format=onnx imgsz="(640,640)" opset=12 dynamic=False simplify=True
```

This command will save the onnx file to `yolov8s.onnx`.


## Quantization

### Magic Numbers

Each YOLO model size variant `n,s,m,l,x` is of a different size so we need to find the magic numbers required for
quantization and compiling by the cixbuilder.  The Cix AI model Zoo provides the magic numbers in the 
`cfg/yolov8_lbuild.cfg` file for the `l` sized model;
```
trigger_float_op = disable & <[(258, 272)]:float16_preferred!>
weight_bits = 8& <[(273,274)]:16>
activation_bits = 8& <[(273,274)]:16>
bias_bits = 32& <[(273,274)]:48>
```

These magic numbers represent the end nodes/layers of the model graph.  They are explicitly excluded from being 
quantized down to int8, but are instead retained at a higher precision (16 bit).  Each line from the config file
has the following meaning.

```
trigger_float_op = disable & <[(258, 272)]:float16_preferred!>
```
 * Default: Float‐triggering is disabled globally.
 * Override: For operators 258 through 272 (inclusive), force the `float16_preferred` mode (the trailing ! makes it a hard override).

```
weight_bits = 8 & <[(273,274)]:16>
```
 * Default: Quantize all layer weights to 8 bits.
 * Override: For operators 273 and 274, quantize weights to 16 bits instead.

```
activation_bits = 8 & <[(273,274)]:16>
```
 * Same pattern as above but for activation tensors: default 8 bits, override to 16 bits on ops 273–274.

```
bias_bits = 32 & <[(273,274)]:48>
```
 * Default: Store all bias terms in 32 bits.
 * Override: For ops 273–274, use 48 bits for bias.


### Finding 

To find the magic numbers for the `yolov8s.onnx` file see [this document](doc/MAGIC.md).



## Calibration Dataset

A post training quantization process requires a calibration dataset so the compiler can run sample images
through the model to gather activation statistics for every layer, these are then used to compute the
INT8 scale and zero-point parameters.

The default Ultralytics models are trained on the [COCO 2017 dataset](https://cocodataset.org/), download 
the Val2017 images (1GB) into the `build` directory.

```
wget http://images.cocodataset.org/zips/val2017.zip 
```

Unpack the zip file which contains 5000 images.   
```
unzip val2017.zip
```

We don't need all these images so will randomly delete all but 200 of them.   A subset sufficient in size for the
quantization process.
```
cd val2017

ls *.jpg \
  | shuf \
  | tail -n +201 \
  | xargs -d '\n' rm --
```

Delete the zip file
```
cd ..
rm -f val2017.zip
```

Confirm you are still in the Ultralytics python environment, if not then run.
```
conda activate ultra
```

Use the `tools/make_calibration.py` script to generate the calibration dataset used by the CIX compiler.
```
cd ../tools
python make_calibration.py ../build/val2017/ ../build/calibration_data.npy
```



## Compile Model

### CIX Config

Next we need to compile the ONNX file into a CIX model to run on the NPU.

The config file `cfg/yolov8_s_build.cfg` exists containing the quantization magic numbers
discovered above and sets paths corresponding to the layout of this project.


### Compile 

Change to the cixbuild python environment.
```
conda activate cixbuild
```

Add cixbuild libraries to LD library path.
```
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.8/site-packages/AIPUBuilder/simulator-lib/:$LD_LIBRARY_PATH
```

Compile the model.
```
cd ../build
cixbuild ../cfg/yolov8_s_build.cfg
```

This will result in the compiled model `yolov8s.cix` in the `build` directory.   In the `../compile/` directory
will be the compile processed output files.



## Orion O6

We will now setup the Orion O6 to run inference, complete all of the following steps on your O6.

### Preparation

Clone this project onto your O6.
```
git clone https://github.com/swdee/orion-o6-npu-yolov8.git
```

Copy the `yolov8s.cix` model to your O6 into the project `orion-o6-npu-yolov8/build` directory.


### Compile C++ Example

Make the C++ example.
```
cd orion-o6-npu-yolov8
make
```

This will compile the `yolov8` program in the project root.

### Run Inference

Run inference using the provide bus.jpg image and `yolov8s.cix` model with default COCO parameters.
```
./yolov8 build/yolov8s.cix bus.jpg 0.25 0.45
```

This will result in the following output:
```
NOE context initialized
Model/Graph loaded
Created Job: 4294967297
Tensor Counts, Input=1, Output=1
Input tensor descriptor:
  id:          0
  size:        1228800
  scale:       255.078
  zero_point:  0
  data_type:   U8
Output tensor descriptor:
  id:          0
  size:        1411200
  scale:       1
  zero_point:  0
  data_type:   F16
Tensor load time: 1.63655 ms
Inference sync time: 14.3992 ms
Fetch outputs time: 4.67951 ms
person 0.852 (106,224,216,558)
person 0.852 (488,188,558,552)
person 0.852 (206,228,280,488)
person 0.500 (84,304,125,553)
bus 0.817 (150,114,620,502)
```

Note that this example only outputs the detected objects to stdout.  If you want to annotate the original image
by drawing the bounding boxes as an overlay, that is an exercise for your own requirements.



## Developer Commentary

The Orion O6 is still very much a platform in development and the NPU SDK and NOE NPU driver are not yet in 
an ideal state.

* The NOE driver [needs to be fixed](https://forum.radxa.com/t/cixbuilder-problems-compiling-onnx-model-slow-inference-times/26972/11) 
  so DMA buffer's can be used to speed up the Tensor input and output handling.
* The NOE driver shared library `libnoe.so` does not match the shipped `npu/cix_noe_standard_api.h` header.
* The current SDK User Guide (v0.6) needs to be fixed up as the example C++ driver code within is a cut and paste mix of original
  Arm China code and CIX's NOE enhancements.
* The python library is slow, so slow it is unusable in my opinion (see Benchmark below).
* The python `utils` and `NOE_Engine` package needs to be distributed as a separate wheel file, so you don't have to
  clone the entire AI Model Hub repository.    I have broken out the `NOE_Engine` into a wheel file for my own use
  [here](https://github.com/swdee/cix-utils/).

### Benchmark

Comparing the shipped YOLOv8 `L` sized model and vendors `inference_npu.py` code:

| Timing                    | Python   | C++     |
|---------------------------|----------|---------|
| Setting input tensors     | 17.22ms  | 3.07ms  |
| Inference pass on NPU     | 55.22ms  | 55.54ms |
| Retrieving output tensors | 42.57ms  | 6.72ms  |
| Total time | 115.01ms | 65.33ms |

As can been seen in the table the inference pass on the NPU is the same, however the python code is extremely slow
loading and retrieving the output tensor data.

The C++ code could also be improved with the use of DMA buffers for the input and output tensors, but we need to
wait for a working driver before that is possible.   I believe we could get loading down to <0.5ms and retrieving to <3ms
based on my experience with Rockchips rknn-toolkit.


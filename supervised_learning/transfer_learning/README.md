# Transfer learning

Play around transfer learning

## Usage
It is not dockerize yet -- Using the MacOS M2 installation of tensorflow.

You can use each file into your code, no main files given.
```
from 0-sequential import build_model

... # Your code
```

All the script run through `python3.11` version and use the following package
* Tensorflow 2.12
* Numpy 1.24.3
* Pycodestyle 2.10.0
* Scipy 1.10.1

As the school ask to code with the following version, it could be that some synthaxe are deprecated or some method or functions too
* `Python 3.5`
* Tensorflow 1.12
* Numpy 1.15
* Pycodestyle 2.5
* Scipy 1.3

As TF 1.12 can't being downloaded, and the school want absolutly us been using this version, we have sometime to use the deprecated API, so it can be some issues while running the code 
## Content
Find the following content through this folder
* `0-transfer.py` First transfer model that use only fine-tunning on the `densenet121`. Pre-computation is enable to allow performance boost 
* `0-transfer-bis.py` First transfer model that create a decision model base on the features extraction from the `densenet121`. Pre-computation is enable to allow performance boost
* `1-transfer.py` Change the last layer of the `densenet121`, and train the last dense block to gain accuracy.
* `2-transfer.py` Train the whole model `densenet121`, based on the imagenet model.

## Gain on the CIFAR10 test

| Model             | Top 1%  |
|-------------------|---------|
| 0-transfer.py     | 88.64%  |
| 0-bis-transfer.py | 90.26%  |
| 1-transfer.py     | 92.23%  |
| 2-transfer.py     | 90.96   |

To get a little bit a nuance here, it seem that the full train from scratch on the densenet121 (`2-transfer.py`) may cost more in computation time and cost, and take more time to converge, as it been cut after 10 epochs
Compared to the most simple one here the (`0-transfer`) that take less time to compute, and seems to give good result.

## Models
Find every model on the given .h5 file
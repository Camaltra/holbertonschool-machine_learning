# Error Analyses

Discover and try metric functions

## Usage
It is not dockerize yet -- Using the MacOS M2 installation of tensorflow.

You can use eahc file into your code, no main files given.
```
from 0-sequential import build_model

... # Your code
```

All the script run through `python3.10.11` version and use the following package
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
* Create a model with given parameters (Dropout & L2 Regularization) using the Sequential class `0-sequential.py`
* 
* Compute the gradient descente with the L2 regularization `1-l2_reg_gradient_descent.py`
* Compute the l2 cost with TF `2-l2_reg_cost.py`
* Create a l2 layer with TF `2-l2_reg_create_layer.py`
* Compute a forward propagation using the dropout regularization method `4-dropout_gradient_descent.py`
* Compute the gradient descente with the dropout regularization `5-dropout_gradient_descent.py`
* Create a dropout layer with TF `6-dropout_create_layer.py`
* Decide the early stop of a model without TF `7-early_stopping.py`

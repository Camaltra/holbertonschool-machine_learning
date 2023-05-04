# Error Analyses

Discover and try metric functions

## Usage
All is dockerize into a docker container, to get the same environement every time.

To run the container:
```
docker compose up --build
docker compose run classification bash
```

All the script run through `python3.10` version and use the following package
* Numpy 1.24.2
* Pycodestyle 2.6.0
* Scipy 1.10.1

As the school ask to code with the following version, it could be that some synthaxe are deprecated or some method or functions too
* `Python 3.5`
* Numpy 1.15
* Pycodestyle 2.5
* Scipy 1.3
* Tensorflow 1.12

As TF 1.12 can't being downloaded, and the school want absolutly us been using this version, we have to use the deprecated API and actually can't run the code. So it is all by blind try to make the checker happy. Actually there is no ways to run the code
As something to note, most of the time we can't type our functions and the code. Here is why there are often no typing on all files

## Content
Find the following content through this folder
* Calculate the L2 cost without TF `0-l2_reg_cost.py`
* Compute the gradient descente with the L2 regularization `1-l2_reg_gradient_descent.py`
* Compute the l2 cost with TF `2-l2_reg_cost.py`
* Create a l2 layer with TF `2-l2_reg_create_layer.py`
* Compute a forward propagation using the dropout regularization method `4-dropout_gradient_descent.py`
* Compute the gradient descente with the dropout regularization `5-dropout_gradient_descent.py`
* Create a dropout layer with TF `6-dropout_create_layer.py`
* Decide the early stop of a model without TF `7-early_stopping.py`

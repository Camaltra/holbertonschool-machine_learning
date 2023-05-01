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

As something to note, most of the time we can't type our functions and the code. Here is why there are often no typing on all files

## Content
Find the following content through this folder
* Create a confusion matrix `0-create_confusion.py`
* Compute sensitivity `1-sensitivity.py`
* Compute precision `2-precision.py`
* Compute specificity `3-specificity.py`
* Compute f1_score `4-f1_score.py`
* Two last files are some QCM awser to understand under and over fitting (ie Variance & Bias)

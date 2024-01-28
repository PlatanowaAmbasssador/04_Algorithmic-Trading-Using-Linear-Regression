## Algorithmic investment strategies using simple econometric model

## Description

### Aim
This project aims to check if a well feature-engineered Linear Regression Model helps in obtaining the best strategy for trading SPX (S&P 500). The next day closing price of SPX is predicted, therefore it is a regression problem.

### Data
SPX prices between 1988-01-01 and 2023-10-30 obtained from Yahoo! Finance. As a part of feature engineering, technical indicators are calculated.

### Models
We use the following econometric models:
- linear regression 
Model is implemented using `scikit-learn` library.

### Performance metrics
Because the goal is to make the best investment strategy, instead of using ML evaluation metrics, individually calculated performance metrics for investment strategies are used. The main performance metric is **Information ratio\*\***.

### Validation approach
Walk forward approach for validation and hyperparameter tuning is used. For details see `02_linear_and_logit_regressions.ipynb` and `03_ML_random_forest_model.ipynb`.

## Project structure

- `01_Executor_LR.ipynb` - setting the range of hyperprameters and using functions from `LR_functions.py` to calculate the model.
- `LR_functions.py` - functions and classes used throughout the project.
- `Performance Metrics` - includes the functions to calculate the performance metrics and .ipynb files calculating them.
- `Results` - model predictions generated and the equity lines calculated.
- `Report` - includes the Final Results Report created using R-Markdown. Contains the reuslts of the base model and the sensitivity anlaysis results.
- `Data_nd_WFO.ipynb` - graphical representation of walk forward approach.
- `requirements.txt` - requirements file to run the code.

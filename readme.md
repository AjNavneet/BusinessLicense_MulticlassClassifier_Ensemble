# Business license Multi-class Ensemble Classification

## Aim

Implementing Ensemble techniques to predict license status for the given business.

---

## Data Description

The dataset used is a licensed dataset. It contains information about 86K different businesses over various features. The target variable is the status of the license, which has five different categories.

---

## Tech Stack
- Language: `Python`
- Libraries: `pandas`, `scikit_learn`, `category_encoders`, `numpy`, `os`, `seaborn`, `matplotlib`, `hyperopt`, `xgboost`
  
---

## Approach

1. Data Description
2. Exploratory Data Analysis
3. Data Cleaning
  -  a. Missing Value imputation
  -  b. Outlier Detection
4. Data Imbalance
5. Data Encoding
6. Model Building
  -  a. Random Forest
  -  b. AdaBoost
  -  c. XGBoost
7. Feature importance
8. Hyperparameter tuning
  -  a. Random search optimization
  -  b. Grid search optimization
  -  c. Bayesian optimization

---

## Modular code overview

1. The `input` folder contains the data that we have for analysis. In our case, it contains `Licence_Data.csv`.
2. The `src` folder is the heart of the project. This folder contains all the modularized code for all the above steps in a modularized manner.

    The `model_selection.py` and `preprocessing.py` files contain all the functions in a modularized manner, which are appropriately named. These Python functions are then called inside the `run.py` file.
    
    The `requirements.txt` file has all the required libraries with respective versions. Kindly install the file by using the command `pip install -r requirements.txt`.
    
3. The `output` folder contains an Excel file for classification metrics scores of each model.
4. The `lib` folder is a reference folder. It contains the original Jupyter notebook.
---




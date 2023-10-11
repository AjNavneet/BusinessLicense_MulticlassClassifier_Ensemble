# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import category_encoders as ce
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler

# Function to compare various models and find the best one
def run_models(X_train, y_train, X_test, y_test, model_type='Imbalanced'):
    
    # Define a dictionary of classifiers to be evaluated
    clfs = {'KNNClassifier': KNeighborsClassifier(n_neighbors=3),
            'LogisticRegression': LogisticRegression(),
            'GaussianNB': GaussianNB(),
            'DecisionTreeClassifier': DecisionTreeClassifier(),
            'RandomForestClassifier': RandomForestClassifier(),
            'AdaBoostClassifier': AdaBoostClassifier(),
            'XGBoostClassifier': XGBClassifier()
            }

    # Define columns for the models report
    cols = ['model', 'precision_score', 'recall_score', 'f1_score']

    # Create an empty DataFrame to store the models' performance metrics
    models_report = pd.DataFrame(columns=cols)
    conf_matrix = dict()

    # Iterate through the classifiers and evaluate their performance
    for clf, clf_name in zip(clfs.values(), clfs.keys()):

        # If the classifier is KNN, perform Min-Max scaling on the data
        if clf_name == "KNNClassifier":
            scaler = MinMaxScaler()
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            scaler.fit(X_test)
            X_test = scaler.transform(X_test)

        # Fit the classifier on the training data and make predictions
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        y_score = clf.predict_proba(X_test)[:, 1]

        print('Computing {} - {} '.format(clf_name, model_type))

        # Calculate precision, recall, and F1-score for the classifier
        tmp = pd.Series({'model_type': model_type,
                         'model': clf_name,                       
                         'precision_score': metrics.precision_score(y_test, y_pred, average='macro'),
                         'recall_score': metrics.recall_score(y_test, y_pred, average='macro'),
                         'f1_score': metrics.f1_score(y_test, y_pred, average='macro')})

        # Append the performance metrics to the models report DataFrame
        models_report = models_report.append(tmp, ignore_index=True)    

    # Sort the models report by F1-score and print it
    print(models_report.sort_values(by=["f1_score"]))

    # Define the directory path to save the models report
    parent_dir = os.path.dirname(os.getcwd())
    data_path = "\\output\\"

    # Save the models report as an Excel file
    models_report.to_excel(parent_dir + data_path + "model_report.xlsx")

    return models_report

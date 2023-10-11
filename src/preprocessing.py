# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import category_encoders as ce
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Function for target encoding
def target_encoding(df, col_to_transform):
    """
    This function performs target encoding on the specified columns and returns the transformed dataframe.

    Args:
        df (pandas DataFrame): The input dataframe to be transformed.
        col_to_transform (List): List of columns to be target-encoded.
    """
    # One-hot encode the target variable
    enc = ce.OneHotEncoder().fit(df.target.astype(str))
    y_onehot = enc.transform(df.target.astype(str)
    
    # Get the class names from one-hot encoding
    class_names = y_onehot.columns
    
    # Iterate through class names and perform target encoding
    for class_ in class_names:
        enc = ce.TargetEncoder(smoothing=0)
        temp = enc.fit_transform(df[col_to_transform], y_onehot[class_])
        temp.columns = [str(x) + '_' + str(class_) for x in temp.columns]
        df = pd.concat([df, temp], axis=1)
    
    return df

# Function for random sampling based on target proportion
def random_sampling(df, target, target_prop):
    """
    Function to generate random sampling based on the target proportion.

    Args:
        df (pandas DataFrame): The input dataframe to be sampled.
        target (List): Distinct values in the target column.
        target_prop (List): Fractions of proportions to be present after the sampling process.
    """
    df_list = []
    
    # Iterate through target values and apply sampling
    for i in range(len(target)):
        if target_prop[i] > 1:
            temp_df = df[df.target == target[i]].sample(frac=target_prop[i], replace=True)
        else:
            temp_df = df[df.target == target[i]].sample(frac=target_prop[i], replace=False)
        df_list.append(temp_df)
    
    # Concatenate sampled dataframes to create the final dataframe
    final_df = pd.concat(df_list)
    
    return final_df

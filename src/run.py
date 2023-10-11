import pandas as pd
import numpy as np
import os
from preprocessing import target_encoding
from preprocessing import random_sampling
from sklearn.model_selection import train_test_split
from model_selection import run_models

# Define the parent directory and data path
parent_dir = os.path.dirname(os.getcwd())
data_path = "\\input\\"

# Read the dataset
df = pd.read_csv(parent_dir + data_path + "License_data.csv")

# Change column names
new_col_name = [col.replace(" ", "_").lower() for col in df.columns]
df.columns = new_col_name

# Drop irrelevant or columns with too many missing values
drop_col_list = ["id", "license_id", "ssa", "location", "application_created_date", "account_number", "address"]
df = df.drop(drop_col_list, axis=1)

# Convert string objects to dates
date_columns = ["application_requirements_complete", "payment_date", "license_term_start_date",
                "license_term_expiration_date", "license_approved_for_issuance", "date_issued"]
for col in date_columns:
    df[col] = pd.DatetimeIndex(df[col])

# Calculate the number of days between different application status dates
df["completion_to_start"] = (df.license_term_start_date - df.application_requirements_complete).dt.days
df["start_to_expiry"] = (df.license_term_expiration_date - df.license_term_start_date).dt.days
df["approval_to_issuance"] = (df.date_issued - df.license_approved_for_issuance).dt.days
df["completion_to_payment"] = (df.payment_date - df.application_requirements_complete).dt.days

# Create a binary column to indicate the presence of inquiry details
df["presence_of_enquiry_details"] = np.where(df.ward.isnull() | df.ward_precinct.isnull() | df.police_district.isnull() | df.precinct.isnull(), 0, 1)

# Encode the 'license_status' column as 'target'
df["target"] = df[['license_status']].apply(lambda col: pd.Categorical(col).codes)

# Target encode specific columns
df = target_encoding(df, col_to_transform=["license_description", "state", "city"])

# Resample the data to balance the classes
target_list = np.sort(df.target.unique()).tolist()
target_prop = [0.3, 0.3, 200, 200, 2]
sampled_df = random_sampling(df, target_list, target_prop)

# Select features and target
X = sampled_df[['latitude', 'longitude', 'completion_to_start', 'start_to_expiry', 'approval_to_issuance',
                'completion_to_payment', 'presence_of_enquiry_details', 'license_description_target_1',
                'state_target_1', 'city_target_1', 'license_description_target_2', 'state_target_2',
                'city_target_2', 'license_description_target_3', 'state_target_3', 'city_target_3',
                'license_description_target_4', 'state_target_4', 'city_target_4', 'license_description_target_5',
                'state_target_5', 'city_target_5']]

y = sampled_df['target']

# Perform mean imputation
X = X.fillna(X.mean())

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)

# Run machine learning models
run_models(X_train, y_train, X_test, y_test)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 12:12:34 2017

@author: Austin
"""

# Imports
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import roc_auc_score
import pandas as pd

# Read in data
df_training_data = pd.read_csv('./Kaggle_Titanic_ML/csv_files/train.csv')
df_training_data.pop('Survived')

# Since 'Age' has missing values, we need to fill in those values with the 
# mean age of the entire ship
df_training_data['Age'].fillna(df_training_data.Age.mean(), inplace = True)

# Get non-object columns and subset original data
numeric_cols = [index for index, dtype in df_training_data.dtypes.iteritems() if dtype != 'object']
df_training_data[numeric_cols].head()


# Making our model
titanic_model = RandomForestRegressor(n_estimators=100, oob_score=True, random_state=42)
titanic_model.fit(df_training_data[numeric_cols], df_training_data)

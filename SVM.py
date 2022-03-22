# import library
import gensim.downloader as api
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split

# Basic Package
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing
train_df = pd.read_csv('train.csv', index_col='row_id')
test_df = pd.read_csv('test.csv', index_col='row_id')

# Labeling and Feature Extraction
X = train_df.copy()
y = X.pop('target')
labeled = LabelEncoder()
y = labeled.fit_transform(y)

# stratify => make sure classes are evenly represented across splits
X_train, X_valid, y_train, y_valid = train_test_split(X, y, train_size=0.8)

""""
Modeling Building
""""
# Creating Scalaer with sckit-learn for scaling
scaler = MinMaxScaler(feature_range=(-1,1))

# Fit the X_train
scaler.fit(X_train)
X_train = scaler.transform(X_train) 

# Fit the X_valid
scaler.fit(X_valid)
X_valid = scaler.transform(X_valid) 

# Fit the test
scaler.fit(test_df)
test_scaler = scaler.transform(test_df)

# SVM
svm = SVC(kernel = 'rbf', C = 1, gamma = 2**(-3)) 
svm.fit(X_train, y_train)

# Build prediction
y_pred = svm.predict(test_scaler)
y_pred = y_pred.astype(np.int32)
y_pred = labeled.inverse_transform(y_pred)

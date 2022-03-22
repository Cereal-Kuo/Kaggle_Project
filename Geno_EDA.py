# Basic
import numpy as np 
import pandas as pd 
# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# Importing the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

# Dropped the duplicate
train_df.drop(columns=["row_id"], inplace=True)
y = train_df['target']
X = train_df.drop(columns=['target'])

# Label Encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dataset Spliting for training/ testing set
from sklearn.model_selection import train_test_split
X_train, X_valid , y_train , y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

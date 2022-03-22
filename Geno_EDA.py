"""
資料格式應用
Basic Packages for further usage
"""
# Basic
import numpy as np 
import pandas as pd 

# Plotting
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns
import tools

"""
資料處理
Data Proccessing
"""

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
# Convert the 10 bacteria names to the integers 0 .. 9
y = encoder.fit_transform(y)

# Dataset Spliting for training/ testing set
from sklearn.model_selection import train_test_split
X_train, X_valid , y_train , y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Check content
tools.check(train_df)

# Feature Extract
feature = [col for col in train_df.columns if col not in ['row_id', 'target']]

# Print the content
fig, axs = plt.subplots(5, 3, figsize=(17,20))
i = 0
for f in f:
    current_ax = axs.flat[i]
    current_ax.hist(train_df[f], bins=100)
    current_ax.set_title(f)
    current_ax.grid()
    i = i + 1

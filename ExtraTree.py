# Importing Packages
import numpy as np
import pandas as pd


# skitlearn packages
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.ensemble import ExtraTreesClassifier

# Importing the dataset
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
test_df.drop(columns=['row_id'])

y = train_df['target']

X = train_df.drop(columns=['row_id','target'])

# Label Encoder
from sklearn.preprocessing import LabelEncoder
encoder = LabelEncoder()
y = encoder.fit_transform(y)

# Dataset Spliting for training/ testing set
from sklearn.model_selection import train_test_split
X_train, X_valid , y_train , y_valid = train_test_split(X, y, test_size = 0.2, random_state = 42)

# Parameter settings 
"""
number of split = 10
using all the chips
random_state = 42
number of estimators = 1000
"""
kf = StratifiedKFold(n_splits = 10, shuffle=True, random_state=42)
et_params = {
    'n_estimators': 1000,
    'n_jobs': -1,
    'random_state': 42
}

# Holders for the accuracy
pred_validation_all_et = []
validation_all = []
validation_ids_all = []

y_pred_test_et = []
y_pred_test_prob_et = []

importances_et = []
accs_et = []

# Establish Modeling
%%time

for fold, (trn_idx, val_idx) in enumerate(kf.split(X=X_train, y = y_train)):
    validation_ids_all.append(val_idx)
    print("===== Number of Fold {} =====".format(fold))
    X_tr = X_train.iloc[trn_idx]
    y_tr = y_train[trn_idx]
    X_val = X_train.iloc[val_idx]
    y_val = y_train[val_idx]
    sample_weight_tr = sample_weights.iloc[trn_idx].values
    sample_weight_val = sample_weights.iloc[val_idx].values
    
    model_et = ExtraTreesClassifier(**et_params)
    
    model_et.fit(X_tr, y_tr, sample_weight_tr)
        
    importances_et.append(model_et.feature_importances_)
    
    pred_val_et = model_et.predict(X_val)
    pred_validation_all_et.append(pred_val_et)
    validation_all.append(y_val)
    
    acc_et = accuracy_score(y_true = y_val, y_pred = pred_val_et, sample_weight=sample_weight_val)
    accs_et.append(acc_et)
    
    print("FOLD", fold, "ETC Accuracy:", acc_et)
    
    # Test data predictions
    y_pred_test_et.append(model_et.predict(X_test))
    y_pred_test_prob_et.append(model_et.predict_proba(X_test))
    
print("======================================")
print("Mean Accuracy (all folds) - ETC:", np.mean(accs_et))


# Final Prediction Extract
y_pred_et = model_et.predict(X_test)
y_pred_et = encoder.inverse_transform(y_pred_et)
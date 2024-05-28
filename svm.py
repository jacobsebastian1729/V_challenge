import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import os
import zipfile

train = pd.read_csv("train_merged.csv")
test = pd.read_csv("test_merged.csv")

null_columns = train.columns[train.isnull().all()]
print(len(null_columns))
#print(null_columns)

df2_train = train.drop(columns = null_columns)
df2_test = test.drop(columns = null_columns)

print(df2_train.shape)
print(df2_test.shape)

print(df2_train.isnull().sum().sum())
print(df2_test.isnull().sum().sum())

# Filling missing values with 0
df2_train.fillna(0, inplace=True)
df2_test.fillna(0, inplace=True)

# Checking for missing values again
print(df2_train.isnull().sum().sum())
print(df2_test.isnull().sum().sum())


train_indices = np.arange(2, len(train), 5)  # Indices 2, 7, 12, 17, etc.
train_indices = np.concatenate([train_indices, np.arange(3, len(train), 5)])  # Adding indices 3, 8, 13, 18, etc.
train_indices = np.unique(train_indices)  # Ensure indices are unique




np.random.seed(40)
X_train = df2_train.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
y_train = df2_train['risk_level']

X_train = X_train.iloc[train_indices]
y_train = y_train.iloc[train_indices]

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the SVM model
classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
classifier.fit(X_train_scaled, y_train)

# Preparing the test data and making predictions
X_test = df2_test.drop(columns=['ChassisId_encoded', 'gen', 'v_category'])
X_test_scaled = scaler.transform(X_test)

y_pred = classifier.predict(X_test_scaled)

# Creating a DataFrame for the predictions
df_pred = pd.DataFrame(data=y_pred, columns=['pred'])

print(df_pred['pred'].value_counts())

df_pred.to_csv('prediction.csv', index=False)

def compress_file(input_file, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file, os.path.basename(input_file))
input_file = 'prediction.csv'  # Input file to compress
output_zip = 'prediction.csv.zip'  # Output ZIP archive
compress_file(input_file, output_zip)

print("done")

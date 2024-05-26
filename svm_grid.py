import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler


import os
import zipfile

df2_train = pd.read_csv("train4_corr.csv")
df2_test = pd.read_csv("test4_corr.csv")

print(df2_train.isnull().sum().sum())
print(df2_test.isnull().sum().sum())

# Filling missing values with 0
df2_train.fillna(0, inplace=True)
df2_test.fillna(0, inplace=True)

# Checking for missing values again
print(df2_train.isnull().sum().sum())
print(df2_test.isnull().sum().sum())

np.random.seed(40)
X_train = df2_train.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
y_train = df2_train['risk_level']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Preparing the test data and making predictions
X_test = df2_test.drop(columns=['ChassisId_encoded', 'gen', 'v_category'])
X_test_scaled = scaler.transform(X_test)

# Define the parameter grid for grid search
param_grid = {
    'gamma': ['scale'],#, 'auto', 0.001, 0.01, 0.1, 1],
    'coef0': [0.0]#, 0.1, 0.5, 1.0, 2.0]
}

# Print the results of the grid search
print("Grid search results:")
for gamma in param_grid['gamma']:
    for coef0 in param_grid['coef0']:
        print(f"gamma: {gamma} and cef0: {coef0}")
        # Initialize and train the SVM model
        classifier = SVC(kernel='sigmoid', gamma=gamma, coef0=coef0, random_state=42)
        classifier.fit(X_train_scaled, y_train)
        y_pred = classifier.predict(X_test_scaled)
        df_pred = pd.DataFrame(data=y_pred, columns=['pred'])
        print(df_pred['pred'].value_counts())






# Creating a DataFrame for the predictions




#df_pred.to_csv('prediction.csv', index=False)

#def compress_file(input_file, output_zip):
#    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
#        zipf.write(input_file, os.path.basename(input_file))
#input_file = 'prediction.csv'  # Input file to compress
#output_zip = 'prediction.csv.zip'  # Output ZIP archive
#compress_file(input_file, output_zip)

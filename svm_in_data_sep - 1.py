import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support
from sklearn.ensemble import VotingClassifier
import os
import zipfile

# Load the datasets
train = pd.read_csv("train_merged_4.csv")
test1 = pd.read_csv("train_merged_1.csv")
test = pd.read_csv("test_merged.csv")

# Drop columns with all null values
null_columns = train.columns[train.isnull().all()]
df2_train = train.drop(columns=null_columns)
df2_test1 = test1.drop(columns=null_columns)
df2_test = test.drop(columns=null_columns)

# Fill missing values with 0
df2_train.fillna(0, inplace=True)
df2_test1.fillna(0, inplace=True)
df2_test.fillna(0, inplace=True)

# Select relevant columns
train_columns_to_select = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
                           17, 19, 21, 22, 23, 24, 25, 26, 27, 29, 30, 36, 37, 39,
                           40, 43, 47, 51, 59, 60, 64, 65, 66, 73, 76, 78, 79, 81,
                           83, 84, 85, 86, 87, 88, 90, 91, 93, 97, 98, 102, 103, 104,
                           108, 110, 114, 115, 116, 118, 119, 120, 122, 123, 124, 127,
                           128, 129, 130, 132, 133, 143, 144, 151, 152, 155, 161, 162,
                           169, 170, 171, 173, 174, 175, 177, 178, 179, 180, 181, 182,
                           184, 186, 187, 188, 189, 190, 193, 194, 195, 197, 198, 199,
                           200, 201, 202, 204, 205, 206, 207, 208, 210, 211, 213, 214,
                           215, 220, 221, 223, 224, 225, 226, 227, 228, 229, 231, 232,
                           233, 234, 235, 236, 238, 239, 240, 241, 242, 243, 244, 245,
                           246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 259, 260,
                           261, 262, 265, 283, 284, 285, 286, 287, 288, 289, 290, 291,
                           292, 295, 296, 298, 299, 301, 302, 303, 304, 305, 306, 307,
                           308, 309, 310, 311]

test_columns_to_select = [i - 1 for i in train_columns_to_select][1:]

df2_train = df2_train.iloc[:, train_columns_to_select]
df2_test1 = df2_test1.iloc[:, train_columns_to_select]
df2_test = df2_test.iloc[:, test_columns_to_select]

# Split the training data into features and labels
X_train = df2_train.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
y_train = df2_train['risk_level']

# Create binary labels for each classifier
y_train_low_vs_rest = y_train.apply(lambda x: 'Not Low' if x != 'Low' else 'Low')
y_train_medium_vs_rest = y_train.apply(lambda x: 'Not Medium' if x != 'Medium' else 'Medium')
y_train_high_vs_rest = y_train.apply(lambda x: 'Not High' if x != 'High' else 'High')

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the classifiers
classifier_low_vs_rest = SVC(kernel='linear', decision_function_shape='ovr', probability=True, random_state=42)
classifier_low_vs_rest.fit(X_train_scaled, y_train_low_vs_rest)

classifier_medium_vs_rest = SVC(kernel='linear', decision_function_shape='ovr', probability=True, random_state=42)
classifier_medium_vs_rest.fit(X_train_scaled, y_train_medium_vs_rest)

classifier_high_vs_rest = SVC(kernel='linear', decision_function_shape='ovr', probability=True, random_state=42)
classifier_high_vs_rest.fit(X_train_scaled, y_train_high_vs_rest)

# Function to calculate macro average F1-score
def macro_avg_f1(y_true, y_pred):
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)
    return round(sum(f1) / len(f1), 2)

print("Prediction on test1")
X_test1 = df2_test1.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test1_scaled = scaler.transform(X_test1)

# Predict probabilities for each class
y_pred_1_low_prob = classifier_low_vs_rest.predict_proba(X_test1_scaled)[:, 1]
y_pred_1_medium_prob = classifier_medium_vs_rest.predict_proba(X_test1_scaled)[:, 1]
y_pred_1_high_prob = classifier_high_vs_rest.predict_proba(X_test1_scaled)[:, 1]

# Combine the predictions by selecting the class with the highest probability
final_predictions_1 = []
for i in range(len(X_test1)):
    probs = {'Low': y_pred_1_low_prob[i], 'Medium': y_pred_1_medium_prob[i], 'High': y_pred_1_high_prob[i]}
    final_predictions_1.append(max(probs, key=probs.get))

final_predictions_1 = pd.Series(final_predictions_1, index=X_test1.index)

# Evaluate the results
y1_true = pd.DataFrame()
y1_true['risk_level'] = df2_test1['risk_level']
y1_true['pred'] = final_predictions_1

score1 = macro_avg_f1(y1_true['risk_level'], y1_true['pred'])
print('\nScores1 Details:\n', score1)
calculate_accuracies(y1_true['risk_level'], y1_true['pred'])

# Prediction for test data
X_test = df2_test.drop(columns=['ChassisId_encoded', 'gen', 'v_category'])
X_test_scaled = scaler.transform(X_test)

# Predict probabilities for each class
y_pred_low_prob = classifier_low_vs_rest.predict_proba(X_test_scaled)[:, 1]
y_pred_medium_prob = classifier_medium_vs_rest.predict_proba(X_test_scaled)[:, 1]
y_pred_high_prob = classifier_high_vs_rest.predict_proba(X_test_scaled)[:, 1]

# Combine the predictions by selecting the class with the highest probability
final_predictions = []
for i in range(len(X_test)):
    probs = {'Low': y_pred_low_prob[i], 'Medium': y_pred_medium_prob[i], 'High': y_pred_high_prob[i]}
    final_predictions.append(max(probs, key=probs.get))

final_predictions = pd.Series(final_predictions, index=X_test.index)

# Save the predictions to CSV
df_pred = pd.DataFrame(data=final_predictions, columns=['pred'])
print(df_pred['pred'].value_counts())
df_pred.to_csv('prediction.csv', index=True)  # Ensure index is saved

# Compress the output file
def compress_file(input_file, output_zip):
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(input_file, os.path.basename(input_file))

input_file = 'prediction.csv'
output_zip = 'prediction.csv.zip'
compress_file(input_file, output_zip)

print("done")

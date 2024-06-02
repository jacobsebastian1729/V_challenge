import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import os
import zipfile

train = pd.read_csv("train_merged_4.csv")
test1 = pd.read_csv("train_merged_1.csv")
test2 = pd.read_csv("train_merged_2.csv")
test3 = pd.read_csv("train_merged_3.csv")
test4 = pd.read_csv("train_merged_4.csv")
test5 = pd.read_csv("train_merged_5.csv")

null_columns = train.columns[train.isnull().all()]
print(len(null_columns))
#print(null_columns)

df2_train = train.drop(columns = null_columns)
df2_test1 = test1.drop(columns = null_columns)
df2_test2 = test2.drop(columns = null_columns)
df2_test3 = test3.drop(columns = null_columns)
df2_test4 = test4.drop(columns = null_columns)
df2_test5 = test5.drop(columns = null_columns)

print(df2_train.shape)
print(df2_test1.shape)
print(df2_test2.shape)
print(df2_test3.shape)
print(df2_test4.shape)
print(df2_test5.shape)

print(df2_train.isnull().sum().sum())
print(df2_test1.isnull().sum().sum())
print(df2_test2.isnull().sum().sum())
print(df2_test3.isnull().sum().sum())
print(df2_test4.isnull().sum().sum())
print(df2_test5.isnull().sum().sum())

# Filling missing values with 0
df2_train.fillna(0, inplace=True)
df2_test1.fillna(0, inplace=True)
df2_test2.fillna(0, inplace=True)
df2_test3.fillna(0, inplace=True)
df2_test4.fillna(0, inplace=True)
df2_test5.fillna(0, inplace=True)

# Checking for missing values again
print(df2_train.isnull().sum().sum())
#print(df2_test.isnull().sum().sum())

np.random.seed(40)
X_train = df2_train.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
y_train = df2_train['risk_level']

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Initialize and train the SVM model
classifier = SVC(kernel='linear', decision_function_shape='ovr', random_state=42)
classifier.fit(X_train_scaled, y_train)



def macro_avg_f1(y_true, y_pred):
    # Get precision, recall, and f1-score for each class
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred)

    # Calculate macro average F1-score
    macro_avg_f1 = sum(f1) / len(f1)

    return round(macro_avg_f1, 2)



# Example usage:
# Assuming y1_true and y1_pred are defined


def calculate_accuracies(y_true, y_pred):
    """
    Calculate the overall and category-wise accuracies, and print the results.

    Parameters:
    y_true (pd.Series): The true labels.
    y_pred (pd.Series): The predicted labels.

    Returns:
    dict: A dictionary containing the overall accuracy and accuracies for each category.
    """

    # Calculate overall accuracy
    total_correct = (y_true == y_pred).sum()
    total_predictions = len(y_true)
    overall_accuracy = (total_correct / total_predictions) * 100

    # Calculate category-wise accuracy
    def category_accuracy(y_true, y_pred, category):
        category_true = y_true == category
        correct_predictions = (y_true[category_true] == y_pred[category_true]).sum()
        total_category = category_true.sum()
        if total_category == 0:
            return 'N/A'
        else:
            return (correct_predictions / total_category) * 100

    accuracy_high = category_accuracy(y_true, y_pred, 'High')
    accuracy_medium = category_accuracy(y_true, y_pred, 'Medium')
    accuracy_low = category_accuracy(y_true, y_pred, 'Low')

    # Calculate true and predicted counts for each category
    def category_distribution(y_true, y_pred, category):
        category_true = y_true == category
        total_category = category_true.sum()
        predicted_low = (y_pred[category_true] == 'Low').sum()
        predicted_medium = (y_pred[category_true] == 'Medium').sum()
        predicted_high = (y_pred[category_true] == 'High').sum()
        return total_category, predicted_low, predicted_medium, predicted_high

    true_low, pred_low_as_low, pred_low_as_medium, pred_low_as_high = category_distribution(y_true, y_pred, 'Low')
    true_medium, pred_medium_as_low, pred_medium_as_medium, pred_medium_as_high = category_distribution(y_true, y_pred, 'Medium')
    true_high, pred_high_as_low, pred_high_as_medium, pred_high_as_high = category_distribution(y_true, y_pred, 'High')

    # Print the results
    print(f'Overall Accuracy: {overall_accuracy:.2f}%')
    print(f'High Risk Accuracy: {accuracy_high if accuracy_high == "N/A" else f"{accuracy_high:.2f}%"}')
    print(f'Medium Risk Accuracy: {accuracy_medium if accuracy_medium == "N/A" else f"{accuracy_medium:.2f}%"}')
    print(f'Low Risk Accuracy: {accuracy_low if accuracy_low == "N/A" else f"{accuracy_low:.2f}%"}')

    print(f'\nTrue Low: {true_low}')
    print(f'  Predicted as Low: {pred_low_as_low}')
    print(f'  Predicted as Medium: {pred_low_as_medium}')
    print(f'  Predicted as High: {pred_low_as_high}')

    print(f'\nTrue Medium: {true_medium}')
    print(f'  Predicted as Low: {pred_medium_as_low}')
    print(f'  Predicted as Medium: {pred_medium_as_medium}')
    print(f'  Predicted as High: {pred_medium_as_high}')

    print(f'\nTrue High: {true_high}')
    print(f'  Predicted as Low: {pred_high_as_low}')
    print(f'  Predicted as Medium: {pred_high_as_medium}')
    print(f'  Predicted as High: {pred_high_as_high}')

    return {
        'overall_accuracy': overall_accuracy,
        'High_accuracy': accuracy_high,
        'Medium_accuracy': accuracy_medium,
        'Low_accuracy': accuracy_low,
        'True_Low': true_low,
        'Pred_Low_as_Low': pred_low_as_low,
        'Pred_Low_as_Medium': pred_low_as_medium,
        'Pred_Low_as_High': pred_low_as_high,
        'True_Medium': true_medium,
        'Pred_Medium_as_Low': pred_medium_as_low,
        'Pred_Medium_as_Medium': pred_medium_as_medium,
        'Pred_Medium_as_High': pred_medium_as_high,
        'True_High': true_high,
        'Pred_High_as_Low': pred_high_as_low,
        'Pred_High_as_Medium': pred_high_as_medium,
        'Pred_High_as_High': pred_high_as_high
    }










print("prediction on test1")
# Preparing the test data and making predictions
X_test1 = df2_test1.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test1_scaled = scaler.transform(X_test1)

y1_pred = classifier.predict(X_test1_scaled)

# Creating a DataFrame for the predictions
#df_pred1 = pd.DataFrame(data=y_pred1, columns=['pred'])


y1_true = pd.DataFrame()
y1_true['risk_level'] = df2_test1['risk_level']
y1_true['pred'] = y1_pred


score1 = macro_avg_f1(y1_true['risk_level'],
                            y1_true['pred'])


print('\nScores1 Details:\n', score1)

calculate_accuracies(y1_true['risk_level'], y1_true['pred'])

print("prediction on test2")
# Preparing the test data and making predictions
X_test2 = df2_test2.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test2_scaled = scaler.transform(X_test2)

y2_pred = classifier.predict(X_test2_scaled)

# Creating a DataFrame for the predictions
#df_pred1 = pd.DataFrame(data=y_pred1, columns=['pred'])


y2_true = pd.DataFrame()
y2_true['risk_level'] = df2_test2['risk_level']
y2_true['pred'] = y2_pred


score2 = macro_avg_f1(y2_true['risk_level'],
                            y2_true['pred'])


print('\nScores2 Details:\n', score2)


calculate_accuracies(y2_true['risk_level'], y2_true['pred'])

print("prediction on test3")
# Preparing the test data and making predictions
X_test3 = df2_test3.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test3_scaled = scaler.transform(X_test3)

y3_pred = classifier.predict(X_test3_scaled)

# Creating a DataFrame for the predictions
#df_pred1 = pd.DataFrame(data=y_pred1, columns=['pred'])


y3_true = pd.DataFrame()
y3_true['risk_level'] = df2_test3['risk_level']
y3_true['pred'] = y3_pred


score3 = macro_avg_f1(y3_true['risk_level'],
                            y3_true['pred'])


print('\nScores3 Details:\n', score3)


calculate_accuracies(y3_true['risk_level'], y3_true['pred'])


print("prediction on test4")
# Preparing the test data and making predictions
X_test4 = df2_test4.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test4_scaled = scaler.transform(X_test4)

y4_pred = classifier.predict(X_test4_scaled)

# Creating a DataFrame for the predictions
#df_pred1 = pd.DataFrame(data=y_pred1, columns=['pred'])


y4_true = pd.DataFrame()
y4_true['risk_level'] = df2_test4['risk_level']
y4_true['pred'] = y4_pred


score4 = macro_avg_f1(y4_true['risk_level'],
                            y4_true['pred'])


print('\nScores4 Details:\n', score4)


calculate_accuracies(y4_true['risk_level'], y4_true['pred'])


print("prediction on test5")
# Preparing the test data and making predictions
X_test5 = df2_test5.drop(columns=['ChassisId_encoded', 'gen', 'risk_level', 'v_category'])
X_test5_scaled = scaler.transform(X_test5)

y5_pred = classifier.predict(X_test5_scaled)

# Creating a DataFrame for the predictions
#df_pred1 = pd.DataFrame(data=y_pred1, columns=['pred'])


y5_true = pd.DataFrame()
y5_true['risk_level'] = df2_test5['risk_level']
y5_true['pred'] = y5_pred


score5 = macro_avg_f1(y5_true['risk_level'],
                            y5_true['pred'])

#score5_l = macro_avg_f1(y5_true[y5_true['risk_level'] == 'Low']['risk_level'],
#                            y5_true[y5_true['risk_level'] == 'Low']['pred'])
#score5_m = macro_avg_f1(y5_true[y5_true['risk_level'] == 'Medium']['risk_level'],
#                            y5_true[y5_true['risk_level'] == 'Medium']['pred'])
#score5_h = macro_avg_f1(y5_true[y5_true['risk_level'] == 'High']['risk_level'],
#                            y5_true[y5_true['risk_level'] == 'High']['pred'])

print('\nScores5 Details:\n', score5)
#print('\nScores5_l Details:\n', score5_l)
#print('\nScores5_m Details:\n', score5_m)
#print('\nScores5_h Details:\n', score5_h)

calculate_accuracies(y5_true['risk_level'], y5_true['pred'])

#print(df_pred['pred'].value_counts())


#df_pred.to_csv('prediction.csv', index=False)

#def compress_file(input_file, output_zip):
#    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
#        zipf.write(input_file, os.path.basename(input_file))
#input_file = 'prediction.csv'  # Input file to compress
#output_zip = 'prediction.csv.zip'  # Output ZIP archive
#compress_file(input_file, output_zip)

print("done")

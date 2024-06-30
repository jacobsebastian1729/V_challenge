import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_fscore_support

import os
import zipfile

df = pd.read_csv('train_merged.csv')
print(df.head(100))
# Step 2: Filter the DataFrame to include only rows where Timesteps <= 10
filtered_df = df[df['Timesteps'] <= 10]

print(filtered_df.shape)
print(filtered_df.head(100))

# Step 3: Save the new DataFrame to a CSV file
#filtered_df.to_csv('filtered_dataframe.csv', index=False)

#print("Filtered DataFrame has been saved to 'filtered_dataframe.csv'")


print("done")

import numpy as np
import pandas as pd

df = pd.read_csv("train_merged_4.csv")

df_new = df.iloc[:, :17]

# Save the new DataFrame to a CSV file
df_new.to_csv('17_dataframe_4.csv', index=False)

print("done")

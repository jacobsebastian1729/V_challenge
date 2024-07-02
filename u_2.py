import pandas as pd
import numpy as np

train = pd.read_csv('train_gen1.csv')
print(train.head())
print(train.shape)
train_variant = pd.read_csv('train_variants.csv')
print(train_variant.shape)
merged_df = pd.merge(train_variant, train, on=["Timesteps", "ChassisId_encoded", "gen", "risk_level"], how="inner")


print(merged_df.head())
print(np.size(merged_df))
print(np.shape(merged_df))
merged_df.to_csv('train_merged.csv', index = False)

#print(train['ChassisId_encoded'].nunique())
#train_ChassisId = train[['Timesteps', 'ChassisId_encoded', 'gen', 'risk_level']]
#print(train_ChassisId.shape)
#train_ChassisId.to_csv('train_ChassisId.csv', index = False)

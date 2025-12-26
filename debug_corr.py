
import pandas as pd
import numpy as np

df = pd.read_csv("dev/data/master_data.csv")

print("Correlation check:")
cols = ['Efficiency', 'RnD_to_Revenue']
print(df[cols].corr())

print("\nSample values:")
print(df[cols].head(10))

# Check difference
df['Diff'] = df['Efficiency'] - df['RnD_to_Revenue']
print("\nMax Diff:", df['Diff'].abs().max())
print("Mean Diff:", df['Diff'].abs().mean())


import pandas as pd
import numpy as np

df = pd.read_csv("dev/data/master_data.csv")

print("Checking new correlation between Efficiency and RnD_to_Revenue:")
cols = ['Efficiency', 'RnD_to_Revenue']
print("\nCorrelation Matrix:")
print(df[cols].corr())

print("\nSample values (first 10 rows):")
print(df[cols].head(10))

# Check if still identical
df['Diff'] = df['Efficiency'] - df['RnD_to_Revenue']
print(f"\nMax Absolute Difference: {df['Diff'].abs().max():.6f}")
print(f"Mean Absolute Difference: {df['Diff'].abs().mean():.6f}")

# Check by sector
print("\n--- By Sector ---")
for sector in df['Sector'].unique():
    subset = df[df['Sector'] == sector]
    corr = subset[cols].corr().iloc[0, 1]
    print(f"{sector}: Correlation = {corr:.4f}")

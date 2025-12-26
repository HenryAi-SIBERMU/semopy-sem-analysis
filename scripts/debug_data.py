
import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv("dev/data/master_data.csv")

print("Total Rows:", len(df))
print("\nMissing Values per Column:")
print(df.isnull().sum())

print("\n--- Data for SEM (Cols: RnD_to_Revenue, Efficiency, ROA, Tobins_Q) ---")
sem_df = df[['RnD_to_Revenue', 'Efficiency', 'ROA', 'Tobins_Q']]
print("Rows before dropna:", len(sem_df))
sem_clean = sem_df.dropna()
print("Rows after dropna:", len(sem_clean))

print("\nStandard Deviations (Must be > 0):")
print(sem_clean.std())

if len(sem_clean) < 5:
    print("\n[DIAGNOSIS]: Valid rows are too few!")
    print("Example of missing data rows:")
    print(df[['Company', 'Sector', 'RnD_to_Revenue', 'Efficiency', 'ROA', 'Tobins_Q']].head(20))

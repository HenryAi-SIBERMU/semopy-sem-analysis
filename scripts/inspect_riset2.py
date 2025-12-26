
import pandas as pd

try:
    df = pd.read_csv("dev/data/Riset2-preview_startup.csv")
    print("Columns:", df.columns.tolist())
    print("-" * 20)
    print("First row values:", df.iloc[0].tolist())
except Exception as e:
    print(e)

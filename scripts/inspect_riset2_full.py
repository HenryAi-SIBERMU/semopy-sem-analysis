
import pandas as pd

try:
    df = pd.read_csv("dev/data/Riset2-preview_startup.csv")
    print("ALL COLUMNS:")
    for col in df.columns:
        print(f"'{col}'")
except Exception as e:
    print(e)

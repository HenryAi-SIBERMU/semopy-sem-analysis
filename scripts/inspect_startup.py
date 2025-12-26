
import pandas as pd

df = pd.read_excel("dev/data/DATA STARTUP.xlsx", header=None)

# Find coords
print("Searching for keywords...")
for r in range(len(df)):
    for c in range(len(df.columns)):
        val = str(df.iloc[r, c])
        if "R&D" in val or "Efisiensi" in val or "Revenue" in val:
            print(f"Found '{val}' at Row {r}, Col {c}")

print("\nSample Data at Row 7 (below found header?):")
if len(df) > 7:
    print(df.iloc[7].values)

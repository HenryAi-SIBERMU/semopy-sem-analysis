
import pandas as pd
from semopy import Model
import numpy as np

# Load Data
df = pd.read_csv("dev/data/master_data.csv")

# Clean cols
cols_needed = ['RnD_to_Revenue', 'Efficiency', 'ROA', 'Tobins_Q']
# Ensure columns exist
if not all(col in df.columns for col in cols_needed):
    print("Cols missing")
    exit()

clean_df = df[cols_needed].dropna()

# Define Model
desc = """
Efficiency ~ RnD_to_Revenue
ROA ~ Efficiency
Tobins_Q ~ Efficiency + ROA
"""

print(f"Data rows: {len(clean_df)}")
model = Model(desc)
model.fit(clean_df)

# Check Inspect
print("\n--- Model Inspection ---")
inspect = model.inspect()
print(inspect[inspect['op'] == '~'])

# Check Prediction
print("\n--- Prediction Check ---")
preds = model.predict(clean_df)
print("Predicted Columns:", preds.columns)
print("Actual Efficiency Head:", clean_df['Efficiency'].head().values)
print("Pred Efficiency Head:", preds['Efficiency'].head().values)

# Check if identical
diff = (clean_df['Efficiency'].values - preds['Efficiency'].values)
print("Max Diff:", np.max(np.abs(diff)))

# Check calc_stats
print("\n--- Calc Stats ---")
try:
    stats = model.calc_stats()
    print("Stats Type:", type(stats))
    print("Stats Keys:", stats.keys() if isinstance(stats, dict) else "Not dict")
    if 'R2' in stats:
        print("R2 Values:", stats['R2'])
except Exception as e:
    print("Calc Stats Failed:", e)

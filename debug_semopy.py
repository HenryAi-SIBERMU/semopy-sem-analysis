
import semopy
if hasattr(semopy, 'calc_stats'):
    print("semopy.calc_stats exists")
else:
    print("semopy.calc_stats MISSING")

import pandas as pd
from semopy import Model

# Simple Test Data
data = pd.DataFrame({
    'X': range(10),
    'Y': [i*2 + 1 for i in range(10)],
    'M': [i*0.5 for i in range(10)]
})

desc = """
Y ~ X
M ~ X
"""
model = Model(desc)
model.fit(data)

print("Model attributes:", dir(model))

try:
    print("R-square attr:", model.r_square)
except AttributeError as e:
    print("R-square attr FAILED:", e)

# Try calc_stats
try:
    stats = semopy.calc_stats(model)
    print("Calc Stats output:")
    print(stats)
except Exception as e:
    print("Calc stats FAILED:", e)
    
# Check inspect
insp = model.inspect()
print("Inspect columns:", insp.columns)

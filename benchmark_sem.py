
import pandas as pd
import numpy as np
import time
from semopy import Model

# Mock Data (Similar to Startup dataset)
N = 35
np.random.seed(42)
data = pd.DataFrame({
    'RnD': np.random.rand(N),
    'Eff': np.random.rand(N),
    'Val': np.random.rand(N)
})

model_spec = """
Eff ~ RnD
Val ~ Eff + RnD
"""
model = Model(model_spec)

start_time = time.time()
n_boots = 50
for i in range(n_boots):
    sample = data.sample(frac=1, replace=True)
    model.fit(sample)
    
end_time = time.time()
duration = end_time - start_time

print(f"Time for {n_boots} runs: {duration:.4f} seconds")
print(f"Avg per run: {duration/n_boots:.4f} seconds")
print(f"Estimated 500 runs: {(duration/n_boots)*500:.2f} seconds")

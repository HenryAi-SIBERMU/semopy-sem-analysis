
import pandas as pd
import numpy as np

# Load existing data
file_path = "dev/data/master_data.csv"
df = pd.read_csv(file_path)

np.random.seed(42)

# Function to generate synthetic scores (1-10 scale)
def generate_score(row):
    # Base score random
    base = np.random.uniform(5, 9)
    # Slight increase over years
    year_effect = (row['Year'] - 2020) * 0.2
    # Company specific random factor (using hash of company name for consistency)
    u_id = hash(row['Company']) % 100 / 100.0
    
    score = base + year_effect + u_id
    return min(max(score, 1), 10)

# Apply to transport sector mostly, but we can do all
df['IT_Governance'] = df.apply(generate_score, axis=1)
df['Digital_Transformation'] = df.apply(lambda x: generate_score(x) + np.random.normal(0, 0.5), axis=1)

# Ensure within bounds 1-10
df['IT_Governance'] = df['IT_Governance'].clip(1, 10)
df['Digital_Transformation'] = df['Digital_Transformation'].clip(1, 10)

# Save back
df.to_csv(file_path, index=False)
print("Added moderators: IT_Governance, Digital_Transformation")

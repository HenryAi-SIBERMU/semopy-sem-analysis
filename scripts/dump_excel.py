
import pandas as pd

df = pd.read_excel("dev/data/DATA STARTUP.xlsx", header=None)

with open("excel_dump.txt", "w", encoding="utf-8") as f:
    # Print header indices
    f.write("\t" + "\t".join([str(i) for i in range(20)]) + "\n")
    
    for r in range(min(20, len(df))):
        row_vals = []
        for c in range(min(20, len(df.columns))):
            val = str(df.iloc[r, c]).replace("\t", " ").replace("\n", " ")
            if val == "nan": val = "."
            # Truncate long values
            if len(val) > 15: val = val[:12] + "..."
            row_vals.append(val)
        f.write(f"{r}\t" + "\t".join(row_vals) + "\n")

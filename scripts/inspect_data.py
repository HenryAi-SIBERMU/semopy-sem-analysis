
import pandas as pd


files = ['DATA TRANSPORT.xlsx', 'DATA STARTUP.xlsx']

with open('inspection_output.txt', 'w', encoding='utf-8') as f_out:
    for file in files:
        f_out.write(f"--- ANALYZING {file} ---\n")
        try:
            # Read the Excel file
            df = pd.read_excel(file)
            
            # Print column names
            f_out.write(f"Columns: {list(df.columns)}\n")
            
            # Print first few rows to see data format
            f_out.write("First 5 rows:\n")
            f_out.write(df.head().to_string() + "\n")
            
            # Check for missing values
            f_out.write("Missing values per column:\n")
            f_out.write(df.isnull().sum().to_string() + "\n")
            
        except Exception as e:
            f_out.write(f"Error reading {file}: {e}\n")
        f_out.write("\n")


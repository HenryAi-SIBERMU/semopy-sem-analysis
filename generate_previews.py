
import pandas as pd
import os

def convert_excel_to_csv(input_path, output_path, header_row=0):
    print(f"Converting {input_path} to {output_path}...")
    try:
        # Read Excel
        df = pd.read_excel(input_path, header=header_row)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        print("Success.")
    except Exception as e:
        print(f"Error converting {input_path}: {e}")

if __name__ == "__main__":
    base_dir = "dev/data"
    
    # 1. Transport Data (Standard Header)
    convert_excel_to_csv(
        os.path.join(base_dir, "DATA TRANSPORT.xlsx"),
        os.path.join(base_dir, "preview_transport.csv"),
        header_row=0
    )
    
    # 2. Startup Data (Header at Row 2 / Index 2 based on previous debugging)
    convert_excel_to_csv(
        os.path.join(base_dir, "DATA STARTUP.xlsx"),
        os.path.join(base_dir, "preview_startup.csv"),
        header_row=2
    )

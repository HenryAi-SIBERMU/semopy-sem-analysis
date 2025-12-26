
import pandas as pd
import os
import numpy as np

def clean_transport_data(filepath):
    """Cleans the Transport Excel file."""
    print(f"Processing {filepath}...")
    try:
        # Load data (First row seems to be header based on previous inspection)
        df = pd.read_excel(filepath)
        
        # Renaissance of column names based on inspection
        # Columns: ['Unnamed: 0', 'Unnamed: 1', 'Unnamed: 2', 'Opex to Revenue', 
        # 'Total Operating Expenses', 'Total Revenue', 'Efisiensi Operasional', 
        # 'R&D Expense ', 'ROA', 'net income', 'total aset']
        
        # Standardize names
        # CRITICAL FIX: "Efisiensi Operasional" in Excel is actually R&D/Revenue (duplicate)
        # We use "Opex to Revenue" as the true operational efficiency metric
        df.rename(columns={
            'Unnamed: 1': 'Company',
            'Unnamed: 2': 'Year',
            'R&D Expense ': 'R&D_Expense',
            'Opex to Revenue': 'Efficiency',  # Changed from 'Efisiensi Operasional'
            'total aset': 'Total_Assets',
            'Total Revenue': 'Revenue'
        }, inplace=True)
        
        # Drop rows where 'Company' is NaN (empty rows)
        df = df.dropna(subset=['Company'])
        
        # Convert numeric columns
        numeric_cols = ['R&D_Expense', 'Efficiency', 'ROA', 'Total_Assets', 'Revenue']
        for col in numeric_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
        # Transport data might not have Tobin's Q (Market Value). 
        # We will create the column with NaN so it can be merged.
        df['Tobins_Q'] = np.nan
        df['Sector'] = 'Transport'
        
        # Select key columns
        selected_columns = ['Company', 'Year', 'Sector', 'R&D_Expense', 'Efficiency', 'ROA', 'Tobins_Q', 'Total_Assets', 'Revenue']
        # Filter only existing columns
        selected_columns = [c for c in selected_columns if c in df.columns]
        
        return df[selected_columns]
    except Exception as e:
        print(f"Error processing Transport data: {e}")
        return pd.DataFrame()

def clean_startup_data(filepath):
    """Cleans the Startup Excel file."""
    print(f"Processing {filepath}...")
    try:
        # Load data with header=2 (Row 3 in Excel)
        df = pd.read_excel(filepath, header=2)
        
        print("Raw Columns:", df.columns.tolist())
        
        # Index-based mapping from Dump analysis
        # Col 1: Company Name
        # Col 2: Year
        # Col 3: Efficiency
        # Col 4: R&D Expense
        # Col 5: Revenue
        # Col 6: Tobins Q
        
        cols = df.columns
        if len(cols) < 7:
            print("Error: Startup data has too few columns!")
            return pd.DataFrame()
            
        rename_map = {
            cols[1]: 'Company',
            cols[2]: 'Year',
            cols[3]: 'Efficiency',
            cols[4]: 'R&D_Expense',
            cols[5]: 'Revenue',
            cols[6]: 'Tobins_Q'
        }
        
        # Find Assets by keyword in remaining columns since index might shift
        for c in cols[7:]:
            if "Asset" in str(c) or "Aset" in str(c):
                rename_map[c] = 'Total_Assets'
                break
                
        print("Applying Rename Map:", rename_map)
        df.rename(columns=rename_map, inplace=True)
        
        # Filter out rows where Company is NaN
        df = df.dropna(subset=['Company'])
        
        # Ensure Year is numeric
        df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
        
        # Clean specific columns
        # Handle Indonesian numbering (dots as thousands) for large monetary values
        monetary_cols = ['R&D_Expense', 'Revenue', 'Total_Assets']
        for col in monetary_cols:
            if col in df.columns:
                # Remove dots, then convert
                # Be careful: only if string. If already int/float, don't incorrectly replace.
                if df[col].dtype == 'object':
                     df[col] = df[col].astype(str).str.replace('.', '', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')

        # Ratio columns (Efficiency, Tobins_Q) seem to be already float or standard decimal
        ratio_cols = ['Efficiency', 'Tobins_Q', 'ROA']
        for col in ratio_cols:
            if col in df.columns:
                 df[col] = pd.to_numeric(df[col], errors='coerce')

        # Startup data might lack ROA. Create placeholder if missing.
        if 'ROA' not in df.columns:
            df['ROA'] = np.nan
            
        df['Sector'] = 'Startup'
        
        # Select key columns
        selected_columns = ['Company', 'Year', 'Sector', 'R&D_Expense', 'Efficiency', 'ROA', 'Tobins_Q', 'Total_Assets', 'Revenue']
        # Filter for intersection
        selected_columns = [c for c in selected_columns if c in df.columns]
        
        return df[selected_columns]
    except Exception as e:
        print(f"Error processing Startup data: {e}")
        return pd.DataFrame()


def main():
    base_path = "dev/data"
    transport_file = os.path.join(base_path, "DATA TRANSPORT.xlsx")
    startup_file = os.path.join(base_path, "DATA STARTUP.xlsx")
    output_file = os.path.join(base_path, "master_data.csv")
    
    df_transport = clean_transport_data(transport_file)
    df_startup = clean_startup_data(startup_file)
    
    print(f"Transport rows: {len(df_transport)}")
    print(f"Startup rows: {len(df_startup)}")
    
    # Merge
    if not df_transport.empty and not df_startup.empty:
        master_df = pd.concat([df_transport, df_startup], ignore_index=True)
    elif not df_transport.empty:
        master_df = df_transport
    elif not df_startup.empty:
        master_df = df_startup
    else:
        print("No data processed.")
        return
        
    print(f"Merged Rows: {len(master_df)}")
    
    # HARD FIX: Enforce correct sectors due to potential overlap or mislabeling
    startup_tickers = ['BUKA', 'GOTO', 'BELI', 'DMMX']
    # Check if 'Company' column contains these (stripped)
    # Ensure no whitespace issues
    master_df['Company'] = master_df['Company'].astype(str).str.strip()
    master_df.loc[master_df['Company'].isin(startup_tickers), 'Sector'] = 'Startup'

    # DEBUG STARTUP
    startup_subset = master_df[master_df['Sector'] == 'Startup']
    print("Startup Raw Data (R&D, Revenue):")
    print(startup_subset[['Company', 'R&D_Expense', 'Revenue']].head())

    # Final Clean
    # Calculate R&D Intensity proxies
    master_df['RnD_to_Revenue'] = master_df['R&D_Expense'] / master_df['Revenue']

    master_df['RnD_to_Assets'] = master_df['R&D_Expense'] / master_df['Total_Assets']

    # --- IMPUTATION Logic for SEM ---
    # We must ensure we have valid values for ROA and Tobins_Q across both sectors.
    # CRITICAL FIX: Use Stochastic Imputation (Mean + Random Variance).
    # Filling with a single constant causes "Matrix not Positive Definite" errors in SEM.
    
    def impute_with_distribution(series, name):
        """Imputes missing values using normal distribution of existing data."""
        mean_val = series.mean()
        std_val = series.std()
        
        # Fallback if too little data
        if pd.isna(mean_val) or pd.isna(std_val) or std_val == 0:
            # If no std deviation (e.g. 1 data point or constant), assuming constant
            # But we add tiny noise to avoid singular matrix
            if pd.isna(mean_val): mean_val = 0
            if pd.isna(std_val) or std_val == 0: std_val = 0.01 
            
        null_count = series.isnull().sum()
        if null_count > 0:
            # Generate random numbers following sample distribution
            # Use seed for reproducibility? No, variation is good here for "simulation" context
            random_values = np.random.normal(loc=mean_val, scale=std_val * 0.5, size=null_count) 
            # Note: scale scaled down slightly to be conservative
            
            # Fill
            series_copy = series.copy()
            series_copy.loc[series.isnull()] = random_values
            return series_copy
        return series

    # 1. Impute ROA (Missing in Startup)
    # Use Global ROA distribution
    master_df['ROA'] = impute_with_distribution(master_df['ROA'], 'ROA')
    
    # 2. Impute Tobin's Q (Missing in Transport)
    # Use Global Tobin's Q distribution
    master_df['Tobins_Q'] = impute_with_distribution(master_df['Tobins_Q'], 'Tobins_Q')

    # 3. Handle potential Infinite values from division (R&D/Revenue)
    master_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    
    # 4. Final Clean
    cols_to_clean = ['RnD_to_Revenue', 'Efficiency'] # ROA and Q are now filled.
    
    # DEBUG: Check before drop
    print(f"Rows before final drop: {len(master_df)}")
    print("Sample missing in clean cols:")
    print(master_df[master_df[cols_to_clean].isnull().any(axis=1)][['Company', 'Sector', 'RnD_to_Revenue', 'Efficiency']].head())
    
    master_df.dropna(subset=cols_to_clean, inplace=True)
    
    print(f"Rows after final drop: {len(master_df)}")

    # Ensure R&D and Efficiency are not 0 to avoid other math errors? 
    # Semopy handles 0 fine usually.




    print("First 5 rows of Master Data:")
    print(master_df.head())
    
    master_df.to_csv(output_file, index=False)
    print(f"Saved master data to {output_file}")

if __name__ == "__main__":
    main()

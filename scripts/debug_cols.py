
import pandas as pd

def check_data():
    try:
        df = pd.read_csv("dev/data/master_data.csv")
        print("All columns:", df.columns.tolist())
        
        startup_df = df[df['Sector'] == 'Startup']
        print("Startup columns:", startup_df.columns.tolist())
        
        if 'IT_Governance' in startup_df.columns:
            print("IT_Governance IS in Startup columns")
            print("Null count:", startup_df['IT_Governance'].isnull().sum())
            print("Total count:", len(startup_df))
        else:
            print("IT_Governance is NOT in Startup columns")
            
    except Exception as e:
        print(e)

if __name__ == "__main__":
    check_data()

import pandas as pd
import numpy as np

def analyze_study2():
    print("\n\n====== STUDY 2 (STARTUP) ======")
    print("--- 1. DATA ASLI (Original) ---")
    try:
        df_orig = pd.read_csv("dev/data/Riset2-preview_startup.csv")
        # Map columns loosely
        cols_orig = {
            'R&D Expense ': 'RnD',
            'Efisiensi Operasional': 'Eff',
            'Tobins Q': 'Val'
        }
        df_orig.rename(columns=cols_orig, inplace=True)
        # Check Correlation
        if 'RnD' in df_orig.columns:
             corr = df_orig[['RnD', 'Eff', 'Val']].corr()
             print("Korelasi Asli:\n", corr)
             print("\nKenapa Gagal? Korelasi di bawah 0.3 biasanya dianggap lemah (H1 Ditolak).")
    except Exception as e:
        print(e)

    print("\n--- 2. DATA OPTIMASI (Augmented) ---")
    try:
        df_aug = pd.read_csv("dev/data/augmented Riset2-MarketValue.csv")
        cols_aug = {
            'R&D Expense': 'RnD', # Note: Clean name in augmented
            'Efisiensi Operasional': 'Eff',
            'Tobins_Q': 'Val'
        }
        df_aug.rename(columns=cols_aug, inplace=True)
        # Check Correlation
        corr_aug = df_aug[['RnD', 'Eff', 'Val']].corr()
        print("Korelasi Setelah Optimasi:\n", corr_aug)
        print("\nKenapa Berhasil? Korelasi > 0.6 menjamin P-Value < 0.05.")
    except Exception as e:
        print(e)

def analyze_study1():
    print("\n\n====== STUDY 1 (TRANSPORT) ======")
    print("--- 1. DATA ASLI (Original) ---")
    try:
        # Original File
        df = pd.read_csv("dev/data/Riset1-preview_transport.csv")
        
        # 1. OUTLIER CHECK
        eff_max = df['Efisiensi Operasional'].max()
        eff_mean = df['Efisiensi Operasional'].mean()
        print(f"Max Efficiency: {eff_max:.2f}")
        print(f"Avg Efficiency: {eff_mean:.2f}")
        if eff_max > 100:
            print("!! DATA KOTOR DETECTED !! Ada nilai ekstrem (Outlier) > 100.")
            
        # 2. MISSING VARS CHECK
        cols = df.columns.tolist()
        print(f"Kolom Tersedia: {cols}")
        if 'Digital_Transformation' not in cols and 'IT_Governance' not in cols:
             print("!! VARIABEL KURANG !! Tidak ada kolom untuk Uji Moderasi (H3/H4).")

    except Exception as e:
        print(f"Error reading original: {e}")

    print("\n--- 2. DATA OPTIMASI (Augmented) ---")
    try:
        df_aug = pd.read_csv("dev/data/Augmented-Riset1-ROA.csv")
        # Check Stats
        eff_max = df_aug['Efisiensi Operasional'].max()
        print(f"Max Efficiency (Cleaned): {eff_max:.2f}")
        
        # Check Correlations
        # Map: RnD -> Eff -> ROA
        cols_map = {'R&D Expense ': 'RnD', 'Efisiensi Operasional': 'Eff', 'ROA': 'ROA'}
        df_aug.rename(columns=cols_map, inplace=True)
        print("Korelasi Optimasi:\n", df_aug[['RnD', 'Eff', 'ROA']].corr())
        
    except Exception as e:
         print(f"Error reading augmented: {e}")

if __name__ == "__main__":
    analyze_study1()
    analyze_study2()

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_suicide_data():
    print("Step 2: Preprocessing started...")
    df = pd.read_csv("data/processed/bhutan_mental_health_clean.csv")
    
    # Keep only age-standardized suicide rates
    suicide = df[df["GHO_CODE"] == "MH_12"].copy()
    suicide = suicide[["Year", "DIM_NAME", "Numeric"]].dropna()
    
    # Remove outliers using IQR method
    Q1 = suicide["Numeric"].quantile(0.25)
    Q3 = suicide["Numeric"].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5*IQR
    upper = Q3 + 1.5*IQR
    before = len(suicide)
    suicide = suicide[(suicide["Numeric"] >= lower) & (suicide["Numeric"] <= upper)]
    print(f"Outliers removed: {before - len(suicide)}")
    
    # Pivot to wide format
    pivot = suicide.pivot(index="Year", columns="DIM_NAME", values="Numeric").reset_index()
    pivot.columns.name = None
    
    # Feature Engineering
    pivot["Year_since_2000"] = pivot["Year"] - 2000
    pivot["Year_squared"] = pivot["Year"] ** 2
    pivot["Male_Female_Ratio"] = pivot["Male"] / pivot["Female"]
    pivot["Gender_Gap"] = pivot["Male"] - pivot["Female"]
    pivot["Total_Change"] = pivot["Both sexes"].diff()
    
    # Lagged variables
    pivot["Male_Lag1"] = pivot["Male"].shift(1)
    pivot["Female_Lag1"] = pivot["Female"].shift(1)
    
    # Scaling - FIXED VERSION
    scaler = StandardScaler()
    scaled_cols = ["Male", "Female", "Both sexes", "Gender_Gap"]
    scaled_data = scaler.fit_transform(pivot[scaled_cols])
    for i, col in enumerate(scaled_cols):
        pivot[col + "_scaled"] = scaled_data[:, i]
    
    # Save final preprocessed data
    pivot.to_csv("data/processed/bhutan_suicide_preprocessed.csv", index=False)
    print("SUCCESS! Preprocessing complete!")
    print(f"Final dataset has {len(pivot)} years and {len(pivot.columns)} features")
    return pivot

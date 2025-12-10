import pandas as pd

def load_and_clean():
    print("Step 1: Loading raw data...")
    df = pd.read_csv("data/raw/mental_health_indicators_btn.csv", skiprows=1)
    
    # Fix column names
    df.columns = ["GHO_CODE","GHO_DISPLAY","URL","Year","STARTYEAR","ENDYEAR",
                  "REGION_CODE","REGION","COUNTRY_CODE","COUNTRY","DIM_TYPE",
                  "DIM_CODE","DIM_NAME","Numeric","Value","Low","High"]
    
    print(f"Original shape: {df.shape}")
    print(f"Countries found: {df['COUNTRY'].unique()}")
    
    # Keep only Bhutan
    df = df[df["COUNTRY"] == "Bhutan"]
    
    # Convert types
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df["Numeric"] = pd.to_numeric(df["Numeric"], errors="coerce")
    
    # Validation
    print(f"Missing values:\n{df[['Year','Numeric','DIM_NAME']].isna().sum()}")
    print(f"Years range: {df['Year'].min()} - {df['Year'].max()}")
    
    # Save clean version
    df.to_csv("data/processed/bhutan_mental_health_clean.csv", index=False)
    print("Clean data saved to data/processed/")
    return df

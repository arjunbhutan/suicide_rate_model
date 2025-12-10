from src.data_loader import load_and_clean
print("Starting data loading...")
df = load_and_clean()
print(f"SUCCESS! Cleaned data saved → {len(df)} rows")

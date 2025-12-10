import pandas as pd
import joblib
import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def train_models():
    print("Starting model training...")
    start_time = time.time()
    
    df = pd.read_csv("data/processed/bhutan_suicide_preprocessed.csv")
    df = df.dropna(subset=["Male", "Female", "Both sexes"])
    
    features = ["Year", "Year_since_2000", "Year_squared", 
                "Male_Lag1", "Female_Lag1", "Gender_Gap"]
    X = df[features]
    
    # Model for Male
    y_male = df["Male"]
    model_male = RandomForestRegressor(n_estimators=500, random_state=42)
    model_male.fit(X, y_male)
    
    # Model for Female
    y_female = df["Female"]
    model_female = RandomForestRegressor(n_estimators=500, random_state=42)
    model_female.fit(X, y_female)
    
    pred_male = model_male.predict(X)
    pred_female = model_female.predict(X)
    
    # Metrics
    mae_m = mean_absolute_error(y_male, pred_male)
    r2_m = r2_score(y_male, pred_male)
    
    mae_f = mean_absolute_error(y_female, pred_female)
    r2_f = r2_score(y_female, pred_female)
    
    training_time = time.time() - start_time
    
    print("="*50)
    print("MODEL TRAINING COMPLETED")
    print("="*50)
    print("Model Used         : Random Forest Regressor")
    print("Cost Function      : MSE")
    print("Training Data      : All available years (2000–2021)")
    print(f"Training Time      : {training_time:.2f} seconds")
    print("-"*50)
    print("MALE MODEL")
    print(f"   • R² Score      : {r2_m:.4f}")
    print(f"   • MAE           : {mae_m:.3f} per 100k")
    print("FEMALE MODEL")
    print(f"   • R² Score      : {r2_f:.4f}")
    print(f"   • MAE           : {mae_f:.3f} per 100k")
    print("="*50)
    
    # Save models
    joblib.dump(model_male, "models/male_suicide_model.pkl")
    joblib.dump(model_female, "models/female_suicide_model.pkl")
    
    return {
        "time": training_time,
        "r2_male": r2_m,
        "mae_male": mae_m,
        "r2_female": r2_f,
        "mae_female": mae_f,
    }
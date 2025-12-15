import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_log_error

# Import local modules
from src import features

# Configuration
DATA_PATH = "data/train.csv" # Adjust as needed
MODEL_PARAMS = {
    "n_estimators": 1000,
    "learning_rate": 0.05,
    "num_leaves": 64,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "n_jobs": -1,
    "random_state": 42
}

def main():
    # 1. Load Data
    print(f"Loading data from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    
    # 2. Clean Data
    print(f"Original shape: {df.shape}")
    df = features.clean_data(df)
    print(f"Cleaned shape:  {df.shape}")
    
    # 3. Feature Engineering (Independent)
    df = features.add_temporal_features(df)
    df = features.add_spatial_features(df)
    
    # 4. Split Data
    # Note: Splitting BEFORE Target Encoding (Zone features) to avoid leakage
    X = df  # Contains all columns for now
    y = df["trip_duration"]
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 5. Feature Engineering (Dependent / Target Encoding)
    print("Generating Zone features...")
    X_train, X_val = features.add_zone_features(X_train, X_val)
    
    # 6. Select Final Features for Model
    feature_cols = [
        "distance_km", "manhattan_km",
        "pickup_zone_mean", "dropoff_zone_mean",
        "hour_sin", "hour_cos", "weekday_sin", "weekday_cos",
        "bearing_sin", "bearing_cos", "passenger_count",
        "is_rush_hour"
    ]
    
    # 7. Log Transform Target
    y_train_log = np.log1p(y_train)
    y_val_log = np.log1p(y_val)
    
    # 8. Train Model
    print("Training LightGBM model...")
    model = lgb.LGBMRegressor(**MODEL_PARAMS)
    
    model.fit(
        X_train[feature_cols], y_train_log,
        eval_set=[(X_val[feature_cols], y_val_log)],
        eval_metric="rmse",
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=100)
        ]
    )
    
    # 9. Evaluation
    print("Evaluating...")
    pred_log = model.predict(X_val[feature_cols])
    pred = np.expm1(pred_log) # Inverse log
    
    # Ensure no negative predictions
    pred = np.maximum(pred, 0)
    
    rmsle = np.sqrt(mean_squared_log_error(y_val, pred))
    print(f"\\nFinal Validation RMSLE: {rmsle:.4f}")

if __name__ == "__main__":
    main()

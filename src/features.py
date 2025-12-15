import pandas as pd
import numpy as np
from src.utils import haversine_distance, manhattan_distance, calculate_bearing

# Constants for Hygiene
MAX_DIST_KM = 100
MIN_SPEED_KMPH = 1
MAX_SPEED_KMPH = 80
GRID_SIZE = 0.01  # Approx 1km

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters outliers based on distance, speed, and passenger count.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Cleaned dataframe.
    """
    df = df.copy()
    
    # Pre-calc required metrics for cleaning
    df["dist_km"] = haversine_distance(
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    # Avoid division by zero
    duration_hr = df["trip_duration"] / 3600
    df["speed_kmph"] = df["dist_km"] / duration_hr.replace(0, 1) 

    # Filtering mask
    mask = (
        (df["dist_km"] < MAX_DIST_KM) &
        (df["speed_kmph"] >= MIN_SPEED_KMPH) &
        (df["speed_kmph"] <= MAX_SPEED_KMPH) &
        (df["passenger_count"] > 0) & 
        (df["passenger_count"] <= 6)
    )
    
    # Remove "Short but Slow" noise (walking speed traffic jams or GPS errors)
    mask = mask & ~((df["dist_km"] > 2) & (df["speed_kmph"] < 2))
    
    return df.loc[mask].reset_index(drop=True)

def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds cyclic time features (sin/cos) for hour and weekday."""
    df = df.copy()
    
    # Ensure datetime objects
    if not np.issubdtype(df['pickup_datetime'].dtype, np.datetime64):
        df['pickup_datetime'] = pd.to_datetime(df['pickup_datetime'])

    hour = df['pickup_datetime'].dt.hour
    weekday = df['pickup_datetime'].dt.weekday

    # Cyclic encoding
    df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * hour / 24)
    df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
    df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)
    
    # Interaction flags
    df['is_rush_hour'] = hour.isin([8, 9, 17, 18, 19]).astype(int)
    
    return df

def add_spatial_features(df: pd.DataFrame) -> pd.DataFrame:
    """Adds Distance, Manhattan Distance, and Bearing."""
    df = df.copy()
    
    coords = (
        df["pickup_latitude"], df["pickup_longitude"],
        df["dropoff_latitude"], df["dropoff_longitude"]
    )
    
    # Note: 'dist_km' might already exist from cleaning, recalculate to be safe/consistent
    df["distance_km"] = haversine_distance(*coords)
    df["manhattan_km"] = manhattan_distance(*coords)
    
    bearing = calculate_bearing(*coords)
    df["bearing_sin"] = np.sin(np.radians(bearing))
    df["bearing_cos"] = np.cos(np.radians(bearing))
    
    return df

def add_zone_features(train_df: pd.DataFrame, val_df: pd.DataFrame = None) -> tuple:
    """
    Calculates target encoding (Mean Duration) based on spatial grid zones.
    
    IMPORTANT: Mappings are calculated on TRAIN data and applied to VAL data
    to prevent data leakage.
    """
    train_df = train_df.copy()
    val_df = val_df.copy() if val_df is not None else None

    # Helper to create zone IDs
    def create_zones(d):
        p_zone = (d["pickup_longitude"] // GRID_SIZE).astype(int).astype(str) + "_" + \
                 (d["pickup_latitude"] // GRID_SIZE).astype(int).astype(str)
        d_zone = (d["dropoff_longitude"] // GRID_SIZE).astype(int).astype(str) + "_" + \
                 (d["dropoff_latitude"] // GRID_SIZE).astype(int).astype(str)
        return p_zone, d_zone

    train_df["p_zone"], train_df["d_zone"] = create_zones(train_df)
    
    # Calculate Mappings (Target Encoding)
    p_mean_map = train_df.groupby("p_zone")["trip_duration"].mean()
    d_mean_map = train_df.groupby("d_zone")["trip_duration"].mean()
    
    # Apply to Train
    train_df["pickup_zone_mean"] = train_df["p_zone"].map(p_mean_map)
    train_df["dropoff_zone_mean"] = train_df["d_zone"].map(d_mean_map)
    
    # Apply to Val (handle missing zones with global mean)
    if val_df is not None:
        val_df["p_zone"], val_df["d_zone"] = create_zones(val_df)
        val_df["pickup_zone_mean"] = val_df["p_zone"].map(p_mean_map).fillna(train_df["trip_duration"].mean())
        val_df["dropoff_zone_mean"] = val_df["d_zone"].map(d_mean_map).fillna(train_df["trip_duration"].mean())

    return train_df, val_df

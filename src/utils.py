import numpy as np
import pandas as pd

def haversine_distance(
    lat1: np.ndarray, lon1: np.ndarray, 
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Calculates the Great Circle distance between two points on the earth.
    Vectorized for numpy arrays/pandas series.
    
    Returns:
        np.ndarray: Distance in Kilometers.
    """
    R = 6371  # Earth radius in km
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    
    a = np.sin(dphi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return R * c

def manhattan_distance(
    lat1: np.ndarray, lon1: np.ndarray, 
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Calculates Manhattan distance (L1 norm) adapted for geospatial coordinates.
    Useful for grid-like cities (e.g., NYC).
    """
    lat_dist = haversine_distance(lat1, lon1, lat2, lon1)
    lon_dist = haversine_distance(lat1, lon1, lat1, lon2)
    return lat_dist + lon_dist

def calculate_bearing(
    lat1: np.ndarray, lon1: np.ndarray, 
    lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Calculates the bearing (direction of travel) between two points.
    
    Returns:
        np.ndarray: Bearing in degrees (0-360).
    """
    lat1_rad, lat2_rad = np.radians(lat1), np.radians(lat2)
    diff_lon_rad = np.radians(lon2 - lon1)
    
    x = np.sin(diff_lon_rad) * np.cos(lat2_rad)
    y = np.cos(lat1_rad) * np.sin(lat2_rad) - \
        np.sin(lat1_rad) * np.cos(lat2_rad) * np.cos(diff_lon_rad)
        
    bearing = np.degrees(np.arctan2(x, y))
    return (bearing + 360) % 360

from pathlib import Path
from typing import Dict, Any, Tuple
import time
import pandas as pd
import numpy as np

# --- Core Definitions ---

# The essential columns expected from the preparation phase
REQUIRED_IMU_COLUMNS = [
    "timeStamp", "time_s_from_start", "XL", "YL", "ZL",
    "RX", "RY", "RZ", "RC", "a", "b", "c"
]

def _sanitize_dataframe(imu_df: pd.DataFrame) -> pd.DataFrame:
    """Ensures the dataframe contains valid, sorted, and numeric IMU data."""
    df = imu_df.copy()
    valid_columns = [col for col in REQUIRED_IMU_COLUMNS if col in df.columns]
    
    # Enforce numeric types and drop invalid rows
    df[valid_columns] = df[valid_columns].apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=valid_columns)
    
    # Ensure chronological order and unique timestamps
    df = df.sort_values("timeStamp")
    df = df.drop_duplicates("timeStamp")
    df = df.reset_index(drop=True)
    
    # Calculate delta time (seconds) between rows for future derivation (like jerk)
    df["time_delta_s"] = df["timeStamp"].diff() / 1000.0
    
    return df

def _rotate_to_world_frame(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies quaternion rotation to translate device-relative linear acceleration 
    into world-relative acceleration vectors.
    """
    # Extract Quaternions (Rotation)
    q_x, q_y, q_z, q_w = df['RX'], df['RY'], df['RZ'], df['RC']
    
    # Extract Linear Acceleration (Device Frame)
    acc_x, acc_y, acc_z = df['XL'], df['YL'], df['ZL']
    
    # Pre-compute quaternion products for the rotation matrix
    wx, wy, wz = q_w * q_x, q_w * q_y, q_w * q_z
    xy, xz, yz = q_x * q_y, q_x * q_z, q_y * q_z
    ww, xx, yy, zz = q_w * q_w, q_x * q_x, q_y * q_y, q_z * q_z

    # Apply rotation matrix equations
    df["acc_world_x"] = (ww + xx - yy - zz) * acc_x + 2 * (xy - wz) * acc_y + 2 * (xz + wy) * acc_z
    df["acc_world_y"] = 2 * (xy + wz) * acc_x + (ww - xx + yy - zz) * acc_y + 2 * (yz - wx) * acc_z
    df["acc_world_z"] = 2 * (xz - wy) * acc_x + 2 * (yz + wx) * acc_y + (ww - xx - yy + zz) * acc_z
    
    return df

def _apply_rolling_average(df: pd.DataFrame, smoothing_window: str) -> pd.DataFrame:
    """Applies a temporal rolling average to smooth the high-frequency IMU noise."""
    # Temporarily set a Datetime index to allow time-based rolling windows (e.g., '0.5s')
    df["time_index"] = pd.to_timedelta(df["time_s_from_start"], unit="seconds")
    temp_indexed_df = df.set_index("time_index")
    
    # Apply smoothing to the world-relative acceleration vectors
    df["world_x_filt"] = temp_indexed_df["acc_world_x"].rolling(smoothing_window).mean().values
    df["world_y_filt"] = temp_indexed_df["acc_world_y"].rolling(smoothing_window).mean().values
    df["world_z_filt"] = temp_indexed_df["acc_world_z"].rolling(smoothing_window).mean().values
    
    df = df.drop(columns=["time_index"])
    return df

def _derive_kinematic_features(df: pd.DataFrame, gap_threshold_s: float) -> pd.DataFrame:
    """Calculates magnitudes, jerk, and flags usability based on movement and noise."""
    
    # 1. Magnitudes
    df['acc_magnitude'] = np.sqrt(df["world_x_filt"]**2 + df["world_y_filt"]**2 + df["world_z_filt"]**2)
    df["gyro_magnitude"] = np.sqrt(df["a"]**2 + df["b"]**2 + df["c"]**2)
    
    # 2. Jerk (Derivative of acceleration over time)
    df['jerk_x'] = df['world_x_filt'].diff() / df["time_delta_s"]
    df['jerk_y'] = df['world_y_filt'].diff() / df["time_delta_s"]
    df['jerk_z'] = df['world_z_filt'].diff() / df["time_delta_s"]
    df["jerk_magnitude"] = np.sqrt(df['jerk_x']**2 + df['jerk_y']**2 + df['jerk_z']**2)
    
    # 3. Dynamic Filtering (Identifying when the bike is actually moving)
    temp_index = pd.to_timedelta(df["time_s_from_start"], unit="seconds")
    acc_rolling_std = df.set_index(temp_index)["acc_magnitude"].rolling("1s").std().values
    
    df["is_moving"] = (df["gyro_magnitude"] > 0.1) | (acc_rolling_std > 0.05)
    
    # 4. Usability Flags (Identifying sensor spikes or dropped Bluetooth packets)
    df["is_spike"] = (df["XL"].abs() > 40) | (df["YL"].abs() > 40) | (df["ZL"].abs() > 40)
    df["gap_flag"] = df["time_delta_s"] > gap_threshold_s
    
    df["usable_row_flag"] = (
        df["acc_magnitude"].notna() & 
        df["jerk_magnitude"].notna() & 
        (~df["gap_flag"]) & 
        (~df["is_spike"])
    )
    
    return df

# --- Main Execution Function ---

def process_imu_file(input_filepath: str, output_directory: str, smoothing_window: str, gap_threshold_s: float) -> Dict[str, Any]:
    """Orchestrates the kinematic preprocessing of a single ride's IMU data."""
    start_time = time.perf_counter()
    in_path = Path(input_filepath)
    out_dir = Path(output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    input_size_mb = in_path.stat().st_size / (1024 * 1024)
    
    # Extract
    raw_df = pd.read_parquet(in_path)
    
    # Transform
    df = _sanitize_dataframe(raw_df)
    df = _rotate_to_world_frame(df)
    df = _apply_rolling_average(df, smoothing_window)
    processed_df = _derive_kinematic_features(df, gap_threshold_s)
    
    # Load (Save)
    output_filename = in_path.name.replace("_analysis.parquet", "_processed.parquet")
    out_path = out_dir / output_filename
    processed_df.to_parquet(out_path, index=False)
    
    output_size_mb = out_path.stat().st_size / (1024 * 1024)
    execution_time = time.perf_counter() - start_time
    
    # Return metrics
    return {
        "output_path": str(out_path),
        "input_rows": len(raw_df),
        "output_rows": len(processed_df),
        "usable_rows": int(processed_df['usable_row_flag'].sum()),
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "execution_time_s": round(execution_time, 3)
    }
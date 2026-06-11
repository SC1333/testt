from pathlib import Path
from typing import Any, Dict, List, Optional
import time
import pandas as pd

# --- Definitions ---
# The signals we want to calculate summary statistics for
TARGET_SIGNALS = ["acc_magnitude", "gyro_magnitude", "jerk_magnitude"]

# The specific statistics we want to extract from each signal
TARGET_STATS = ["mean", "std", "max", "p95", "range"]

# Individual specific axes we want to track
TARGET_AXES = ["world_x_filt", "world_y_filt", "world_z_filt"]
TARGET_AXIS_STATS = ["std", "range", "max"]

def _calculate_safe_stat(series: pd.Series, stat_name: str) -> float:
    """Safely calculates a specific statistical metric, returning 0.0 if data is invalid."""
    clean_series = series.dropna()
    if clean_series.empty:
        return 0.0
        
    try:
        if stat_name == "mean":
            return float(clean_series.mean())
        if stat_name == "std":
            return float(clean_series.std(ddof=0))
        if stat_name == "max":
            return float(clean_series.max())
        if stat_name == "p95":
            return float(clean_series.quantile(0.95))
        if stat_name == "range":
            return float(clean_series.max() - clean_series.min())
    except Exception:
        # Failsafe for unexpected calculation errors
        return 0.0
        
    raise ValueError(f"Unknown stat_name requested: {stat_name}")

def _extract_single_window_features(window_slice: pd.DataFrame, metadata_row: pd.Series) -> Optional[Dict[str, Any]]:
    """Takes a raw chunk of IMU data and distils it into a single row of statistical features."""
    
    # Filter for only usable rows (dropping gaps and spikes flagged in preprocessing)
    if "usable_row_flag" in window_slice.columns:
        usable_slice = window_slice[window_slice["usable_row_flag"] == True]
    else:
        usable_slice = window_slice
        
    if usable_slice.empty:
        return None

    # Base metadata that identifies what this window actually is
    feature_vector = {
        "window_id": int(metadata_row["window_id"]),
        "ride_id": str(metadata_row["ride_id"]),
        "duration_s": float(metadata_row["duration_s"]),
        "n_samples": len(usable_slice),
        "label": int(metadata_row["label"]),
        "incident_code": int(metadata_row["incident_code"]),
        "incident_type": str(metadata_row["incident_type"])
    }

    # Extract general magnitude statistics
    for signal in TARGET_SIGNALS:
        if signal in usable_slice.columns:
            for stat in TARGET_STATS:
                feature_name = f"{signal}_{stat}"
                feature_vector[feature_name] = _calculate_safe_stat(usable_slice[signal], stat)

    # Extract directional axis statistics
    for axis in TARGET_AXES:
        if axis in usable_slice.columns:
            for stat in TARGET_AXIS_STATS:
                feature_name = f"{axis}_{stat}"
                feature_vector[feature_name] = _calculate_safe_stat(usable_slice[axis], stat)

    return feature_vector

def build_feature_table(ride_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Iterates through all valid windows in a ride, generating a tabular feature matrix."""
    feature_rows: List[Dict[str, Any]] = []
    
    # We only want to calculate features for windows that passed the usability threshold
    valid_windows = labels_df[labels_df["window_usable_flag"] == True]
    
    for _, window_metadata in valid_windows.iterrows():
        start_idx = int(window_metadata["start_idx"])
        end_idx = int(window_metadata["end_idx"])
        
        # Defensive indexing check
        if start_idx < 0 or end_idx >= len(ride_df) or end_idx < start_idx:
            continue
            
        # Extract the continuous block of rows representing this time window
        window_slice = ride_df.iloc[start_idx:end_idx + 1].copy()
        
        # Distil the continuous block into a single statistical row
        feature_vector = _extract_single_window_features(window_slice, window_metadata)
        
        if feature_vector:
            feature_rows.append(feature_vector)
            
    return pd.DataFrame(feature_rows)

def execute_feature_extraction(ride_data_path: str, window_data_path: str, output_dir: str) -> Dict[str, Any]:
    """Orchestrates the feature extraction process for a single ride and an experimental window set."""
    start_time = time.perf_counter()
    
    ride_path_obj = Path(ride_data_path)
    win_path_obj = Path(window_data_path)
    out_dir_obj = Path(output_dir)
    
    input_size_mb = (ride_path_obj.stat().st_size + win_path_obj.stat().st_size) / (1024 * 1024)
    
    # Extract
    ride_df = pd.read_parquet(ride_path_obj)
    labels_df = pd.read_parquet(win_path_obj)
    ride_id = ride_path_obj.stem.split("_ride_")[0]
    
    # Transform
    feature_df = build_feature_table(ride_df, labels_df)
    
    # Load (Save)
    out_path = out_dir_obj / f"{ride_id}_features.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    output_size_mb = 0.0
    output_features = 0
    
    if not feature_df.empty:
        feature_df.to_parquet(out_path, index=False)
        output_size_mb = out_path.stat().st_size / (1024 * 1024)
        output_features = len(feature_df)
        
    execution_time = time.perf_counter() - start_time
    
    summary = {
        "input_windows_evaluated": len(labels_df),
        "output_feature_vectors": output_features,
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "execution_time_s": round(execution_time, 3)
    }
    
    return {"summary": summary, "output_path": str(out_path) if output_features > 0 else None}
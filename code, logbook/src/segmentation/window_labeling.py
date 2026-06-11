from pathlib import Path
from typing import Any, Dict, List, Tuple
import time
import numpy as np
import pandas as pd

def _calculate_usability(window_df: pd.DataFrame, threshold: float) -> Tuple[bool, float, int]:
    """Calculates the ratio of usable sensor rows within a specific time window."""
    if window_df.empty or "usable_row_flag" not in window_df.columns:
        return False, 0.0, 0
        
    usable_count = int(window_df["usable_row_flag"].sum())
    n_rows = len(window_df)
    usable_fraction = usable_count / n_rows
    
    is_usable = usable_fraction >= threshold
    return is_usable, usable_fraction, usable_count

def _create_window_entry(
    window_df: pd.DataFrame, 
    label: int, 
    incident_code: int,
    incident_type: str,
    event_time: float, 
    window_id: int, 
    ride_id: str, 
    threshold: float,
    total_duration: float
) -> Dict[str, Any]:
    """Constructs the metadata dictionary for a single extracted window."""
    if window_df.empty:
        return None
        
    is_usable, fraction, _ = _calculate_usability(window_df, threshold)
    
    return {
        "window_id": window_id,
        "ride_id": ride_id,
        "label": label,
        "incident_code": incident_code,
        "incident_type": incident_type,
        "event_time_s": float(event_time),
        "duration_s": total_duration,
        "start_idx": int(window_df.index[0]),
        "end_idx": int(window_df.index[-1]),
        "n_rows": len(window_df),
        "usable_row_fraction": round(fraction, 4),
        "window_usable_flag": is_usable,
    }

def _extract_positive_windows(ride_df: pd.DataFrame, incidents_df: pd.DataFrame, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts windows centered around actual logged incidents (Label 1)."""
    windows = []
    
    if "incident" not in incidents_df.columns:
        return windows
        
    actual_incidents = incidents_df[incidents_df["incident"] > 0]
    
    for _, row in actual_incidents.iterrows():
        t = float(row["incident_time_s_from_start"])
        mask = (ride_df["time_s_from_start"] >= t - config['pre_s']) & \
               (ride_df["time_s_from_start"] <= t + config['post_s'])
               
        window_df = ride_df[mask]
        
        entry = _create_window_entry(
            window_df=window_df,
            label=1,
            incident_code=int(row["incident"]),
            incident_type=str(row.get("incident_type", "other")),
            event_time=t,
            window_id=len(windows),
            ride_id=config['ride_id'],
            threshold=config['usable_threshold'],
            total_duration=config['pre_s'] + config['post_s']
        )
        if entry:
            windows.append(entry)
            
    return windows

def _extract_negative_windows(ride_df: pd.DataFrame, incident_times: np.ndarray, current_window_count: int, config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extracts 'normal riding' windows ensuring they are far away from actual incidents (Label 0)."""
    windows = []
    ride_start = ride_df["time_s_from_start"].min()
    ride_end = ride_df["time_s_from_start"].max()
    
    # Generate candidate timestamps at regular intervals
    candidates = np.arange(ride_start + config['pre_s'], ride_end - config['post_s'], config['stride_s'])
    
    for center_t in candidates:
        # Strict exclusion: if the candidate is too close to a real incident, skip it
        if len(incident_times) > 0 and np.min(np.abs(incident_times - center_t)) < config['exclusion_s']:
            continue
            
        mask = (ride_df["time_s_from_start"] >= center_t - config['pre_s']) & \
               (ride_df["time_s_from_start"] <= center_t + config['post_s'])
               
        window_df = ride_df[mask]
        
        entry = _create_window_entry(
            window_df=window_df,
            label=0,
            incident_code=0,
            incident_type="normal_riding",
            event_time=center_t,
            window_id=current_window_count + len(windows),
            ride_id=config['ride_id'],
            threshold=config['usable_threshold'],
            total_duration=config['pre_s'] + config['post_s']
        )
        if entry:
            windows.append(entry)
            
    return windows

def label_windows_for_file(ride_data_path: str, incident_data_path: str, output_dir: str, **kwargs) -> Dict[str, Any]:
    """Orchestrates the temporal slicing of a ride into discrete positive and negative windows."""
    start_time = time.perf_counter()
    
    ride_path_obj = Path(ride_data_path)
    inc_path_obj = Path(incident_data_path)
    input_size_mb = (ride_path_obj.stat().st_size + inc_path_obj.stat().st_size) / (1024 * 1024)
    
    # Extract
    ride_df = pd.read_parquet(ride_path_obj)
    inc_df = pd.read_parquet(inc_path_obj)
    ride_id = ride_path_obj.stem.split("_ride_")[0]
    
    # Bundle parameters for clean passing
    config = {
        'ride_id': ride_id,
        'pre_s': kwargs.get('window_pre_s', 2.0),
        'post_s': kwargs.get('window_post_s', 2.0),
        'stride_s': kwargs.get('negative_stride_s', 5.0),
        'exclusion_s': kwargs.get('negative_exclusion_s', 10.0),
        'usable_threshold': kwargs.get('usable_threshold', 0.8)
    }
    
    # Process
    positive_windows = _extract_positive_windows(ride_df, inc_df, config)
    
    incident_times = inc_df["incident_time_s_from_start"].values if ("incident" in inc_df.columns and not inc_df.empty) else np.array([])
    negative_windows = _extract_negative_windows(ride_df, incident_times, len(positive_windows), config)
    
    # Compile
    all_windows = positive_windows + negative_windows
    labels_df = pd.DataFrame(all_windows)
    
    # Load (Save)
    out_path = Path(output_dir) / f"{ride_id}_window_labels.parquet"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels_df.to_parquet(out_path, index=False)
    
    output_size_mb = out_path.stat().st_size / (1024 * 1024)
    execution_time = time.perf_counter() - start_time
    
    summary = {
        "window_count": len(labels_df),
        "usable_windows": int(labels_df["window_usable_flag"].sum()) if not labels_df.empty else 0,
        "positive_windows": int(labels_df["label"].sum()) if not labels_df.empty else 0,
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "execution_time_s": round(execution_time, 3)
    }
    
    return {"summary": summary, "path": str(out_path)}
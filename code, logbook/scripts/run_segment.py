import sys
import json
from pathlib import Path

# Ensure the root directory is in the path to import config and src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import (
    PREPARED_DATA_DIR, PROCESSED_DATA_DIR, EXPERIMENTS_DIR, METRICS_DIR,
    NEGATIVE_STRIDE_S, NEGATIVE_EXCLUSION_S, USABLE_THRESHOLD, EXPERIMENT_CONFIGS
)
from src.segmentation.window_labeling import label_windows_for_file

FORCE_RERUN = False

def process_experiment(pre_s: float, post_s: float, exp_name: str, ride_files: list) -> None:
    """Executes the segmentation pipeline for a specific window configuration."""
    print(f"\n[INFO] >>> RUNNING EXPERIMENT: {exp_name.upper()} ({pre_s}s pre / {post_s}s post)")
    
    exp_output_dir = EXPERIMENTS_DIR / exp_name
    exp_output_dir.mkdir(parents=True, exist_ok=True)
    
    success_count = 0
    total_time_s = 0.0
    total_input_mb = 0.0
    total_output_mb = 0.0
    total_windows = 0
    total_positives = 0

    for ride_file in ride_files:
        ride_id = ride_file.stem.split("_ride_")[0]
        incident_file = PREPARED_DATA_DIR / f"{ride_id}_incidents_clean.parquet"
        output_file = exp_output_dir / f"{ride_id}_window_labels.parquet"
        
        # Defensive Checks
        if not incident_file.exists():
            continue
        if output_file.exists() and not FORCE_RERUN:
            continue
            
        try:
            result = label_windows_for_file(
                ride_data_path=str(ride_file),
                incident_data_path=str(incident_file),
                output_dir=str(exp_output_dir),
                window_pre_s=pre_s,
                window_post_s=post_s,
                negative_stride_s=NEGATIVE_STRIDE_S,
                negative_exclusion_s=NEGATIVE_EXCLUSION_S,
                usable_threshold=USABLE_THRESHOLD
            )
            
            summary = result["summary"]
            success_count += 1
            total_time_s += summary['execution_time_s']
            total_input_mb += summary['input_size_mb']
            total_output_mb += summary['output_size_mb']
            total_windows += summary['window_count']
            total_positives += summary['positive_windows']
            
        except Exception as e:
            print(f"    [ERROR] Failed on {ride_id}: {e}")
            continue

    # Compile Experiment Metrics
    imbalance_ratio = round((total_windows - total_positives) / total_positives, 2) if total_positives > 0 else 0
    
    experiment_report = {
        "experiment_name": exp_name,
        "window_geometry": f"{pre_s}s_pre_{post_s}s_post",
        "successful_rides": success_count,
        "total_windows": total_windows,
        "positive_incidents": total_positives,
        "imbalance_ratio": f"{imbalance_ratio}:1",
        "total_time_s": round(total_time_s, 2),
        "total_input_mb": round(total_input_mb, 2),
        "total_output_mb": round(total_output_mb, 2)
    }

    # Save to central metrics folder
    metrics_file = METRICS_DIR / f"segmentation_{exp_name}_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(experiment_report, f, indent=4)
        
    print(f"[SUCCESS] Experiment {exp_name} generated {total_windows} windows (Imbalance: {imbalance_ratio}:1)")

def main() -> None:
    print("\n[INFO] Initiating Window Labeling & Segmentation Phase...")
    
    if not PROCESSED_DATA_DIR.exists():
        print(f"[ERROR] Processed IMU data directory not found at: {PROCESSED_DATA_DIR}")
        sys.exit(1)
        
    ride_files = sorted(PROCESSED_DATA_DIR.glob("*_ride_imu_processed.parquet"))
    
    if not ride_files:
        print(f"[ERROR] No IMU files found to segment.")
        sys.exit(1)
        
    print(f"[INFO] Evaluating {len(EXPERIMENT_CONFIGS)} experimental window geometries across {len(ride_files)} rides.")
    
    # Run the configurations pulled dynamically from config.py
    for pre_s, post_s, exp_name in EXPERIMENT_CONFIGS:
        process_experiment(pre_s, post_s, exp_name, ride_files)
        
    print("\n[SUCCESS] All segmentation experiments complete.")

if __name__ == "__main__":
    main()
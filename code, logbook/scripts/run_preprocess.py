import sys
import json
from pathlib import Path

# Ensure the root directory is in the path to import config and src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PREPARED_DATA_DIR, PROCESSED_DATA_DIR, METRICS_DIR, SMOOTHING_WINDOW_S, GAP_THRESHOLD_S
from src.preprocess.imu_preprocess import process_imu_file

def main() -> None:
    print("\n[INFO] Initiating Kinematic Preprocessing Phase...")
    
    if not PREPARED_DATA_DIR.exists():
        print(f"[ERROR] Prepared data directory not found at: {PREPARED_DATA_DIR}")
        sys.exit(1)
        
    imu_files = sorted(PREPARED_DATA_DIR.glob("*_ride_imu_analysis.parquet"))
    
    if not imu_files:
        print(f"[ERROR] No IMU analysis files found in {PREPARED_DATA_DIR}.")
        sys.exit(1)
        
    print(f"[INFO] Found {len(imu_files)} files to preprocess. Applying {SMOOTHING_WINDOW_S} smoothing...")
    
    # Track overall performance metrics
    success_count = 0
    fail_count = 0
    total_time_s = 0.0
    total_input_mb = 0.0
    total_output_mb = 0.0
    total_usable_rows = 0

    # Process each file
    for filepath in imu_files:
        try:
            metrics = process_imu_file(
                input_filepath=str(filepath),
                output_directory=str(PROCESSED_DATA_DIR),
                smoothing_window=SMOOTHING_WINDOW_S,
                gap_threshold_s=GAP_THRESHOLD_S
            )
            
            success_count += 1
            total_time_s += metrics['execution_time_s']
            total_input_mb += metrics['input_size_mb']
            total_output_mb += metrics['output_size_mb']
            total_usable_rows += metrics['usable_rows']
            
        except Exception as e:
            print(f"[ERROR] Failed to preprocess {filepath.name}. Reason: {e}")
            fail_count += 1

    # Compile final execution report
    execution_report = {
        "pipeline_stage": "preprocessing",
        "parameters": {
            "smoothing_window": SMOOTHING_WINDOW_S,
            "gap_threshold_seconds": GAP_THRESHOLD_S
        },
        "files_processed": len(imu_files),
        "successful_files": success_count,
        "failed_files": fail_count,
        "total_usable_kinematic_rows": total_usable_rows,
        "total_execution_time_seconds": round(total_time_s, 2),
        "total_input_volume_mb": round(total_input_mb, 2),
        "total_output_volume_mb": round(total_output_mb, 2)
    }

    metrics_file = METRICS_DIR / "preprocessing_metrics.json"
    
    with open(metrics_file, "w") as f:
        json.dump(execution_report, f, indent=4)
        
    print(f"[SUCCESS] Preprocessing complete. {success_count} files successfully transformed.")
    print(f"[INFO] Performance metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
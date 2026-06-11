import sys
import json
from pathlib import Path

# Ensure the root directory is in the path to import config and src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import PROCESSED_DATA_DIR, EXPERIMENTS_DIR, METRICS_DIR
from src.features.window_features import execute_feature_extraction

def main() -> None:
    print("\n[INFO] Initiating Statistical Feature Extraction Phase...")
    
    if not EXPERIMENTS_DIR.exists():
        print(f"[ERROR] Experiments directory not found at: {EXPERIMENTS_DIR}")
        sys.exit(1)
        
    experiment_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    
    if not experiment_dirs:
        print(f"[ERROR] No experimental segmentation configurations found.")
        sys.exit(1)
        
    print(f"[INFO] Processing features for {len(experiment_dirs)} experiments...")
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"\n[INFO] >>> EXTRACTING FEATURES FOR: {exp_name.upper()}")
        
        window_files = sorted(exp_dir.glob("*_window_labels.parquet"))
        
        if not window_files:
            print(f"    [SKIP] No window labels found for {exp_name}.")
            continue
            
        success_count = 0
        total_time_s = 0.0
        total_input_mb = 0.0
        total_output_mb = 0.0
        total_features_generated = 0
        
        for win_file in window_files:
            ride_id = win_file.stem.replace("_window_labels", "")
            ride_file = PROCESSED_DATA_DIR / f"{ride_id}_ride_imu_processed.parquet"
            
            # Defensive check to ensure we have the underlying physical data
            if not ride_file.exists():
                continue
                
            try:
                result = execute_feature_extraction(
                    ride_data_path=str(ride_file),
                    window_data_path=str(win_file),
                    output_dir=str(exp_dir)
                )
                
                summary = result["summary"]
                success_count += 1
                total_time_s += summary['execution_time_s']
                total_input_mb += summary['input_size_mb']
                total_output_mb += summary['output_size_mb']
                total_features_generated += summary['output_feature_vectors']
                
            except Exception as e:
                print(f"    [ERROR] Failed extracting {ride_id}: {e}")
                continue

        # Compile Experiment Metrics
        experiment_report = {
            "experiment_name": exp_name,
            "pipeline_stage": "feature_extraction",
            "rides_processed": success_count,
            "total_tabular_feature_rows": total_features_generated,
            "total_execution_time_seconds": round(total_time_s, 2),
            "total_input_volume_mb": round(total_input_mb, 2),
            "total_output_volume_mb": round(total_output_mb, 2)
        }
        
        # Save to central metrics folder
        metrics_file = METRICS_DIR / f"features_{exp_name}_metrics.json"
        with open(metrics_file, "w") as f:
            json.dump(experiment_report, f, indent=4)
            
        print(f"[SUCCESS] Experiment {exp_name} complete. Generated {total_features_generated} feature vectors.")

if __name__ == "__main__":
    main()
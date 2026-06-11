import sys
import json

# Ensure the root directory is in the path to import config and src
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

# Import from our central config
from config import RAW_DATA_DIR, PREPARED_DATA_DIR, METRICS_DIR
from src.prep.prepare_simra import process_simra_file

def main() -> None:
    print("\n[INFO] Initiating Data Preparation Phase...")
    
    if not RAW_DATA_DIR.exists():
        print(f"[ERROR] Raw data directory not found at: {RAW_DATA_DIR}")
        sys.exit(1)
        
    ride_files = sorted([p for p in RAW_DATA_DIR.iterdir() if p.is_file()])
    
    if not ride_files:
        print(f"[ERROR] No files found in {RAW_DATA_DIR} to process.")
        sys.exit(1)
        
    print(f"[INFO] Found {len(ride_files)} raw files to process.")
    
    # Track overall performance metrics
    success_count = 0
    fail_count = 0
    total_time_s = 0.0
    total_input_mb = 0.0
    total_output_mb = 0.0
    total_ride_rows = 0
    total_incidents = 0

    # Process each file
    for filepath in ride_files:
        try:
            metrics = process_simra_file(
                raw_filepath=str(filepath), 
                output_directory=str(PREPARED_DATA_DIR)
            )
            
            success_count += 1
            total_time_s += metrics['execution_time_s']
            total_input_mb += metrics['input_size_mb']
            total_output_mb += metrics['output_size_mb']
            total_ride_rows += metrics['ride_row_count']
            total_incidents += metrics['incident_row_count']
            
        except Exception as e:
            print(f"[ERROR] Failed to process {filepath.name}. Reason: {e}")
            fail_count += 1

    # Compile final execution report
    execution_report = {
        "pipeline_stage": "preparation",
        "files_processed": len(ride_files),
        "successful_files": success_count,
        "failed_files": fail_count,
        "total_ride_rows_extracted": total_ride_rows,
        "total_incidents_extracted": total_incidents,
        "total_execution_time_seconds": round(total_time_s, 2),
        "total_input_volume_mb": round(total_input_mb, 2),
        "total_output_volume_mb": round(total_output_mb, 2)
    }

    metrics_file = METRICS_DIR / "preparation_metrics.json"
    
    with open(metrics_file, "w") as f:
        json.dump(execution_report, f, indent=4)
        
    print(f"[SUCCESS] Preparation complete. {success_count} files successfully prepared.")
    print(f"[INFO] Performance metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
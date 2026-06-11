import os
import subprocess
import sys
from pathlib import Path

def run_script(script_path):
    print(f"\n[SYSTEM] >>> INITIATING: {script_path}")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"\n[ERROR] CRITICAL FAILURE: {script_path} exited with error code {result.returncode}")
        sys.exit(1)

def check_data_status():
    """Checks for the presence of raw vs prepared data to determine pipeline start point."""
    raw_path = Path("data/raw")
    prep_path = Path("data/prepared")
    
    # Check if directories exist and have files
    has_raw = raw_path.exists() and any(raw_path.iterdir())
    has_prepared = prep_path.exists() and any(f.suffix == '.parquet' for f in prep_path.iterdir())
    
    return has_raw, has_prepared

def main():
    print("\n[SYSTEM] STARTING END-TO-END KINEMATIC ANALYSIS PIPELINE")
    
    has_raw, has_prepared = check_data_status()
    
    # Dynamically build the execution sequence
    pipeline_sequence = []
    
    # Step 1: Preparation Gateway
    if not has_prepared:
        if has_raw:
            print("[INFO] Raw data found. Adding preparation phase to pipeline.")
            pipeline_sequence.append("scripts/run_prepare.py")
        else:
            print("[ERROR] Neither raw data nor prepared data (.parquet) found in /data/")
            sys.exit(1)
    else:
        print("[INFO] Prepared Data (.parquet) detected. Bypassing raw data parsing.")
    
    # Step 2: Core Engineering & Evaluation Pipeline
    pipeline_sequence.extend([
        "scripts/run_preprocess.py",
        "scripts/run_segment.py",
        "scripts/run_features.py",
        "scripts/run_stage1.py",
        "scripts/run_stage2.py",
        "scripts/run_stage3.py"
    ])
    
    # Step 3: Execute Sequence
    for script in pipeline_sequence:
        if Path(script).exists():
            run_script(script)
        else:
            print(f"\n[ERROR] '{script}' is missing. Cannot continue.")
            sys.exit(1)
            
    print("\n[SYSTEM] FULL PIPELINE EXECUTION SUCCESSFUL")

if __name__ == "__main__":
    main()
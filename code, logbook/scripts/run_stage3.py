import sys
from pathlib import Path

# Ensure the root directory is in the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import STAGE2_DATA_DIR, MODELS_DIR, FIGURES_DIR, METRICS_DIR
from src.models.stage3 import execute_stage3_specialist_classification

def _run_evaluation_mode(ds_file: Path, exp_name: str, use_kinematic: bool) -> None:
    """Executes a single run of Stage 3 (either full target classes or kinematic mapping)."""
    mode_str = "KINEMATIC MAPPED" if use_kinematic else "FULL CLASSES"
    print(f"\n[INFO] --- EVALUATING SPECIALIST ({mode_str}) ---")
    
    result = execute_stage3_specialist_classification(
        input_parquet_path=str(ds_file),
        output_model_dir=str(MODELS_DIR),
        output_figures_dir=str(FIGURES_DIR),
        output_metrics_dir=str(METRICS_DIR),
        use_kinematic_mapping=use_kinematic
    )
    
    if result["status"] != "success":
        print(f"    [SKIP] Failed to process (Status: {result['status']})")
        return
        
    print(f"    [DATA] Classes Evaluated: {', '.join(result['classes_evaluated'])}")
    print(f"    [DATA] CV Macro-F1:       {result['cv_macro_f1']:.4f}")
    print(f"    [SAVED] Best Estimator:   {result['best_model_name']}")
    print(f"    [TEST]  Test Macro-F1:    {result['test_macro_f1']:.4f}")

def main() -> None:
    print("\n[INFO] Initiating Stage 3: Multi-Class Specialist Evaluation...")
    
    if not STAGE2_DATA_DIR.exists():
        print(f"[ERROR] Stage 2 dataset directory not found at {STAGE2_DATA_DIR}")
        sys.exit(1)
        
    dataset_files = sorted(STAGE2_DATA_DIR.glob("stage2_refined_*.parquet"))
    
    if not dataset_files:
        print(f"[ERROR] No refined datasets found in {STAGE2_DATA_DIR}")
        sys.exit(1)
        
    print(f"[INFO] Evaluating models across {len(dataset_files)} refined datasets...")
    
    for ds_file in dataset_files:
        exp_name = ds_file.stem.replace("stage2_refined_", "")
        
        # Consistent, professional logging
        print(f"\n[INFO] >>> PROCESSING EXPERIMENT: {exp_name.upper()}")
        
        try:
            # 1. Evaluate against all raw incident labels
            _run_evaluation_mode(ds_file, exp_name, use_kinematic=False)
            
            # 2. Evaluate against condensed kinematic groups (e.g., 'lateral_swerve')
            _run_evaluation_mode(ds_file, exp_name, use_kinematic=True)
            
        except Exception as e:
            print(f"    [ERROR] Critical failure evaluating {exp_name}. Reason: {e}")

if __name__ == "__main__":
    main()
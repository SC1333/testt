import sys
import json
from pathlib import Path

# Ensure the root directory is in the path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import EXPERIMENTS_DIR, MODELS_DIR, STAGE2_DATA_DIR, METRICS_DIR
from src.models.stage2 import execute_stage2_dataset_refinement

def main() -> None:
    print("\n[INFO] Initiating Stage 2: Hard Negative Mining & Dataset Refinement...")
    
    if not EXPERIMENTS_DIR.exists():
        print(f"[ERROR] Experiments directory not found at {EXPERIMENTS_DIR}")
        sys.exit(1)
        
    experiment_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    STAGE2_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Refining datasets for {len(experiment_dirs)} experiments...")
    
    overall_report = {}
    success_count = 0

    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        print(f"\n[INFO] >>> REFINING EXPERIMENT: {exp_name.upper()}")
        
        # Locate the specific Sentinel model generated in Stage 1
        model_file = MODELS_DIR / f"stage1_sentinel_{exp_name}.joblib"
        
        try:
            result = execute_stage2_dataset_refinement(
                experiment_path=str(exp_dir), 
                model_path=str(model_file), 
                output_dir=str(STAGE2_DATA_DIR)
            )
            
            if result["status"] == "missing_model":
                print(f"    [SKIP] Missing Stage 1 Sentinel model: {model_file.name}")
                continue
            elif result["status"] == "no_data":
                print("    [SKIP] No tabular feature data found.")
                continue
                
            metrics = result["metrics"]
            
            print(f"    [DATA] Original Windows: {metrics['total_original_windows']}")
            print(f"    [DATA] Actual Incidents: {metrics['total_actual_incidents']} (Sentinel missed {metrics['incidents_missed_by_sentinel']})")
            print(f"    [DATA] Hard Negatives:   {metrics['hard_negatives_extracted']}")
            print(f"    [SAVED] Size Reduced By: {metrics['dataset_reduction_percentage']}%")
            print(f"    [SAVED] New Ratio:       {metrics['refined_imbalance_ratio']}:1")
            
            # Store metrics for the final combined JSON log
            overall_report[exp_name] = {
                "input_size_mb": result["input_size_mb"],
                "output_size_mb": result["output_size_mb"],
                "metrics": metrics
            }
            success_count += 1
            
        except Exception as e:
            print(f"    [ERROR] Failed to refine dataset for {exp_name}. Reason: {e}")

    # Save comprehensive metrics to the central reports directory
    metrics_file = METRICS_DIR / "stage2_refinement_metrics.json"
    with open(metrics_file, "w") as f:
        json.dump({
            "pipeline_stage": "hard_negative_mining",
            "successful_experiments": success_count,
            "experiment_results": overall_report
        }, f, indent=4)
        
    print(f"\n[SUCCESS] Stage 2 Complete. Performance metrics saved to: {metrics_file}")

if __name__ == "__main__":
    main()
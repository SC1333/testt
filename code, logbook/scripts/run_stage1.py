import sys
from pathlib import Path

# Ensure the root directory is in the path to import config and src
sys.path.append(str(Path(__file__).resolve().parents[1]))

from config import EXPERIMENTS_DIR, MODELS_DIR, METRICS_DIR
from src.models.stage1 import execute_stage1_binary_classification

def main() -> None:
    print("\n[INFO] Initiating Stage 1: Sentinel Binary Classification...")
    
    if not EXPERIMENTS_DIR.exists():
        print(f"[ERROR] Experiments directory not found at {EXPERIMENTS_DIR}")
        sys.exit(1)
        
    experiment_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    
    if not experiment_dirs:
        print(f"[ERROR] No experimental segmentation configurations found.")
        sys.exit(1)
        
    print(f"[INFO] Evaluating base models across {len(experiment_dirs)} experiments...")
    
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        
        # Optional: You can filter experiments here if you only want to run specific ones
        # if exp_name.upper() in ["BALANCED_4S", "REACTION_FOCUSED"]: continue
            
        print(f"\n[INFO] >>> EVALUATING EXPERIMENT: {exp_name.upper()}")
        
        try:
            result = execute_stage1_binary_classification(
                experiment_path=str(exp_dir), 
                output_model_dir=str(MODELS_DIR),
                results_dir=str(METRICS_DIR)
            )
            
            if not result:
                print("    [SKIP] No tabular feature data found.")
                continue
                
            print(f"    [DATA] Total Windows: {result['n_windows_total']} | Imbalance Ratio: {result['imbalance_ratio']:.1f}:1")
            
            # Print the summary of model performance
            print("    [RESULTS] Cross-Validation F2 Scores:")
            for name, score in result["evaluation_results"].items():
                marker = "-->" if name == result["best_model_name"] else "   "
                print(f"    {marker} {name:<20}: {score:.4f}")
                
            print(f"    [SAVED] Best Estimator: {result['best_model_name']} ({result['output_size_mb']}MB)")
            print(f"    [TIME]  {result['execution_time_s']}s")
            
        except Exception as e:
            print(f"    [ERROR] Failed to execute Stage 1 for {exp_name}. Reason: {e}")

if __name__ == "__main__":
    main()
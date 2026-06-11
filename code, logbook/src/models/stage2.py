from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, List
import time
import pandas as pd
import joblib

def _mine_hard_negatives(
    df: pd.DataFrame, 
    model: Any, 
    drop_cols: List[str], 
    target_col: str = "label"
) -> Dict[str, Any]:
    """
    Applies the trained Sentinel model to the dataset. Retains all actual incidents, 
    but filters 'normal riding' down to only the instances the model misclassified (Hard Negatives).
    """
    start_time = time.perf_counter()
    
    # Isolate feature columns for prediction
    feature_cols = [c for c in df.columns if c not in drop_cols]
    X = df[feature_cols]
    
    # Generate predictions using the Stage 1 Sentinel
    df['sentinel_prediction'] = model.predict(X)
    
    # 1. Isolate Actual Incidents (True Positives + False Negatives)
    actual_incidents = df[df[target_col] == 1].copy()
    missed_count = len(actual_incidents[actual_incidents['sentinel_prediction'] == 0])
    
    # 2. Isolate Hard Negatives (False Positives)
    # These are 'normal riding' windows that kinematically resemble incidents
    hard_negatives = df[(df[target_col] == 0) & (df['sentinel_prediction'] == 1)].copy()
    
    # Enforce strict metadata for hard negatives so Stage 3 knows they are non-events
    hard_negatives['incident_type'] = "normal_riding"
    hard_negatives['incident_code'] = 0
    
    # 3. Combine into the newly refined dataset
    refined_df = pd.concat([actual_incidents, hard_negatives], ignore_index=True)
    refined_df = refined_df.drop(columns=['sentinel_prediction'])
    
    execution_time = time.perf_counter() - start_time
    imbalance_ratio = len(hard_negatives) / len(actual_incidents) if len(actual_incidents) > 0 else 0
    
    metrics = {
        "total_original_windows": len(df),
        "total_actual_incidents": len(actual_incidents),
        "incidents_missed_by_sentinel": missed_count,
        "hard_negatives_extracted": len(hard_negatives),
        "total_refined_windows": len(refined_df),
        "dataset_reduction_percentage": round(((len(df) - len(refined_df)) / len(df)) * 100, 2) if len(df) > 0 else 0,
        "refined_imbalance_ratio": round(imbalance_ratio, 2),
        "execution_time_s": round(execution_time, 3)
    }
    
    return {"refined_df": refined_df, "metrics": metrics}

def _load_experiment_data(exp_path: Path) -> pd.DataFrame:
    """Aggregates all feature parquet files within an experiment directory."""
    files = list(exp_path.glob("*_features.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def execute_stage2_dataset_refinement(
    experiment_path: str,
    model_path: str,
    output_dir: str,
    target_col: str = "label"
) -> Dict[str, Any]:
    """Orchestrates the hard negative mining process to generate the Stage 2 dataset."""
    exp_path = Path(experiment_path)
    mod_path = Path(model_path)
    out_dir = Path(output_dir)
    
    if not mod_path.exists():
        return {"status": "missing_model"}
        
    df = _load_experiment_data(exp_path)
    if df.empty:
        return {"status": "no_data"}
        
    input_size_mb = sum(f.stat().st_size for f in exp_path.glob("*_features.parquet")) / (1024 * 1024)
    
    # Standard columns to drop before passing to the model
    drop_cols = ["window_id", "ride_id", "incident_type", "incident_code", target_col]
    
    # Load model and apply mining
    sentinel_model = joblib.load(mod_path)
    mining_results = _mine_hard_negatives(df, sentinel_model, drop_cols, target_col)
    
    refined_df = mining_results["refined_df"]
    metrics = mining_results["metrics"]
    
    # Save the refined dataset
    out_name = f"stage2_refined_{exp_path.name}.parquet"
    out_path = out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    refined_df.to_parquet(out_path, index=False)
    
    output_size_mb = out_path.stat().st_size / (1024 * 1024)
    
    return {
        "status": "success",
        "metrics": metrics,
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "output_name": out_name
    }
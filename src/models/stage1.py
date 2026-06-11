from __future__ import annotations
from pathlib import Path
from typing import Dict, Any, Optional
import time
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import StratifiedGroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import (
    make_scorer, fbeta_score, classification_report, 
    confusion_matrix, accuracy_score, roc_auc_score, 
    average_precision_score
)

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder to handle NumPy data types."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

def _get_model_configurations(imbalance_ratio: float) -> Dict[str, Dict[str, Any]]:
    """Defines the baseline models and their hyperparameter grids for evaluation."""
    return {
        "Logistic_Regression": {
            "estimator": Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
            ]),
            "params": {
                'clf__C': [0.1, 1.0, 10.0]
            }
        },
        "Random_Forest": {
            "estimator": RandomForestClassifier(class_weight='balanced_subsample', random_state=42),
            "params": {
                'n_estimators': [50, 100],
                'max_depth': [5, 10, None],
                'min_samples_split': [2, 5]
            }
        },
        "Gradient_Boosting": {
            "estimator": GradientBoostingClassifier(random_state=42),
            "params": {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5],
                'n_estimators': [50, 100]
            }
        },
        "XGBoost": {
            "estimator": XGBClassifier(scale_pos_weight=imbalance_ratio, eval_metric='logloss', random_state=42),
            "params": {
                'learning_rate': [0.01, 0.1],
                'max_depth': [3, 5, 7],
                'n_estimators': [50, 100]
            }
        }
    }

def _evaluate_baselines(X_train: pd.DataFrame, y_train: pd.Series, groups_train: pd.Series, imbalance_ratio: float, n_splits: int = 3) -> Dict[str, Any]:
    """Executes a Grid Search across defined baseline models, optimizing for the F2 score."""
    model_configs = _get_model_configurations(imbalance_ratio)
    
    # We use F2 score because Recall (catching incidents) is more important than Precision
    f2_scorer = make_scorer(fbeta_score, beta=2, zero_division=0)
    
    # StratifiedGroupKFold ensures that rare incidents are distributed evenly across folds
    cv_strategy = StratifiedGroupKFold(n_splits=n_splits)
    
    results = {}
    best_score = -1
    best_estimator = None
    best_model_name = ""
    
    for name, config in model_configs.items():
        grid_search = GridSearchCV(
            estimator=config["estimator"], 
            param_grid=config["params"], 
            scoring=f2_scorer, 
            cv=cv_strategy, 
            n_jobs=-1
        )
        grid_search.fit(X_train, y_train, groups=groups_train)
        
        results[name] = {
            "best_cv_f2": grid_search.best_score_, 
            "best_params": grid_search.best_params_, 
            "trained_model": grid_search.best_estimator_
        }
        
        if grid_search.best_score_ > best_score:
            best_score = grid_search.best_score_
            best_estimator = grid_search.best_estimator_
            best_model_name = name
            
    return {
        "evaluation_results": results, 
        "best_model_name": best_model_name, 
        "best_cv_score": best_score, 
        "best_estimator": best_estimator
    }

def _load_experiment_data(exp_path: Path) -> pd.DataFrame:
    """Aggregates all feature parquet files within an experiment directory into a single DataFrame."""
    files = list(exp_path.glob("*_features.parquet"))
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

def execute_stage1_binary_classification(
    experiment_path: str, 
    output_model_dir: str, 
    results_dir: str,
    target_col: str = "label",
    group_col: str = "ride_id"
) -> Optional[Dict[str, Any]]:
    """Orchestrates the Stage 1 binary classification evaluation and saves the best model."""
    start_time = time.perf_counter()
    
    exp_path = Path(experiment_path)
    out_dir = Path(output_model_dir)
    res_dir = Path(results_dir) / exp_path.name
    
    df = _load_experiment_data(exp_path)
    if df.empty:
        return None
        
    # Drop structural metadata columns before training
    drop_cols = ["window_id", "ride_id", "incident_type", "incident_code", "label", "window_usable_flag", "event_time_s", "start_idx", "end_idx", "n_rows", "usable_row_fraction"]
    
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df[target_col]
    groups = df[group_col]
    
    # 1. Establish a strict, grouped, unseen Test Set (20% of rides)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    groups_train = groups.iloc[train_idx]
    
    # Calculate the positive/negative imbalance for scaling models (e.g., XGBoost scale_pos_weight)
    imbalance_ratio = (y_train == 0).sum() / (y_train == 1).sum() if (y_train == 1).sum() > 0 else 1.0
    
    # 2. Execute the evaluation across baseline models
    evaluation_data = _evaluate_baselines(X_train, y_train, groups_train, imbalance_ratio)
    
    # 3. Log comprehensive metrics for ALL evaluated models on the unseen Test Set
    res_dir.mkdir(parents=True, exist_ok=True)
    
    for model_name, model_data in evaluation_data["evaluation_results"].items():
        model = model_data["trained_model"]
        y_pred = model.predict(X_test)
        probas = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        cm = confusion_matrix(y_test, y_pred)
        
        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model_metadata": {
                "model_name": model_name, 
                "is_best_estimator": model_name == evaluation_data["best_model_name"]
            },
            "data_characteristics": {
                "test_samples": len(y_test), 
                "actual_incidents": int(sum(y_test))
            },
            "performance": {
                "cv_f2_score": model_data["best_cv_f2"],
                "test_f2_score": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
                "accuracy": accuracy_score(y_test, y_pred),
                "confusion_matrix": {
                    "True_Negatives": cm[0][0], 
                    "False_Positives": cm[0][1], 
                    "False_Negatives": cm[1][0], 
                    "True_Positives": cm[1][1], 
                    "Total_Windows_Flagged": cm[0][1] + cm[1][1]
                },
                "classification_report": classification_report(y_test, y_pred, output_dict=True, zero_division=0)
            }
        }
        
        if probas is not None:
            metrics["performance"]["roc_auc"] = roc_auc_score(y_test, probas)
            metrics["performance"]["pr_auc"] = average_precision_score(y_test, probas)
            
        with open(res_dir / f"stage1_{model_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
            
    # 4. Save ONLY the best performing model to disk for use in Stage 2
    out_name = f"stage1_sentinel_{exp_path.name}.joblib"
    out_path = out_dir / out_name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(evaluation_data["best_estimator"], out_path)
    
    execution_time = time.perf_counter() - start_time
    
    # Clean up the output dictionary to only pass relevant data back to the script
    return {
        "n_windows_total": len(df),
        "imbalance_ratio": imbalance_ratio,
        "best_model_name": evaluation_data["best_model_name"],
        "best_cv_f2": evaluation_data["best_cv_score"],
        "evaluation_results": {k: v["best_cv_f2"] for k, v in evaluation_data["evaluation_results"].items()},
        "output_name": out_name,
        "output_size_mb": round(out_path.stat().st_size / (1024 * 1024), 2),
        "execution_time_s": round(execution_time, 3)
    }
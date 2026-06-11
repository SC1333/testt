from __future__ import annotations
import time
import json
from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import GroupShuffleSplit, GridSearchCV, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, make_scorer, 
    f1_score, roc_auc_score, log_loss
)

# --- Configuration & Custom Classes ---

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for NumPy types to ensure logging compatibility."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

# Optional mapping to group context-specific labels into physical kinematic signatures
KINEMATIC_MAPPING = {
    'close_pass': 'lateral_swerve',
    'dodging_obstacle': 'lateral_swerve',
    'near_left_or_right_hook': 'lateral_swerve',
    'near_dooring': 'lateral_swerve',
    'someone_pulling_in_or_out': 'longitudinal_brake',
    'tailgating': 'longitudinal_brake',
    'someone_approaching_head_on': 'longitudinal_brake',
    'normal_riding': 'normal_riding',
    'other': 'normal_riding'
}

# --- Core Logic Functions ---

def _get_multiclass_configurations() -> Dict[str, Dict[str, Any]]:
    """Defines the specialized multiclass algorithms and their hyperparameter grids."""
    return {
        "Random_Forest_Multi": {
            "estimator": RandomForestClassifier(class_weight='balanced', random_state=42),
            "params": {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
        },
        "XGBoost_Multi": {
            "estimator": XGBClassifier(objective='multi:softprob', eval_metric='mlogloss', random_state=42),
            "params": {'learning_rate': [0.01, 0.1], 'max_depth': [3, 5, 7]}
        },
        "Linear_SVC": {
            "estimator": Pipeline([
                ('scaler', StandardScaler()),
                ('clf', LinearSVC(class_weight='balanced', max_iter=2000, random_state=42, dual=False))
            ]),
            "params": {'clf__C': [0.1, 1, 10]}
        }
    }

def _enforce_cv_integrity(df: pd.DataFrame, target_col: str, group_col: str, min_required_groups: int = 4) -> pd.DataFrame:
    """
    Safeguard: Drops target classes that do not have enough distinct groups (rides)
    to survive a Train/Test split AND a subsequent K-Fold Cross Validation.
    Prevents XGBoost from throwing missing-class ValueError exceptions.
    """
    group_counts_per_class = df.groupby(target_col)[group_col].nunique()
    valid_classes = group_counts_per_class[group_counts_per_class >= min_required_groups].index
    
    if len(valid_classes) < len(group_counts_per_class):
        dropped_classes = set(group_counts_per_class.index) - set(valid_classes)
        print(f"\n    [WARNING] Dropping ultra-rare classes to preserve Stratified K-Fold integrity: {dropped_classes}")
        return df[df[target_col].isin(valid_classes)].copy()
        
    return df

def _generate_evaluation_artifacts(
    y_test: np.ndarray, 
    y_pred: np.ndarray, 
    class_names: List[str], 
    model: Any, 
    feature_names: List[str], 
    model_name: str, 
    cm_path: Path, 
    fi_path: Path
) -> None:
    """Generates and saves the Confusion Matrix and Feature Importance visual artifacts."""
    # 1. Normalized Confusion Matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.title(f"Normalized Confusion Matrix ({model_name})")
    plt.ylabel('True Incident Class')
    plt.xlabel('Predicted Incident Class')
    plt.tight_layout()
    plt.savefig(cm_path, dpi=300)
    plt.close()

    # 2. Feature Importance (If supported by model)
    estimator = model.named_steps['clf'] if isinstance(model, Pipeline) else model
    if hasattr(estimator, 'feature_importances_'):
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1][:15] # Top 15 features
        
        plt.figure(figsize=(12, 6))
        plt.title(f"Top 15 Feature Importances ({model_name})")
        plt.bar(range(15), importances[indices], align="center")
        plt.xticks(range(15), [feature_names[i] for i in indices], rotation=45, ha='right')
        plt.xlim([-1, 15])
        plt.tight_layout()
        plt.savefig(fi_path, dpi=300)
        plt.close()

# --- Main Execution Function ---

def execute_stage3_specialist_classification(
    input_parquet_path: str,
    output_model_dir: str,
    output_figures_dir: str,
    output_metrics_dir: str,
    use_kinematic_mapping: bool = False,
    target_col: str = "incident_type",
    group_col: str = "ride_id"
) -> Dict[str, Any]:
    """Orchestrates the Stage 3 multiclass evaluation."""
    start_time = time.perf_counter()
    in_file = Path(input_parquet_path)
    
    if not in_file.exists():
        return {"status": "missing_data"}
        
    df = pd.read_parquet(in_file)
    if df.empty:
        return {"status": "empty_data"}
        
    input_size_mb = in_file.stat().st_size / (1024 * 1024)
    exp_name = in_file.stem.replace("stage2_refined_", "")
    mapping_mode = "kinematic" if use_kinematic_mapping else "full"
    
    # 1. Transform targets if using physical mapping
    if use_kinematic_mapping:
        df[target_col] = df[target_col].replace(KINEMATIC_MAPPING)
        
    # 2. Enforce Data Integrity
    df = _enforce_cv_integrity(df, target_col, group_col, min_required_groups=4)
    if df.empty:
        return {"status": "empty_data_after_integrity_filter"}
        
    # 3. Prepare Feature Matrix and Label Encoding
    drop_cols = ["window_id", "ride_id", "incident_code", "label"]
    X = df.drop(columns=[c for c in drop_cols + [target_col] if c in df.columns])
    groups = df[group_col]
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(df[target_col])
    class_names = le.classes_
    
    # 4. Grouped Train/Test Split
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y_encoded, groups))
    
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]
    groups_train = groups.iloc[train_idx]
    
    # Optimize for Macro F1 (Treats rare classes equally to common classes)
    macro_f1_scorer = make_scorer(f1_score, average='macro', zero_division=0)
    cv_strategy = StratifiedGroupKFold(n_splits=3)
    
    best_cv_score = -1
    best_test_score = -1
    best_model = None
    best_model_name = ""
    
    # Setup Output Directories
    res_dir = Path(output_metrics_dir) / exp_name
    fig_dir = Path(output_figures_dir) / exp_name
    mod_dir = Path(output_model_dir)
    for d in [res_dir, fig_dir, mod_dir]: d.mkdir(parents=True, exist_ok=True)
    
    # 5. Evaluate Multi-class Models
    for name, config in _get_multiclass_configurations().items():
        grid = GridSearchCV(
            estimator=config["estimator"], 
            param_grid=config["params"],
            scoring=macro_f1_scorer, 
            cv=cv_strategy, 
            n_jobs=-1
        )
        grid.fit(X_train, y_train, groups=groups_train)
        
        current_model = grid.best_estimator_
        y_pred = current_model.predict(X_test)
        
        # Identify which classes actually appeared in the test set to format report correctly
        present_labels = np.unique(np.concatenate((y_test, y_pred)))
        present_names = [class_names[i] for i in present_labels]
        
        clf_report = classification_report(y_test, y_pred, labels=present_labels, target_names=present_names, zero_division=0, output_dict=True)
        test_macro_f1 = clf_report['macro avg']['f1-score']
        
        # Log Metrics
        metrics = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mapping_mode": mapping_mode,
            "model_name": name,
            "performance": {
                "cv_f1_macro": grid.best_score_,
                "test_f1_macro": test_macro_f1,
                "classification_report": clf_report
            }
        }
        
        # Visual Artifacts
        safe_name = name.replace(" ", "_")
        cm_path = fig_dir / f"stage3_cm_{mapping_mode}_{safe_name}.png"
        fi_path = fig_dir / f"stage3_fi_{mapping_mode}_{safe_name}.png"
        
        _generate_evaluation_artifacts(y_test, y_pred, class_names, current_model, X.columns.tolist(), name, cm_path, fi_path)
        
        with open(res_dir / f"stage3_{mapping_mode}_{safe_name}_metrics.json", "w") as f:
            json.dump(metrics, f, indent=4, cls=NumpyEncoder)
            
        # Track the Best Model
        if grid.best_score_ > best_cv_score:
            best_cv_score = grid.best_score_
            best_test_score = test_macro_f1
            best_model = current_model
            best_model_name = name

    # 6. Save the Final Specialist Model
    mod_path = mod_dir / f"stage3_specialist_{mapping_mode}_{exp_name}.joblib"
    joblib.dump(best_model, mod_path)
    output_size_mb = mod_path.stat().st_size / (1024 * 1024)
    
    execution_time = time.perf_counter() - start_time
    
    return {
        "status": "success",
        "best_model_name": best_model_name,
        "cv_macro_f1": best_cv_score,
        "test_macro_f1": best_test_score,
        "classes_evaluated": list(class_names),
        "input_size_mb": round(input_size_mb, 2),
        "output_size_mb": round(output_size_mb, 2),
        "execution_time_s": round(execution_time, 3),
        "model_path": str(mod_path.name)
    }
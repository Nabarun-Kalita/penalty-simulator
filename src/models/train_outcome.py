"""
Phase 3 - Step 3: Train the Outcome model.

Input:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    data/processed/feature_metadata.json

Output:
    models/outcome_model.pkl       (the XGBoost classifier)
    models/outcome_calibrator.pkl  (per-class isotonic calibrators)
    models/outcome_metrics.json    (cross-val + test scores, feature importance)

What this model does:
    Given the taker, keeper, the zone the shot went to, and context,
    predicts the outcome category (GOAL / SAVED / POST / WIDE / OVER)
    as a 5-class probability distribution.

Trains on ALL rows (on-target and off-target) since the outcome
includes both. Off-target zones get treated as a separate category.

Pipeline:
    1. Load features + metadata.
    2. Compute baselines:
       - Population outcome distribution (constant prediction)
       - Zone-conditional outcome (look-up by zone)
    3. Cross-validate small XGBoost grid (5-fold stratified).
    4. Refit best on full train; calibrate via isotonic regression.
    5. Evaluate on 2022 World Cup test set.
    6. Save model, calibrator, and metrics.

Usage:
    python src/models/train_outcome.py
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from itertools import product

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from src.models.calibration import IsotonicMultiClassCalibrator

import xgboost as xgb

# ----- Paths -----
TRAIN_FILE = "data/processed/train_features.parquet"
TEST_FILE = "data/processed/test_features.parquet"
META_FILE = "data/processed/feature_metadata.json"

MODELS_DIR = "models"
MODEL_FILE = os.path.join(MODELS_DIR, "outcome_model.pkl")
CALIB_FILE = os.path.join(MODELS_DIR, "outcome_calibrator.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "outcome_metrics.json")

# ----- Config -----
N_SPLITS = 5
RANDOM_STATE = 42
OUTCOME_CLASSES = ['GOAL', 'OVER', 'POST', 'SAVED', 'WIDE']  # 5 classes, alphabetical
ZONES = ['TL', 'TC', 'TR', 'BL', 'BC', 'BR']

# Small hyperparameter grid 
PARAM_GRID = {
    'max_depth': [3, 4],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
    'min_child_weight': [5, 10],
    'reg_lambda': [1, 5],
    'subsample': [0.8],
    'colsample_bytree': [0.8],
}


# ============================================================
# DATA PREP
# ============================================================

def load_data():
    train = pd.read_parquet(TRAIN_FILE)
    test = pd.read_parquet(TEST_FILE)
    with open(META_FILE) as f:
        meta = json.load(f)
    return train, test, meta


def prepare_xy(df, feature_cols, target_col, encoder=None):
    """Encode target as integers. No row filtering — outcome model uses all rows."""
    X = df[feature_cols].copy()
    y_str = df[target_col]
    if encoder is None:
        encoder = LabelEncoder()
        y = encoder.fit_transform(y_str)
    else:
        y = encoder.transform(y_str)
    return X, y, encoder


# ============================================================
# BASELINES
# ============================================================

def population_baseline_proba(n, outcome_counts, encoder):
    """Predict the population outcome distribution for every row."""
    total = sum(outcome_counts.values())
    proba_row = np.array([outcome_counts[c] / total for c in encoder.classes_])
    proba_row = proba_row / proba_row.sum()
    return np.tile(proba_row, (n, 1))


def zone_conditional_baseline_proba(df, train_df, encoder):
    """
    For each row, predict the outcome distribution conditional on the row's zone,
    learned from the training data only.

    Off-target zones (POST, WIDE, OVER, etc. — values not in the 6 grid zones)
    fall back to the overall distribution.
    """
    # Build per-zone outcome distributions from train_df
    zone_to_dist = {}
    for zone in ZONES:
        z_rows = train_df[train_df['zone'] == zone]
        if len(z_rows) > 0:
            counts = z_rows['outcome_category'].value_counts()
            dist = np.array([
                counts.get(c, 0) / len(z_rows) for c in encoder.classes_
            ])
            zone_to_dist[zone] = dist / dist.sum() if dist.sum() > 0 else dist
        else:
            zone_to_dist[zone] = None

    # Overall distribution fallback (for off-target rows)
    overall_counts = train_df['outcome_category'].value_counts()
    overall_dist = np.array([
        overall_counts.get(c, 0) / len(train_df) for c in encoder.classes_
    ])
    overall_dist = overall_dist / overall_dist.sum()

    n = len(df)
    proba = np.zeros((n, len(encoder.classes_)))
    for i, zone in enumerate(df['zone'].values):
        if zone in ZONES and zone_to_dist.get(zone) is not None:
            proba[i] = zone_to_dist[zone]
        else:
            proba[i] = overall_dist

    return proba


# ============================================================
# CROSS-VALIDATION GRID SEARCH
# ============================================================

def cross_validate_xgb(X, y, params, n_classes, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses = []
    accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=random_state,
            **params,
        )
        model.fit(X_tr, y_tr, verbose=False)

        proba = model.predict_proba(X_val)
        losses.append(log_loss(y_val, proba, labels=np.arange(n_classes)))
        accs.append(accuracy_score(y_val, proba.argmax(axis=1)))

    return {
        'logloss_mean': float(np.mean(losses)),
        'logloss_std': float(np.std(losses)),
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
    }


def grid_search(X, y, param_grid, n_classes):
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} parameter combinations")
    print(f"  {N_SPLITS}-fold stratified CV on each\n")

    all_results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        scores = cross_validate_xgb(X, y, params, n_classes)
        all_results.append({**params, **scores})
        print(f"  [{i+1}/{len(combos)}] "
              f"max_depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, "
              f"n_est={params['n_estimators']}, "
              f"min_child_weight={params['min_child_weight']} "
              f"→ logloss={scores['logloss_mean']:.4f} ± {scores['logloss_std']:.4f}, "
              f"acc={scores['accuracy_mean']:.3f}")

    best = min(all_results, key=lambda r: r['logloss_mean'])
    best_params = {k: best[k] for k in keys}
    return best_params, all_results


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    train, test, meta = load_data()
    print(f"  Train: {len(train)} rows")
    print(f"  Test:  {len(test)} rows")

    # Outcome model uses ALL feature groups
    feature_cols = (
        meta['taker_features_loo']
        + meta['keeper_features_loo']
        + meta['context_features']
        + meta['zone_features']
    )
    target_col = meta['target_columns']['outcome_5class']
    n_classes = len(OUTCOME_CLASSES)
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target classes: {n_classes}")

    # ----- Drop rows with NaN in zone_features (off-target rows have NaNs there) -----
    # Strategy: fill NaN with column means rather than dropping.
    # For tree models this is fine, and we don't lose the 70 off-target rows.
    print("\nFilling NaN values in zone features with column means...")
    nan_before = train[feature_cols].isna().sum().sum()
    print(f"  NaN values before fill: {nan_before}")
    train_filled = train.copy()
    test_filled = test.copy()
    for col in feature_cols:
        if train_filled[col].isna().any():
            mean_val = train_filled[col].mean()
            train_filled[col] = train_filled[col].fillna(mean_val)
            test_filled[col] = test_filled[col].fillna(mean_val)
    nan_after = train_filled[feature_cols].isna().sum().sum()
    print(f"  NaN values after fill:  {nan_after}")

    # ----- Prepare X, y -----
    X_train, y_train, encoder = prepare_xy(train_filled, feature_cols, target_col)
    X_test, y_test, _ = prepare_xy(test_filled, feature_cols, target_col, encoder)

    print(f"\nOutcome class encoding:")
    for c, name in enumerate(encoder.classes_):
        count = (y_train == c).sum()
        print(f"  {c}: {name} ({count} train rows)")

    # ----- Baselines -----
    print("\n--- Baselines (on training data, in-sample) ---")
    outcome_counts = train_filled['outcome_category'].value_counts().to_dict()
    pop_proba_train = population_baseline_proba(len(X_train), outcome_counts, encoder)
    zone_proba_train = zone_conditional_baseline_proba(train_filled, train_filled, encoder)

    pop_logloss = log_loss(y_train, pop_proba_train, labels=np.arange(n_classes))
    pop_acc = accuracy_score(y_train, pop_proba_train.argmax(axis=1))
    zone_logloss = log_loss(y_train, zone_proba_train, labels=np.arange(n_classes))
    zone_acc = accuracy_score(y_train, zone_proba_train.argmax(axis=1))

    print(f"  Population baseline:       logloss={pop_logloss:.4f}, accuracy={pop_acc:.3f}")
    print(f"  Zone-conditional baseline: logloss={zone_logloss:.4f}, accuracy={zone_acc:.3f}")

    # ----- Grid search -----
    best_params, all_results = grid_search(X_train, y_train, PARAM_GRID, n_classes)
    print(f"\nBest params: {best_params}")

    # ----- Refit best on full training set -----
    print("\nRefitting best model on full training set...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=n_classes,
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        **best_params,
    )
    model.fit(X_train, y_train, verbose=False)

    # ----- Calibration: fit on out-of-fold predictions -----
    print("\nGenerating out-of-fold predictions for calibration...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(X_train), n_classes))
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=n_classes,
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=RANDOM_STATE,
            **best_params,
        )
        m.fit(X_train.iloc[train_idx], y_train[train_idx], verbose=False)
        oof_proba[val_idx] = m.predict_proba(X_train.iloc[val_idx])

    calibrator = IsotonicMultiClassCalibrator()
    calibrator.fit(oof_proba, y_train, n_classes=n_classes)
    print("Calibrator fitted.")

    # ----- Test set evaluation -----
    print("\n--- Test set evaluation (2022 World Cup) ---")
    test_proba_raw = model.predict_proba(X_test)
    test_proba_cal = calibrator.predict_proba(test_proba_raw)

    test_logloss_raw = log_loss(y_test, test_proba_raw, labels=np.arange(n_classes))
    test_logloss_cal = log_loss(y_test, test_proba_cal, labels=np.arange(n_classes))
    test_acc_raw = accuracy_score(y_test, test_proba_raw.argmax(axis=1))
    test_acc_cal = accuracy_score(y_test, test_proba_cal.argmax(axis=1))

    print(f"  Raw model:        logloss={test_logloss_raw:.4f}, accuracy={test_acc_raw:.3f}")
    print(f"  Calibrated model: logloss={test_logloss_cal:.4f}, accuracy={test_acc_cal:.3f}")

    pop_proba_test = population_baseline_proba(len(X_test), outcome_counts, encoder)
    zone_proba_test = zone_conditional_baseline_proba(test_filled, train_filled, encoder)
    pop_logloss_test = log_loss(y_test, pop_proba_test, labels=np.arange(n_classes))
    zone_logloss_test = log_loss(y_test, zone_proba_test, labels=np.arange(n_classes))
    print(f"  Population baseline:       logloss={pop_logloss_test:.4f}")
    print(f"  Zone-conditional baseline: logloss={zone_logloss_test:.4f}")

    # ----- Binary GOAL/NOT-GOAL evaluation (most important for the dashboard) -----
    print("\n--- Binary evaluation (GOAL vs anything else) ---")
    goal_class = list(encoder.classes_).index('GOAL')

    # Predicted probability of GOAL
    p_goal_test = test_proba_cal[:, goal_class]
    y_goal_test = (y_test == goal_class).astype(int)

    # Binary log-loss treating GOAL as class 1
    from sklearn.metrics import log_loss as binary_logloss
    bin_ll = binary_logloss(y_goal_test, p_goal_test)
    bin_acc = accuracy_score(y_goal_test, (p_goal_test > 0.5).astype(int))
    print(f"  Calibrated GOAL probability — binary logloss: {bin_ll:.4f}")
    print(f"  Accuracy at 0.5 threshold: {bin_acc:.3f}")
    print(f"  Mean predicted GOAL probability: {p_goal_test.mean():.3f}")
    print(f"  Actual GOAL rate in test:        {y_goal_test.mean():.3f}")

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_test, test_proba_cal.argmax(axis=1),
                          labels=np.arange(n_classes))
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    classes_short = [c[:5] for c in encoder.classes_]
    print("        " + "  ".join(f"{c:>5}" for c in classes_short))
    for i, c in enumerate(classes_short):
        print(f"  {c:>5} " + "  ".join(f"{cm[i, j]:>5}" for j in range(n_classes)))

    # ----- Feature importance -----
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 15 features by importance:")
    for name, val in importance.head(15).items():
        print(f"  {name:45} {val:.4f}")

    # ----- Save artifacts -----
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        'model': model,
        'encoder': encoder,
        'feature_cols': feature_cols,
    }, MODEL_FILE)
    joblib.dump(calibrator, CALIB_FILE)

    metrics = {
        'best_params': best_params,
        'grid_search_results': all_results,
        'cv': {
            'logloss_mean': float(min(r['logloss_mean'] for r in all_results)),
        },
        'baselines': {
            'population_logloss_train': float(pop_logloss),
            'zone_conditional_logloss_train': float(zone_logloss),
            'population_logloss_test': float(pop_logloss_test),
            'zone_conditional_logloss_test': float(zone_logloss_test),
        },
        'test': {
            'logloss_raw': float(test_logloss_raw),
            'logloss_calibrated': float(test_logloss_cal),
            'accuracy_raw': float(test_acc_raw),
            'accuracy_calibrated': float(test_acc_cal),
            'binary_goal_logloss': float(bin_ll),
            'binary_goal_accuracy': float(bin_acc),
            'mean_predicted_goal_prob': float(p_goal_test.mean()),
            'actual_goal_rate': float(y_goal_test.mean()),
            'confusion_matrix': cm.tolist(),
            'classes': encoder.classes_.tolist(),
        },
        'feature_importance': importance.to_dict(),
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PHASE 3 STEP 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:      {MODEL_FILE}")
    print(f"  Calibrator: {CALIB_FILE}")
    print(f"  Metrics:    {METRICS_FILE}")


if __name__ == "__main__":
    main()

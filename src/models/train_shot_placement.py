"""
Phase 3 - Step 2: Train the Shot Placement model.

Input:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    data/processed/feature_metadata.json

Output:
    models/shot_placement_model.pkl       (the XGBoost classifier)
    models/shot_placement_calibrator.pkl  (per-class isotonic calibrators)
    models/shot_placement_metrics.json    (cross-val + test scores, feature importance)

What this model does:
    Predicts the zone a taker will aim for (TL/TC/TR/BL/BC/BR),
    given the taker's profile and match context. Returns a
    6-class probability distribution.

Training only uses ON-TARGET rows (zone in the 6-zone grid).
Off-target outcomes are handled by the outcome model in Step 3.

Pipeline:
    1. Load features + metadata.
    2. Filter to on-target rows.
    3. Compute two baselines:
       - Population zone distribution (constant prediction)
       - Per-taker LOO zone distribution (taker_zone_{Z}_prob_loo)
    4. Cross-validate small XGBoost grid (5-fold stratified).
    5. Refit best model on full train; calibrate via isotonic regression.
    6. Evaluate on 2022 World Cup test set.
    7. Save model, calibrator, and metrics.

Usage:
    python src/models/train_shot_placement.py
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
MODEL_FILE = os.path.join(MODELS_DIR, "shot_placement_model.pkl")
CALIB_FILE = os.path.join(MODELS_DIR, "shot_placement_calibrator.pkl")
METRICS_FILE = os.path.join(MODELS_DIR, "shot_placement_metrics.json")

# ----- Config -----
N_SPLITS = 5
RANDOM_STATE = 42
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


def prepare_xy(df, feature_cols, target_col, zone_encoder=None):
    """Filter to on-target rows and split into X, y."""
    on_target = df[df[target_col].isin(ZONES)].copy().reset_index(drop=True)
    X = on_target[feature_cols].copy()
    y_str = on_target[target_col]
    if zone_encoder is None:
        zone_encoder = LabelEncoder()
        y = zone_encoder.fit_transform(y_str)
    else:
        y = zone_encoder.transform(y_str)
    return X, y, zone_encoder, on_target


# ============================================================
# BASELINES
# ============================================================

def population_baseline_proba(n, zone_distribution, zone_encoder):
    """
    Predict the population zone distribution for every row.
    Returns an (n, 6) array where every row is the same distribution.
    """
    # Order probs to match the encoder's class order
    proba_row = np.array([zone_distribution[z] for z in zone_encoder.classes_])
    proba_row = proba_row / proba_row.sum()  # ensure sums to 1
    return np.tile(proba_row, (n, 1))


def taker_prior_baseline_proba(df, zone_encoder):
    """
    Use each row's taker_zone_{Z}_prob_loo as the prediction.
    This is the strongest no-ML baseline — it's just the LOO profile.
    """
    cols = [f'taker_zone_{z}_prob_loo' for z in zone_encoder.classes_]
    proba = df[cols].values
    # Normalize defensively (should already sum to 1, but just in case)
    proba = proba / proba.sum(axis=1, keepdims=True)
    return proba


# ============================================================
# CROSS-VALIDATION GRID SEARCH
# ============================================================

def cross_validate_xgb(X, y, params, n_splits=N_SPLITS, random_state=RANDOM_STATE):
    """Run stratified k-fold CV, return mean and std log-loss."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    losses = []
    accs = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(ZONES),
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=random_state,
            **params,
        )
        model.fit(X_tr, y_tr, verbose=False)

        proba = model.predict_proba(X_val)
        # Provide labels to log_loss so missing classes in y_val don't crash it
        losses.append(log_loss(y_val, proba, labels=np.arange(len(ZONES))))
        accs.append(accuracy_score(y_val, proba.argmax(axis=1)))

    return {
        'logloss_mean': float(np.mean(losses)),
        'logloss_std': float(np.std(losses)),
        'accuracy_mean': float(np.mean(accs)),
        'accuracy_std': float(np.std(accs)),
    }


def grid_search(X, y, param_grid):
    """Try every combo in the grid, return best params + all results."""
    keys = list(param_grid.keys())
    combos = list(product(*[param_grid[k] for k in keys]))

    print(f"\nGrid search: {len(combos)} parameter combinations")
    print(f"  {N_SPLITS}-fold stratified CV on each\n")

    all_results = []
    for i, combo in enumerate(combos):
        params = dict(zip(keys, combo))
        scores = cross_validate_xgb(X, y, params)
        all_results.append({**params, **scores})
        print(f"  [{i+1}/{len(combos)}] "
              f"max_depth={params['max_depth']}, "
              f"lr={params['learning_rate']}, "
              f"n_est={params['n_estimators']}, "
              f"min_child_weight={params['min_child_weight']} "
              f"→ logloss={scores['logloss_mean']:.4f} ± {scores['logloss_std']:.4f}, "
              f"acc={scores['accuracy_mean']:.3f}")

    # Best = lowest mean logloss
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

    feature_cols = meta['taker_features_loo'] + meta['context_features']
    target_col = meta['target_columns']['zone_6class']
    print(f"  Features: {len(feature_cols)}")

    # ----- Prepare X, y -----
    X_train, y_train, zone_encoder, train_on_target = prepare_xy(train, feature_cols, target_col)
    X_test, y_test, _, test_on_target = prepare_xy(test, feature_cols, target_col, zone_encoder)
    print(f"\nAfter filtering to on-target:")
    print(f"  Train: {len(X_train)} rows ({len(train) - len(X_train)} off-target dropped)")
    print(f"  Test:  {len(X_test)} rows ({len(test) - len(X_test)} off-target dropped)")

    print(f"\nZone class encoding (encoder order):")
    for c, name in enumerate(zone_encoder.classes_):
        count = (y_train == c).sum()
        print(f"  {c}: {name} ({count} train rows)")

    # ----- Baselines -----
    print("\n--- Baselines (on training data, in-sample) ---")
    pop_proba_train = population_baseline_proba(len(X_train), meta_zone_dist(), zone_encoder)
    taker_proba_train = taker_prior_baseline_proba(train_on_target, zone_encoder)

    pop_logloss = log_loss(y_train, pop_proba_train, labels=np.arange(len(ZONES)))
    pop_acc = accuracy_score(y_train, pop_proba_train.argmax(axis=1))
    taker_logloss = log_loss(y_train, taker_proba_train, labels=np.arange(len(ZONES)))
    taker_acc = accuracy_score(y_train, taker_proba_train.argmax(axis=1))

    print(f"  Population baseline:  logloss={pop_logloss:.4f}, accuracy={pop_acc:.3f}")
    print(f"  Taker-prior baseline: logloss={taker_logloss:.4f}, accuracy={taker_acc:.3f}")

    # ----- Grid search -----
    best_params, all_results = grid_search(X_train, y_train, PARAM_GRID)
    print(f"\nBest params: {best_params}")

    # ----- Refit best on full training set -----
    print("\nRefitting best model on full training set...")
    model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=len(ZONES),
        eval_metric='mlogloss',
        tree_method='hist',
        random_state=RANDOM_STATE,
        **best_params,
    )
    model.fit(X_train, y_train, verbose=False)

    # ----- Calibration: fit on out-of-fold predictions -----
    print("\nGenerating out-of-fold predictions for calibration...")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    oof_proba = np.zeros((len(X_train), len(ZONES)))
    for train_idx, val_idx in skf.split(X_train, y_train):
        m = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=len(ZONES),
            eval_metric='mlogloss',
            tree_method='hist',
            random_state=RANDOM_STATE,
            **best_params,
        )
        m.fit(X_train.iloc[train_idx], y_train[train_idx], verbose=False)
        oof_proba[val_idx] = m.predict_proba(X_train.iloc[val_idx])

    calibrator = IsotonicMultiClassCalibrator()
    calibrator.fit(oof_proba, y_train, n_classes=len(ZONES))
    print("Calibrator fitted.")

    # ----- Test set evaluation -----
    print("\n--- Test set evaluation (2022 World Cup) ---")
    test_proba_raw = model.predict_proba(X_test)
    test_proba_cal = calibrator.predict_proba(test_proba_raw)

    test_logloss_raw = log_loss(y_test, test_proba_raw, labels=np.arange(len(ZONES)))
    test_logloss_cal = log_loss(y_test, test_proba_cal, labels=np.arange(len(ZONES)))
    test_acc_raw = accuracy_score(y_test, test_proba_raw.argmax(axis=1))
    test_acc_cal = accuracy_score(y_test, test_proba_cal.argmax(axis=1))

    print(f"  Raw model:        logloss={test_logloss_raw:.4f}, accuracy={test_acc_raw:.3f}")
    print(f"  Calibrated model: logloss={test_logloss_cal:.4f}, accuracy={test_acc_cal:.3f}")

    # Population baseline on test for comparison
    pop_proba_test = population_baseline_proba(len(X_test), meta_zone_dist(), zone_encoder)
    taker_proba_test = taker_prior_baseline_proba(test_on_target, zone_encoder)
    pop_logloss_test = log_loss(y_test, pop_proba_test, labels=np.arange(len(ZONES)))
    taker_logloss_test = log_loss(y_test, taker_proba_test, labels=np.arange(len(ZONES)))
    print(f"  Population baseline: logloss={pop_logloss_test:.4f}")
    print(f"  Taker-prior baseline: logloss={taker_logloss_test:.4f}")

    # ----- Confusion matrix -----
    cm = confusion_matrix(y_test, test_proba_cal.argmax(axis=1),
                          labels=np.arange(len(ZONES)))
    print(f"\nConfusion matrix (rows=actual, cols=predicted):")
    print("       " + "  ".join(f"{z:>4}" for z in zone_encoder.classes_))
    for i, z in enumerate(zone_encoder.classes_):
        print(f"  {z:>4} " + "  ".join(f"{cm[i, j]:>4}" for j in range(len(ZONES))))

    # ----- Feature importance -----
    importance = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    print(f"\nTop 10 features by importance:")
    for name, val in importance.head(10).items():
        print(f"  {name:40} {val:.4f}")

    # ----- Save artifacts -----
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump({
        'model': model,
        'zone_encoder': zone_encoder,
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
            'taker_prior_logloss_train': float(taker_logloss),
            'population_logloss_test': float(pop_logloss_test),
            'taker_prior_logloss_test': float(taker_logloss_test),
        },
        'test': {
            'logloss_raw': float(test_logloss_raw),
            'logloss_calibrated': float(test_logloss_cal),
            'accuracy_raw': float(test_acc_raw),
            'accuracy_calibrated': float(test_acc_cal),
            'confusion_matrix': cm.tolist(),
            'classes': zone_encoder.classes_.tolist(),
        },
        'feature_importance': importance.to_dict(),
    }
    with open(METRICS_FILE, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PHASE 3 STEP 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Model:      {MODEL_FILE}")
    print(f"  Calibrator: {CALIB_FILE}")
    print(f"  Metrics:    {METRICS_FILE}")


def meta_zone_dist():
    """Read the population zone distribution from priors.json."""
    with open("data/processed/priors.json") as f:
        priors = json.load(f)
    return priors['zone_distribution']


if __name__ == "__main__":
    main()

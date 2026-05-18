"""
Phase 3 - Step 1: Build model-ready feature matrices.

Input:
    data/processed/penalties_clean.csv
    data/processed/taker_profiles.csv
    data/processed/keeper_profiles.csv
    data/processed/priors.json

Output:
    data/processed/train_features.parquet
    data/processed/test_features.parquet
    data/processed/feature_metadata.json   (column descriptions, used downstream)

Pipeline:
1. Merge WAYWARD outcomes into WIDE (only 2 cases).
2. For each penalty, recompute taker and keeper profiles using
   leave-one-out (LOO) smoothing to prevent target leakage.
3. Add engineered features (entropy, pressure flag, matchup signals).
4. One-hot / label encode categoricals.
5. Hold out 2022 FIFA World Cup as the test set; everything else trains.
6. Save as Parquet for fast loading by the model training scripts.

Usage:
    python src/features/build_features.py
"""

import os
import json
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy

# ----- Paths -----
CLEAN_FILE = "data/processed/penalties_clean.csv"
TAKER_FILE = "data/processed/taker_profiles.csv"
KEEPER_FILE = "data/processed/keeper_profiles.csv"
PRIORS_FILE = "data/processed/priors.json"

OUTPUT_DIR = "data/processed"
TRAIN_FILE = os.path.join(OUTPUT_DIR, "train_features.parquet")
TEST_FILE = os.path.join(OUTPUT_DIR, "test_features.parquet")
META_FILE = os.path.join(OUTPUT_DIR, "feature_metadata.json")

# ----- Config (must match build_profiles.py) -----
SMOOTHING_K = 10
ZONES = ['TL', 'TC', 'TR', 'BL', 'BC', 'BR']
TEST_COMPETITION = "FIFA World Cup"
TEST_YEAR = 2022


# ============================================================
# SMOOTHING (duplicated from build_profiles.py for self-containment)
# ============================================================

def beta_smooth(successes, total, prior_rate, k=SMOOTHING_K):
    alpha = prior_rate * k
    beta = (1 - prior_rate) * k
    return (alpha + successes) / (alpha + beta + total)


def dirichlet_smooth(counts_dict, prior_dict, k=SMOOTHING_K):
    total = sum(counts_dict.values())
    return {
        cat: (prior_dict[cat] * k + counts_dict.get(cat, 0)) / (k + total)
        for cat in prior_dict
    }


# ============================================================
# LEAVE-ONE-OUT PROFILE COMPUTATION
# ============================================================

def precompute_taker_totals(df):
    """
    For each taker, compute totals we'll need to do LOO efficiently.
    Returns a dict: taker_id -> {field: total}
    """
    totals = {}
    for taker_id, group in df.groupby('taker_id'):
        if pd.isna(taker_id):
            continue
        on_target = group[group['zone'].isin(ZONES)]
        totals[taker_id] = {
            'n': len(group),
            'goals': int((group['outcome_category'] == 'GOAL').sum()),
            'saved_against': int((group['outcome_category'] == 'SAVED').sum()),
            'shootout_n': int(group['is_shootout'].sum()),
            'shootout_goals': int(((group['outcome_category'] == 'GOAL') & group['is_shootout']).sum()),
            'in_game_n': int((~group['is_shootout']).sum()),
            'in_game_goals': int(((group['outcome_category'] == 'GOAL') & ~group['is_shootout']).sum()),
            'right_foot_n': int(group['is_right_foot'].sum()),
            'left_foot_n': int(group['is_left_foot'].sum()),
            'zone_counts': {z: int((on_target['zone'] == z).sum()) for z in ZONES},
            'must_score_n': int(((group['shootout_pressure'] == 'MUST_SCORE')).sum()),
            'must_score_goals': int(
                ((group['shootout_pressure'] == 'MUST_SCORE') &
                 (group['outcome_category'] == 'GOAL')).sum()
            ),
        }
    return totals


def precompute_keeper_totals(df):
    """Same trick for keepers."""
    totals = {}
    for keeper_id, group in df.groupby('keeper_id'):
        if pd.isna(keeper_id):
            continue
        on_target = group[group['zone'].isin(ZONES)]
        zone_stats = {}
        for z in ZONES:
            z_shots = on_target[on_target['zone'] == z]
            zone_stats[z] = {
                'n': len(z_shots),
                'saves': int((z_shots['outcome_category'] == 'SAVED').sum()),
            }
        totals[keeper_id] = {
            'n': len(group),
            'saves': int((group['outcome_category'] == 'SAVED').sum()),
            'goals_conceded': int((group['outcome_category'] == 'GOAL').sum()),
            'shootout_n': int(group['is_shootout'].sum()),
            'shootout_saves': int(((group['outcome_category'] == 'SAVED') & group['is_shootout']).sum()),
            'in_game_n': int((~group['is_shootout']).sum()),
            'in_game_saves': int(((group['outcome_category'] == 'SAVED') & ~group['is_shootout']).sum()),
            'zone_stats': zone_stats,
        }
    return totals


def compute_loo_taker_features(row, taker_totals, priors, k=SMOOTHING_K):
    """
    Compute taker profile features for ONE penalty row,
    using all of the taker's OTHER penalties (leave-one-out).
    """
    taker_id = row['taker_id']
    if pd.isna(taker_id) or taker_id not in taker_totals:
        # Cold start — use priors directly
        feats = {
            'taker_total_penalties_loo': 0,
            'taker_conversion_rate_loo': priors['conversion_rate'],
            'taker_shootout_conversion_rate_loo': priors['shootout_conversion_rate'],
            'taker_in_game_conversion_rate_loo': priors['in_game_conversion_rate'],
            'taker_right_foot_pct_loo': priors['right_foot_rate'],
            'taker_left_foot_pct_loo': priors['left_foot_rate'],
            'taker_must_score_conversion_rate_loo': 0.50,
        }
        for z in ZONES:
            feats[f'taker_zone_{z}_prob_loo'] = priors['zone_distribution'][z]
        return feats

    t = taker_totals[taker_id]

    # Subtract this row's contribution (LOO)
    n = t['n'] - 1
    goals = t['goals'] - int(row['outcome_category'] == 'GOAL')
    shootout_n = t['shootout_n'] - int(bool(row['is_shootout']))
    shootout_goals = t['shootout_goals'] - int(
        (row['outcome_category'] == 'GOAL') and bool(row['is_shootout'])
    )
    in_game_n = t['in_game_n'] - int(not bool(row['is_shootout']))
    in_game_goals = t['in_game_goals'] - int(
        (row['outcome_category'] == 'GOAL') and not bool(row['is_shootout'])
    )
    right_foot_n = t['right_foot_n'] - int(bool(row.get('is_right_foot', False)))
    left_foot_n = t['left_foot_n'] - int(bool(row.get('is_left_foot', False)))

    # Zone counts (LOO)
    row_zone = row['zone']
    zone_counts = dict(t['zone_counts'])
    if row_zone in ZONES:
        zone_counts[row_zone] -= 1

    # Must-score (LOO)
    is_must_score = row['shootout_pressure'] == 'MUST_SCORE'
    must_score_n = t['must_score_n'] - int(is_must_score)
    must_score_goals = t['must_score_goals'] - int(
        is_must_score and (row['outcome_category'] == 'GOAL')
    )

    # Build smoothed features
    feats = {
        'taker_total_penalties_loo': n,
        'taker_conversion_rate_loo': beta_smooth(goals, n, priors['conversion_rate'], k),
        'taker_shootout_conversion_rate_loo': beta_smooth(
            shootout_goals, shootout_n, priors['shootout_conversion_rate'], k
        ),
        'taker_in_game_conversion_rate_loo': beta_smooth(
            in_game_goals, in_game_n, priors['in_game_conversion_rate'], k
        ),
        'taker_right_foot_pct_loo': (right_foot_n / n) if n > 0 else priors['right_foot_rate'],
        'taker_left_foot_pct_loo': (left_foot_n / n) if n > 0 else priors['left_foot_rate'],
        'taker_must_score_conversion_rate_loo': beta_smooth(
            must_score_goals, must_score_n, 0.50, k
        ),
    }

    smoothed_zones = dirichlet_smooth(zone_counts, priors['zone_distribution'], k=k)
    for z in ZONES:
        feats[f'taker_zone_{z}_prob_loo'] = smoothed_zones[z]

    return feats


def compute_loo_keeper_features(row, keeper_totals, priors, k=SMOOTHING_K):
    """Compute keeper profile features for one row, leave-one-out."""
    keeper_id = row['keeper_id']
    if pd.isna(keeper_id) or keeper_id not in keeper_totals:
        feats = {
            'keeper_total_penalties_faced_loo': 0,
            'keeper_save_rate_loo': priors['save_rate'],
            'keeper_shootout_save_rate_loo': 1 - priors['shootout_conversion_rate'],
            'keeper_in_game_save_rate_loo': 1 - priors['in_game_conversion_rate'],
        }
        for z in ZONES:
            feats[f'keeper_save_rate_{z}_loo'] = priors['save_rate_by_zone'][z]
        return feats

    k_totals = keeper_totals[keeper_id]

    n = k_totals['n'] - 1
    saves = k_totals['saves'] - int(row['outcome_category'] == 'SAVED')
    shootout_n = k_totals['shootout_n'] - int(bool(row['is_shootout']))
    shootout_saves = k_totals['shootout_saves'] - int(
        (row['outcome_category'] == 'SAVED') and bool(row['is_shootout'])
    )
    in_game_n = k_totals['in_game_n'] - int(not bool(row['is_shootout']))
    in_game_saves = k_totals['in_game_saves'] - int(
        (row['outcome_category'] == 'SAVED') and not bool(row['is_shootout'])
    )

    feats = {
        'keeper_total_penalties_faced_loo': n,
        'keeper_save_rate_loo': beta_smooth(saves, n, priors['save_rate'], k),
        'keeper_shootout_save_rate_loo': beta_smooth(
            shootout_saves, shootout_n, 1 - priors['shootout_conversion_rate'], k
        ),
        'keeper_in_game_save_rate_loo': beta_smooth(
            in_game_saves, in_game_n, 1 - priors['in_game_conversion_rate'], k
        ),
    }

    # Zone-specific save rates (LOO)
    row_zone = row['zone']
    for z in ZONES:
        z_stats = k_totals['zone_stats'][z]
        z_n = z_stats['n']
        z_saves = z_stats['saves']
        if z == row_zone:
            z_n -= 1
            z_saves -= int(row['outcome_category'] == 'SAVED')
        feats[f'keeper_save_rate_{z}_loo'] = beta_smooth(
            z_saves, z_n, priors['save_rate_by_zone'][z], k
        )

    return feats


# ============================================================
# DERIVED / ENGINEERED FEATURES
# ============================================================

def add_engineered_features(df):
    """Add derived columns that combine signals or capture intuition."""

    # High-pressure flag
    df['is_high_pressure'] = (
        (df['shootout_pressure'] == 'MUST_SCORE') |
        ((df['game_state'] == 'TRAILING') & (df['minute'] >= 80))
    ).astype(int)

    # Zone entropy: how predictable is this taker's placement?
    zone_cols = [f'taker_zone_{z}_prob_loo' for z in ZONES]
    df['taker_zone_entropy_loo'] = df[zone_cols].apply(
        lambda r: scipy_entropy(r.values + 1e-12), axis=1
    )

    # Career age (years between first penalty and this one)
    # Using last_year as a proxy; for LOO we'd need first_year per taker
    # but it's a static field and doesn't leak the outcome.
    # If the taker isn't in the static taker df, leave as 0.
    df['taker_career_age_years'] = (df['year'] - df['taker_first_year_static']).fillna(0)

    return df


def add_zone_features(df):
    """Add zone-derived flags for shots that are on-target."""
    df['zone_is_corner'] = df['zone'].isin(['TL', 'TR', 'BL', 'BR']).astype(int)
    df['zone_is_top'] = df['zone'].isin(['TL', 'TC', 'TR']).astype(int)
    df['zone_is_left'] = df['zone'].isin(['TL', 'BL']).astype(int)
    df['zone_is_right'] = df['zone'].isin(['TR', 'BR']).astype(int)
    df['zone_is_center'] = df['zone'].isin(['TC', 'BC']).astype(int)

    # Pick the keeper's save rate AT the actual zone of this shot
    def pick_keeper_save_at_zone(row):
        z = row['zone']
        if z in ZONES:
            return row[f'keeper_save_rate_{z}_loo']
        return np.nan  # off-target shots

    df['keeper_save_rate_at_zone_loo'] = df.apply(pick_keeper_save_at_zone, axis=1)

    # Matchup gap: how much does the taker's zone preference exceed the keeper's save rate there?
    def pick_taker_zone_prob(row):
        z = row['zone']
        if z in ZONES:
            return row[f'taker_zone_{z}_prob_loo']
        return np.nan

    df['taker_zone_prob_at_zone_loo'] = df.apply(pick_taker_zone_prob, axis=1)
    df['matchup_gap_loo'] = df['taker_zone_prob_at_zone_loo'] - df['keeper_save_rate_at_zone_loo']

    return df


# ============================================================
# CATEGORICAL ENCODING
# ============================================================

GAME_STATE_MAP = {'LEADING': 0, 'LEVEL': 1, 'TRAILING': 2, 'SHOOTOUT': 3, 'UNKNOWN': -1}
PRESSURE_MAP = {'CAN_WIN': 0, 'STANDARD': 1, 'MUST_SCORE': 2, 'N/A': -1, 'UNKNOWN': -1}


def encode_categoricals(df):
    df['game_state_code'] = df['game_state'].map(GAME_STATE_MAP).fillna(-1).astype(int)
    df['shootout_pressure_code'] = df['shootout_pressure'].map(PRESSURE_MAP).fillna(-1).astype(int)
    df['is_shootout_int'] = df['is_shootout'].astype(int)
    df['is_left_foot_int'] = df['is_left_foot'].astype(int)
    df['is_right_foot_int'] = df['is_right_foot'].astype(int)
    return df


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading inputs...")
    df = pd.read_csv(CLEAN_FILE)
    takers = pd.read_csv(TAKER_FILE)
    keepers = pd.read_csv(KEEPER_FILE)
    with open(PRIORS_FILE) as f:
        priors = json.load(f)
    print(f"  {len(df)} penalties, {len(takers)} takers, {len(keepers)} keepers")

    # ----- Merge WAYWARD into WIDE -----
    n_wayward = (df['outcome_category'] == 'WAYWARD').sum()
    df['outcome_category'] = df['outcome_category'].replace('WAYWARD', 'WIDE')
    print(f"\nMerged {n_wayward} WAYWARD rows into WIDE")

    # ----- Bring static taker fields (career start year, etc.) -----
    df = df.merge(
        takers[['taker_id', 'first_year']].rename(columns={'first_year': 'taker_first_year_static'}),
        on='taker_id', how='left'
    )

    # ----- Precompute taker/keeper totals for LOO -----
    print("\nPrecomputing per-player totals (for fast LOO)...")
    taker_totals = precompute_taker_totals(df)
    keeper_totals = precompute_keeper_totals(df)

    # ----- Build LOO features per row -----
    print("\nComputing leave-one-out features for each penalty...")
    loo_feature_rows = []
    for idx, row in df.iterrows():
        t_feats = compute_loo_taker_features(row, taker_totals, priors)
        k_feats = compute_loo_keeper_features(row, keeper_totals, priors)
        combined = {**t_feats, **k_feats}
        loo_feature_rows.append(combined)
        # progress indicator every 200 rows
        if idx % 200 == 0:
            print(f"    {idx}/{len(df)}")

    loo_df = pd.DataFrame(loo_feature_rows)
    df = pd.concat([df.reset_index(drop=True), loo_df.reset_index(drop=True)], axis=1)

    # ----- Engineered features -----
    print("\nAdding engineered features...")
    df = add_engineered_features(df)
    df = add_zone_features(df)
    df = encode_categoricals(df)

    # ----- Train / test split -----
    test_mask = (df['competition'] == TEST_COMPETITION) & (df['year'] == TEST_YEAR)
    train_df = df[~test_mask].copy().reset_index(drop=True)
    test_df = df[test_mask].copy().reset_index(drop=True)

    # ----- Save -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    train_df.to_parquet(TRAIN_FILE, index=False)
    test_df.to_parquet(TEST_FILE, index=False)

    # ----- Feature metadata for downstream scripts -----
    metadata = {
        'target_columns': {
            'outcome_5class': 'outcome_category',   # for model 2
            'zone_6class': 'zone',                  # for model 1 (only on-target rows)
        },
        'taker_features_loo': [
            'taker_total_penalties_loo',
            'taker_conversion_rate_loo',
            'taker_shootout_conversion_rate_loo',
            'taker_in_game_conversion_rate_loo',
            'taker_must_score_conversion_rate_loo',
            'taker_right_foot_pct_loo',
            'taker_left_foot_pct_loo',
            'taker_zone_entropy_loo',
            'taker_career_age_years',
        ] + [f'taker_zone_{z}_prob_loo' for z in ZONES],
        'keeper_features_loo': [
            'keeper_total_penalties_faced_loo',
            'keeper_save_rate_loo',
            'keeper_shootout_save_rate_loo',
            'keeper_in_game_save_rate_loo',
        ] + [f'keeper_save_rate_{z}_loo' for z in ZONES],
        'context_features': [
            'is_shootout_int',
            'shootout_pressure_code',
            'game_state_code',
            'score_diff_at_penalty',
            'minute',
            'is_high_pressure',
            'is_left_foot_int',
            'is_right_foot_int',
        ],
        'zone_features': [
            'zone_is_corner', 'zone_is_top', 'zone_is_left',
            'zone_is_right', 'zone_is_center',
            'keeper_save_rate_at_zone_loo',
            'taker_zone_prob_at_zone_loo',
            'matchup_gap_loo',
        ],
        'rows_train': len(train_df),
        'rows_test': len(test_df),
        'outcome_classes': sorted(df['outcome_category'].dropna().unique().tolist()),
        'zone_classes': ZONES,
    }
    with open(META_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'=' * 60}")
    print(f"PHASE 3 STEP 1 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Train: {len(train_df)} rows → {TRAIN_FILE}")
    print(f"  Test:  {len(test_df)} rows → {TEST_FILE}")
    print(f"  Metadata: {META_FILE}")
    print(f"\nOutcome distribution (train):")
    print(train_df['outcome_category'].value_counts().to_string())
    print(f"\nOutcome distribution (test):")
    print(test_df['outcome_category'].value_counts().to_string())

    # Quick sanity check on LOO behaviour
    print(f"\nSanity check — Messi's LOO conversion rate across his penalties:")
    messi = train_df[train_df['taker_name'].str.contains('Messi', case=False, na=False)]
    if len(messi) > 0:
        print(f"  Sample size in this row's LOO total: "
              f"{messi['taker_total_penalties_loo'].min()} - {messi['taker_total_penalties_loo'].max()}")
        print(f"  LOO conversion rates range: "
              f"{messi['taker_conversion_rate_loo'].min():.3f} - {messi['taker_conversion_rate_loo'].max():.3f}")
        print(f"  (Should always be N-1 of his total, and rates barely vary)")


if __name__ == "__main__":
    main()

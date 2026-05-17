"""
Phase 2 - Step 3: Build per-player profiles with Bayesian smoothing.

Input:  data/processed/penalties_clean.csv
Output:
    data/processed/taker_profiles.csv
    data/processed/keeper_profiles.csv
    data/processed/priors.json  (population-level stats used for smoothing)

For each unique taker and keeper, computes:
- Raw stats (penalty count, goals, etc.)
- Smoothed rates using Beta/Dirichlet priors
- Context splits (shootout vs in-game)
- Reliability flag based on sample size

Smoothing strength (alpha/k) defaults to 10 — meaning the prior is worth
~10 phantom observations. Larger k = more conservative (data must overwhelm
the prior); smaller k = trust the player's own data faster.

Usage:
    python src/data/build_profiles.py
"""

import os
import json
import pandas as pd
import numpy as np

# ----- Config -----
CLEAN_FILE = "data/processed/penalties_clean.csv"
OUTPUT_DIR = "data/processed"
TAKER_FILE = os.path.join(OUTPUT_DIR, "taker_profiles.csv")
KEEPER_FILE = os.path.join(OUTPUT_DIR, "keeper_profiles.csv")
PRIORS_FILE = os.path.join(OUTPUT_DIR, "priors.json")

# Smoothing strength (treats prior as worth K phantom observations)
SMOOTHING_K = 10

# Zones we care about (on-target only)
ZONES = ['TL', 'TC', 'TR', 'BL', 'BC', 'BR']

# Reliability thresholds (in penalty count)
RELIABLE_THRESHOLD = 10
SOMEWHAT_RELIABLE_THRESHOLD = 3


# ============================================================
# SMOOTHING FUNCTIONS
# ============================================================

def beta_smooth(successes, total, prior_rate, k=SMOOTHING_K):
    """
    Beta-smoothed rate for a single probability (e.g., conversion rate).
    
    Equivalent to assuming we've already seen `k` phantom observations
    with the prior_rate, then adding the real (successes, total).
    """
    alpha = prior_rate * k
    beta = (1 - prior_rate) * k
    return (alpha + successes) / (alpha + beta + total)


def dirichlet_smooth(counts_by_category, prior_distribution, k=SMOOTHING_K):
    """
    Dirichlet-smoothed multinomial distribution.
    
    counts_by_category: dict {category: observed_count}
    prior_distribution: dict {category: prior_prob} (must sum to 1)
    Returns: dict {category: smoothed_prob}
    """
    total_observed = sum(counts_by_category.values())
    smoothed = {}
    for cat, prior_p in prior_distribution.items():
        alpha = prior_p * k
        observed = counts_by_category.get(cat, 0)
        smoothed[cat] = (alpha + observed) / (k + total_observed)
    return smoothed


# ============================================================
# PRIORS (population-level statistics)
# ============================================================

def compute_priors(df):
    """Compute population-level priors used to smooth individual profiles."""
    on_target = df[df['zone'].isin(ZONES)]
    
    priors = {
        # Overall outcome rates
        'conversion_rate': (df['outcome_category'] == 'GOAL').mean(),
        'save_rate': (df['outcome_category'] == 'SAVED').mean(),
        'post_rate': (df['outcome_category'] == 'POST').mean(),
        'miss_rate': df['outcome_category'].isin(['WIDE', 'OVER', 'WAYWARD']).mean(),
        
        # Zone distribution (where do takers shoot?)
        'zone_distribution': {
            zone: (on_target['zone'] == zone).mean()
            for zone in ZONES
        },
        
        # Save rate per zone (how often does each zone get saved?)
        'save_rate_by_zone': {
            zone: (on_target[on_target['zone'] == zone]['outcome_category'] == 'SAVED').mean()
            for zone in ZONES
        },
        
        # Context-specific conversion rates
        'shootout_conversion_rate': (
            df[df['is_shootout']]['outcome_category'] == 'GOAL'
        ).mean(),
        'in_game_conversion_rate': (
            df[~df['is_shootout']]['outcome_category'] == 'GOAL'
        ).mean(),
        
        # Foot usage
        'right_foot_rate': df['is_right_foot'].mean(),
        'left_foot_rate': df['is_left_foot'].mean(),
    }
    
    return priors


# ============================================================
# TAKER PROFILES
# ============================================================

def build_taker_profile(taker_pens, priors, k=SMOOTHING_K):
    """Build a single taker's profile from their penalty rows."""
    n = len(taker_pens)
    on_target = taker_pens[taker_pens['zone'].isin(ZONES)]
    
    # ----- Identity -----
    profile = {
        'taker_id': taker_pens['taker_id'].iloc[0],
        'taker_name': taker_pens['taker_name'].iloc[0],
        'last_team': taker_pens.sort_values('year')['taker_team'].iloc[-1],
        
        # ----- Sample size -----
        'total_penalties': n,
        'first_year': int(taker_pens['year'].min()),
        'last_year': int(taker_pens['year'].max()),
    }
    
    # ----- Reliability flag -----
    if n >= RELIABLE_THRESHOLD:
        profile['reliability'] = 'high'
    elif n >= SOMEWHAT_RELIABLE_THRESHOLD:
        profile['reliability'] = 'medium'
    else:
        profile['reliability'] = 'low'
    
    # ----- Raw outcome rates -----
    goals = (taker_pens['outcome_category'] == 'GOAL').sum()
    saved = (taker_pens['outcome_category'] == 'SAVED').sum()
    profile['raw_conversion_rate'] = goals / n
    
    # ----- Smoothed outcome rates -----
    profile['conversion_rate'] = beta_smooth(
        successes=goals, total=n,
        prior_rate=priors['conversion_rate'], k=k
    )
    profile['save_rate_against'] = beta_smooth(
        successes=saved, total=n,
        prior_rate=priors['save_rate'], k=k
    )
    
    # ----- Zone preferences (Dirichlet-smoothed) -----
    zone_counts = {zone: int((on_target['zone'] == zone).sum()) for zone in ZONES}
    smoothed_zones = dirichlet_smooth(zone_counts, priors['zone_distribution'], k=k)
    for zone in ZONES:
        profile[f'zone_{zone}_prob'] = smoothed_zones[zone]
        profile[f'zone_{zone}_raw_count'] = zone_counts[zone]
    
    # ----- Foot usage -----
    profile['right_foot_pct'] = taker_pens['is_right_foot'].mean()
    profile['left_foot_pct'] = taker_pens['is_left_foot'].mean()
    profile['preferred_foot'] = (
        'Right' if profile['right_foot_pct'] > profile['left_foot_pct']
        else 'Left'
    )
    
    # ----- Context splits (smoothed) -----
    shootouts = taker_pens[taker_pens['is_shootout']]
    in_game = taker_pens[~taker_pens['is_shootout']]
    
    profile['shootout_count'] = len(shootouts)
    profile['in_game_count'] = len(in_game)
    
    profile['shootout_conversion_rate'] = beta_smooth(
        successes=(shootouts['outcome_category'] == 'GOAL').sum(),
        total=len(shootouts),
        prior_rate=priors['shootout_conversion_rate'], k=k
    )
    profile['in_game_conversion_rate'] = beta_smooth(
        successes=(in_game['outcome_category'] == 'GOAL').sum(),
        total=len(in_game),
        prior_rate=priors['in_game_conversion_rate'], k=k
    )
    
    # ----- Pressure performance (shootout MUST_SCORE specifically) -----
    must_score = shootouts[shootouts['shootout_pressure'] == 'MUST_SCORE']
    if len(must_score) > 0:
        profile['must_score_count'] = len(must_score)
        profile['must_score_conversion_rate'] = beta_smooth(
            successes=(must_score['outcome_category'] == 'GOAL').sum(),
            total=len(must_score),
            prior_rate=0.50,  # empirical MUST_SCORE baseline
            k=k
        )
    else:
        profile['must_score_count'] = 0
        profile['must_score_conversion_rate'] = np.nan
    
    return profile


def build_all_taker_profiles(df, priors):
    """Build profiles for every unique taker in the data."""
    profiles = []
    for taker_id, group in df.groupby('taker_id'):
        if pd.isna(taker_id):
            continue
        profiles.append(build_taker_profile(group, priors))
    return pd.DataFrame(profiles)


# ============================================================
# KEEPER PROFILES
# ============================================================

def build_keeper_profile(keeper_pens, priors, k=SMOOTHING_K):
    """Build a single keeper's profile from penalties they faced."""
    n = len(keeper_pens)
    on_target = keeper_pens[keeper_pens['zone'].isin(ZONES)]
    
    profile = {
        'keeper_id': keeper_pens['keeper_id'].iloc[0],
        'keeper_name': keeper_pens['keeper_name'].iloc[0],
        # Defender team = the team that ISN'T the taker_team
        'last_team': keeper_pens.sort_values('year').apply(
            lambda r: r['away_team'] if r['taker_team'] == r['home_team'] else r['home_team'],
            axis=1
        ).iloc[-1] if len(keeper_pens) > 0 else None,
        
        'total_penalties_faced': n,
        'first_year': int(keeper_pens['year'].min()),
        'last_year': int(keeper_pens['year'].max()),
    }
    
    # ----- Reliability -----
    if n >= RELIABLE_THRESHOLD:
        profile['reliability'] = 'high'
    elif n >= SOMEWHAT_RELIABLE_THRESHOLD:
        profile['reliability'] = 'medium'
    else:
        profile['reliability'] = 'low'
    
    # ----- Raw and smoothed save rate -----
    saved = (keeper_pens['outcome_category'] == 'SAVED').sum()
    goals_conceded = (keeper_pens['outcome_category'] == 'GOAL').sum()
    
    profile['raw_save_rate'] = saved / n
    profile['save_rate'] = beta_smooth(
        successes=saved, total=n,
        prior_rate=priors['save_rate'], k=k
    )
    profile['concession_rate'] = beta_smooth(
        successes=goals_conceded, total=n,
        prior_rate=priors['conversion_rate'], k=k
    )
    
    # ----- Save rate by zone (smoothed) -----
    # For each zone: how often does this keeper save shots to that zone?
    for zone in ZONES:
        zone_shots = on_target[on_target['zone'] == zone]
        zone_saves = (zone_shots['outcome_category'] == 'SAVED').sum()
        profile[f'save_rate_{zone}'] = beta_smooth(
            successes=zone_saves,
            total=len(zone_shots),
            prior_rate=priors['save_rate_by_zone'][zone],
            k=k
        )
        profile[f'shots_faced_{zone}'] = len(zone_shots)
    
    # ----- Context splits -----
    shootouts = keeper_pens[keeper_pens['is_shootout']]
    in_game = keeper_pens[~keeper_pens['is_shootout']]
    
    profile['shootouts_faced'] = len(shootouts)
    profile['in_game_faced'] = len(in_game)
    
    profile['shootout_save_rate'] = beta_smooth(
        successes=(shootouts['outcome_category'] == 'SAVED').sum(),
        total=len(shootouts),
        prior_rate=1 - priors['shootout_conversion_rate'],
        k=k
    )
    profile['in_game_save_rate'] = beta_smooth(
        successes=(in_game['outcome_category'] == 'SAVED').sum(),
        total=len(in_game),
        prior_rate=1 - priors['in_game_conversion_rate'],
        k=k
    )
    
    # ----- Inferred dive tendency -----
    # If keeper saves a lot on left side (TL/BL) and concedes on right (TR/BR),
    # they probably dive left more often.
    left_saves = (
        ((on_target['zone'].isin(['TL', 'BL'])) & 
         (on_target['outcome_category'] == 'SAVED'))
    ).sum()
    right_saves = (
        ((on_target['zone'].isin(['TR', 'BR'])) & 
         (on_target['outcome_category'] == 'SAVED'))
    ).sum()
    center_saves = (
        ((on_target['zone'].isin(['TC', 'BC'])) & 
         (on_target['outcome_category'] == 'SAVED'))
    ).sum()
    total_saves = left_saves + right_saves + center_saves
    if total_saves > 0:
        profile['inferred_dive_left_pct'] = left_saves / total_saves
        profile['inferred_dive_right_pct'] = right_saves / total_saves
        profile['inferred_stay_center_pct'] = center_saves / total_saves
    else:
        profile['inferred_dive_left_pct'] = np.nan
        profile['inferred_dive_right_pct'] = np.nan
        profile['inferred_stay_center_pct'] = np.nan
    
    return profile


def build_all_keeper_profiles(df, priors):
    """Build profiles for every unique keeper in the data."""
    profiles = []
    for keeper_id, group in df.groupby('keeper_id'):
        if pd.isna(keeper_id):
            continue
        profiles.append(build_keeper_profile(group, priors))
    return pd.DataFrame(profiles)


# ============================================================
# MAIN
# ============================================================

def main():
    print(f"Loading {CLEAN_FILE}...")
    df = pd.read_csv(CLEAN_FILE)
    print(f"  {len(df)} penalties")
    
    # ----- Compute priors -----
    print("\nComputing population priors...")
    priors = compute_priors(df)
    print(f"  Overall conversion rate: {priors['conversion_rate']:.1%}")
    print(f"  Overall save rate:       {priors['save_rate']:.1%}")
    print(f"  Most common zone:        BL ({priors['zone_distribution']['BL']:.1%})")
    print(f"  Hardest zone to save:    TC ({priors['save_rate_by_zone']['TC']:.1%} saved)")
    
    # Save priors as JSON
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    with open(PRIORS_FILE, 'w') as f:
        json.dump(priors, f, indent=2)
    print(f"  Saved priors to {PRIORS_FILE}")
    
    # ----- Build taker profiles -----
    print("\nBuilding taker profiles...")
    takers = build_all_taker_profiles(df, priors)
    takers = takers.sort_values('total_penalties', ascending=False).reset_index(drop=True)
    takers.to_csv(TAKER_FILE, index=False)
    print(f"  Saved {len(takers)} taker profiles to {TAKER_FILE}")
    
    print("\n  Top 5 takers by sample size:")
    print(takers.head(5)[[
        'taker_name', 'total_penalties', 'raw_conversion_rate', 
        'conversion_rate', 'reliability'
    ]].to_string(index=False))
    
    print("\n  Reliability distribution:")
    print(takers['reliability'].value_counts().to_string())
    
    # ----- Build keeper profiles -----
    print("\nBuilding keeper profiles...")
    keepers = build_all_keeper_profiles(df, priors)
    keepers = keepers.sort_values('total_penalties_faced', ascending=False).reset_index(drop=True)
    keepers.to_csv(KEEPER_FILE, index=False)
    print(f"  Saved {len(keepers)} keeper profiles to {KEEPER_FILE}")
    
    print("\n  Top 5 keepers by sample size:")
    print(keepers.head(5)[[
        'keeper_name', 'total_penalties_faced', 'raw_save_rate',
        'save_rate', 'reliability'
    ]].to_string(index=False))
    
    print("\n  Reliability distribution:")
    print(keepers['reliability'].value_counts().to_string())
    
    # ----- Summary -----
    print(f"\n{'=' * 60}")
    print(f"PHASE 2 STEP 3 COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Takers profiled:  {len(takers)}")
    print(f"  Keepers profiled: {len(keepers)}")
    print(f"  Reliable takers (10+ pens):  {(takers['reliability'] == 'high').sum()}")
    print(f"  Reliable keepers (10+ pens): {(keepers['reliability'] == 'high').sum()}")


if __name__ == "__main__":
    main()

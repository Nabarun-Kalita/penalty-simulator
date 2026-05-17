"""
Phase 2 - Step 1b: Enrich cleaned penalties with score state and pressure.

Input:  data/processed/penalties_clean.csv
Output: data/processed/penalties_clean.csv  (overwrites with new columns)

For each penalty, computes:
- Score for each team at penalty time
- Score differential from taker's perspective (positive = winning)
- Game state category: LEADING, LEVEL, TRAILING
- For shootouts: pressure category (MUST_SCORE, CAN_WIN, STANDARD)

Match events are cached to disk and progress is saved every 50 rows, so
re-runs skip already-enriched rows and resume after a network drop.

Usage:
    python src/data/enrich_score_state.py
"""

import os
import pickle
import pandas as pd
import numpy as np
from statsbombpy import sb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

CLEAN_FILE = "data/processed/penalties_clean.csv"
CACHE_DIR = "data/cache/events"
PROGRESS_FILE = "data/cache/enrich_progress.csv"
SAVE_EVERY = 50  # Save progress every N penalties


def get_events_cached(match_id):
    """Fetch match events from disk cache, or from API and save to disk."""
    cache_path = os.path.join(CACHE_DIR, f"{match_id}.pkl")
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass  # cache file corrupted, refetch
    try:
        events = sb.events(match_id=match_id).reset_index(drop=True)
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(cache_path, 'wb') as f:
            pickle.dump(events, f)
        return events
    except Exception as e:
        print(f"  Failed to fetch match {match_id}: {e}")
        return None


def compute_score_at_event(events_df, target_event_idx, home_team, away_team):
    """Count goals by each team BEFORE the target event, periods 1-4 only."""
    goals_before = events_df.iloc[:target_event_idx]
    goal_events = goals_before[
        (goals_before['type'] == 'Shot') &
        (goals_before['shot_outcome'] == 'Goal') &
        (goals_before['period'] < 5)
    ]
    own_goals = goals_before[goals_before['type'] == 'Own Goal Against']

    home_score = (
        (goal_events['team'] == home_team).sum() +
        (own_goals['team'] == away_team).sum()
    )
    away_score = (
        (goal_events['team'] == away_team).sum() +
        (own_goals['team'] == home_team).sum()
    )
    return int(home_score), int(away_score)


def compute_shootout_state(events_df, target_event_idx, home_team, away_team):
    """Shootout score BEFORE this kick."""
    pens_before = events_df[
        (events_df.index < target_event_idx) &
        (events_df['period'] == 5) &
        (events_df['type'] == 'Shot') &
        (events_df.get('shot_type') == 'Penalty') &
        (events_df['shot_outcome'] == 'Goal')
    ]
    home_score = (pens_before['team'] == home_team).sum()
    away_score = (pens_before['team'] == away_team).sum()
    return int(home_score), int(away_score)


def classify_pressure(taker_score, opponent_score, kick_num_taker):
    """Heuristic shootout pressure classification."""
    diff = taker_score - opponent_score
    if diff < 0 and kick_num_taker >= 4:
        return 'MUST_SCORE'
    if diff > 0 and kick_num_taker >= 4:
        return 'CAN_WIN'
    if diff <= -2:
        return 'MUST_SCORE'
    return 'STANDARD'


def enrich_one_penalty(row, events):
    """Compute all score-state columns for one penalty row."""
    if events is None:
        return {
            'taker_score_at_penalty': np.nan,
            'opponent_score_at_penalty': np.nan,
            'score_diff_at_penalty': np.nan,
            'game_state': 'UNKNOWN',
            'shootout_pressure': 'UNKNOWN',
            'shootout_kick_num': np.nan,
        }

    home_team = row['home_team']
    away_team = row['away_team']
    taker_team = row['taker_team']
    is_shootout = row['is_shootout']

    # Find this penalty's index in events
    matches = events[
        (events['type'] == 'Shot') &
        (events.get('shot_type') == 'Penalty') &
        (events['player'] == row['taker_name']) &
        (events['minute'] == row['minute']) &
        (events.get('second', 0) == row.get('second', 0))
    ]

    if len(matches) == 0:
        return {
            'taker_score_at_penalty': np.nan,
            'opponent_score_at_penalty': np.nan,
            'score_diff_at_penalty': np.nan,
            'game_state': 'UNKNOWN',
            'shootout_pressure': 'UNKNOWN',
            'shootout_kick_num': np.nan,
        }

    target_idx = matches.index[0]

    if is_shootout:
        h_pens, a_pens = compute_shootout_state(events, target_idx, home_team, away_team)
        if taker_team == home_team:
            taker_score, opp_score = h_pens, a_pens
        else:
            taker_score, opp_score = a_pens, h_pens

        shootout_pens = events[
            (events.index <= target_idx) &
            (events['period'] == 5) &
            (events['type'] == 'Shot') &
            (events.get('shot_type') == 'Penalty') &
            (events['team'] == taker_team)
        ]
        kick_num = len(shootout_pens)
        pressure = classify_pressure(taker_score, opp_score, kick_num)
        game_state = 'SHOOTOUT'
    else:
        h_score, a_score = compute_score_at_event(events, target_idx, home_team, away_team)
        if taker_team == home_team:
            taker_score, opp_score = h_score, a_score
        else:
            taker_score, opp_score = a_score, h_score

        diff = taker_score - opp_score
        if diff > 0:
            game_state = 'LEADING'
        elif diff < 0:
            game_state = 'TRAILING'
        else:
            game_state = 'LEVEL'
        pressure = 'N/A'
        kick_num = np.nan

    return {
        'taker_score_at_penalty': taker_score,
        'opponent_score_at_penalty': opp_score,
        'score_diff_at_penalty': taker_score - opp_score,
        'game_state': game_state,
        'shootout_pressure': pressure,
        'shootout_kick_num': kick_num,
    }


def main():
    print(f"Loading {CLEAN_FILE}...")
    df = pd.read_csv(CLEAN_FILE)
    print(f"  {len(df)} penalties")

    new_cols = [
        'taker_score_at_penalty', 'opponent_score_at_penalty', 'score_diff_at_penalty',
        'game_state', 'shootout_pressure', 'shootout_kick_num',
    ]

    # Resume logic:
    # 1. If progress file exists, load enriched columns from it (highest priority)
    # 2. Otherwise, use whatever's already in the CSV (from a previous run without progress file)
    # 3. Only initialize columns to NaN if they don't exist anywhere
    if os.path.exists(PROGRESS_FILE):
        print(f"\nFound progress file — resuming from {PROGRESS_FILE}")
        progress_df = pd.read_csv(PROGRESS_FILE)
        for col in new_cols:
            if col in progress_df.columns:
                df[col] = progress_df[col]
    else:
        # No progress file — keep existing columns in df (from CSV), init missing ones
        for col in new_cols:
            if col not in df.columns:
                df[col] = np.nan

    # Report how many rows are already enriched
    if 'game_state' in df.columns:
        already_done = ((df['game_state'].notna()) & (df['game_state'] != 'UNKNOWN')).sum()
        print(f"  {already_done} of {len(df)} penalties already enriched")
        print(f"  {len(df) - already_done} to process (UNKNOWN or missing)")

    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)

    processed_since_save = 0

    for idx in tqdm(range(len(df)), desc="Enriching"):
        # Skip already-processed rows
        if pd.notna(df.at[idx, 'game_state']) and df.at[idx, 'game_state'] != 'UNKNOWN':
            continue

        row = df.iloc[idx]
        events = get_events_cached(row['match_id'])
        result = enrich_one_penalty(row, events)

        for col, value in result.items():
            df.at[idx, col] = value

        processed_since_save += 1
        if processed_since_save >= SAVE_EVERY:
            df.to_csv(PROGRESS_FILE, index=False)
            processed_since_save = 0

    # Final save
    df.to_csv(PROGRESS_FILE, index=False)
    df.to_csv(CLEAN_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE: Updated {CLEAN_FILE} with score state")
    print(f"{'=' * 60}")

    print("\nGame state distribution:")
    print(df['game_state'].value_counts().to_string())

    print("\nIn-game conversion by game state:")
    in_game = df[~df['is_shootout']]
    print(in_game.groupby('game_state', group_keys=False).apply(
        lambda g: f"{(g['outcome_category'] == 'GOAL').mean():.1%} ({(g['outcome_category'] == 'GOAL').sum()}/{len(g)})",
        include_groups=False
    ).to_string())

    print("\nShootout conversion by pressure:")
    shootouts = df[df['is_shootout']]
    print(shootouts.groupby('shootout_pressure', group_keys=False).apply(
        lambda g: f"{(g['outcome_category'] == 'GOAL').mean():.1%} ({(g['outcome_category'] == 'GOAL').sum()}/{len(g)})",
        include_groups=False
    ).to_string())

    print(f"\nCache directory: {CACHE_DIR}")
    print(f"Cached events: {len(os.listdir(CACHE_DIR)) if os.path.exists(CACHE_DIR) else 0} matches")


if __name__ == "__main__":
    main()

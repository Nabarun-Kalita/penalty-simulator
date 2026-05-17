"""
Phase 2 - Step 1a: Clean and enrich the raw penalty dataset.

Input:  data/raw/penalties_raw.csv
Output: data/processed/penalties_clean.csv

Operations:
1. Drop Women's World Cup penalties
2. Map shot end locations to a 6-zone goal grid (TL, TC, TR, BL, BC, BR)
3. Classify off-target outcomes (WIDE / OVER / POST / WAYWARD / SAVED)
4. Add a unified 'outcome_category' column for modeling
5. Extract year from season
6. Add derived flags (is_left_foot, is_right_foot)

Usage:
    python src/data/clean_penalties.py
"""

import os
import pandas as pd
import numpy as np

# ----- Config -----
INPUT_FILE = "data/raw/penalties_raw.csv"
OUTPUT_DIR = "data/processed"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "penalties_clean.csv")

# StatsBomb goal coordinates
GOAL_Y_MIN = 36.0    # left post
GOAL_Y_MAX = 44.0    # right post
GOAL_HEIGHT = 2.67   # crossbar (≈ 8 feet)

# Zone boundaries within the goal mouth
Y_LEFT_MID = 38.67    # boundary between left and center thirds (36 + 8/3)
Y_MID_RIGHT = 41.33   # boundary between center and right thirds (44 - 8/3)
Z_LOW_HIGH = 1.33     # boundary between low and high halves (2.67 / 2)


def classify_zone(end_y, end_z, shot_outcome):
    """
    Map a shot's end location to a zone label.

    On-target shots (Goal / Saved / Saved to Post) → one of 6 grid zones (TL/TC/TR/BL/BC/BR)
    Off-target → WIDE / OVER / POST / WAYWARD
    """
    # Handle truly missing coords
    if pd.isna(end_y):
        return 'UNKNOWN'

    # Off-target categories take priority
    if shot_outcome == 'Off T':
        # Off target: could be wide or over. Use coords if available.
        if pd.notna(end_z) and end_z > GOAL_HEIGHT:
            return 'OVER'
        if end_y < GOAL_Y_MIN or end_y > GOAL_Y_MAX:
            return 'WIDE'
        # If coords say it's within the goal but outcome says off-target, trust outcome
        return 'WIDE'

    if shot_outcome == 'Post':
        return 'POST'

    if shot_outcome == 'Wayward':
        return 'WAYWARD'

    # On-target: Goal, Saved, Saved to Post — map to 6-zone grid
    # If z is missing, use a default low value (most shots are low)
    z = end_z if pd.notna(end_z) else 0.5

    # Horizontal third
    if end_y < Y_LEFT_MID:
        h_zone = 'L'
    elif end_y > Y_MID_RIGHT:
        h_zone = 'R'
    else:
        h_zone = 'C'

    # Vertical half
    v_zone = 'T' if z > Z_LOW_HIGH else 'B'

    return f"{v_zone}{h_zone}"


def classify_outcome(row):
    """
    Map StatsBomb's shot_outcome to a clean outcome category for modeling.
    
    Categories:
        GOAL    - ball ended in net
        SAVED   - keeper made the save (incl. saved-to-post)
        POST    - hit woodwork, didn't go in (no keeper touch)
        WIDE    - missed wide of the post
        OVER    - missed over the bar
        WAYWARD - very off shots
    """
    outcome = row['shot_outcome']
    zone = row['zone']

    if outcome == 'Goal':
        return 'GOAL'
    if outcome in ('Saved', 'Saved to Post'):
        return 'SAVED'
    if outcome == 'Post':
        return 'POST'
    if outcome == 'Wayward':
        return 'WAYWARD'
    if outcome == 'Off T':
        # Use the zone we already classified
        return 'OVER' if zone == 'OVER' else 'WIDE'
    return 'UNKNOWN'


def extract_year(season_str):
    """Get the year from season strings like '2022', '2018/2019', '2023/2024'."""
    if pd.isna(season_str):
        return np.nan
    s = str(season_str)
    # Take first 4-digit number found
    import re
    match = re.search(r'\d{4}', s)
    return int(match.group()) if match else np.nan


def main():
    print(f"Loading raw data from {INPUT_FILE}...")
    df = pd.read_csv(INPUT_FILE)
    initial_count = len(df)
    print(f"  Loaded {initial_count} penalties")

    # ----- Filter 1: Drop Women's World Cup -----
    womens_mask = df['competition'].str.contains("Women's", case=False, na=False)
    print(f"\nDropping {womens_mask.sum()} Women's World Cup penalties")
    df = df[~womens_mask].reset_index(drop=True)
    print(f"  Remaining: {len(df)}")

    # ----- Extract year -----
    df['year'] = df['season'].apply(extract_year)

    # ----- Zone mapping -----
    print("\nMapping shots to goal zones...")
    df['zone'] = df.apply(
        lambda r: classify_zone(r['end_y'], r['end_z'], r['shot_outcome']),
        axis=1
    )
    print("  Zone distribution:")
    print(df['zone'].value_counts().to_string())

    # ----- Outcome category -----
    print("\nClassifying outcomes...")
    df['outcome_category'] = df.apply(classify_outcome, axis=1)
    print("  Outcome distribution:")
    print(df['outcome_category'].value_counts().to_string())

    # ----- Foot flags -----
    df['is_left_foot'] = df['shot_body_part'] == 'Left Foot'
    df['is_right_foot'] = df['shot_body_part'] == 'Right Foot'

    # ----- Score state -----
    # (We don't have running score per minute, but we can flag final result context)
    # Skipping for now; would need event-by-event score tracking. Done in step 1b.

    # ----- Final summary -----
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE: Saved {len(df)} cleaned penalties to {OUTPUT_FILE}")
    print(f"{'=' * 60}")

    print(f"\nConversion rate: {(df['outcome_category'] == 'GOAL').mean():.1%}")
    print(f"\nBy outcome:")
    print(df['outcome_category'].value_counts(normalize=True).mul(100).round(1).to_string())

    print(f"\nGoal zones (on-target shots only):")
    on_target = df[df['zone'].isin(['TL', 'TC', 'TR', 'BL', 'BC', 'BR'])]
    print(on_target['zone'].value_counts().to_string())

    print(f"\nConversion rate by zone:")
    zone_conv = on_target.groupby('zone').apply(
        lambda g: f"{(g['outcome_category'] == 'GOAL').mean():.1%} ({(g['outcome_category'] == 'GOAL').sum()}/{len(g)})"
    )
    print(zone_conv.to_string())

    print(f"\nYear range: {int(df['year'].min())} – {int(df['year'].max())}")
    print(f"\nFoot usage:")
    print(f"  Right foot: {df['is_right_foot'].sum()}")
    print(f"  Left foot:  {df['is_left_foot'].sum()}")
    print(f"  Other/Unknown: {(~df['is_left_foot'] & ~df['is_right_foot']).sum()}")


if __name__ == "__main__":
    main()

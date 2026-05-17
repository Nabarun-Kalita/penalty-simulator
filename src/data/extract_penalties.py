"""
Phase 1: Extract all penalty kicks from StatsBomb open data.

Saves: data/raw/penalties_raw.csv

Strategy for identifying the keeper:
1. Try the shot's freeze frame (rare for penalties)
2. Fall back to the defending team's lineup (works for almost all cases)
3. Handle keeper substitutions by parsing the time window each keeper played

Usage:
    python src/data/extract_penalties.py
"""

import os
import pandas as pd
from statsbombpy import sb
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ----- Config -----
OUTPUT_DIR = "data/raw"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "penalties_raw.csv")

TARGET_KEYWORDS = [
    "World Cup",
    "UEFA Euro",
    "Champions League",
    "Copa America",
    "African Cup of Nations",
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
]


def get_target_competitions():
    """Return competitions matching our keywords."""
    comps = sb.competitions()
    if TARGET_KEYWORDS is None:
        return comps
    mask = comps['competition_name'].str.contains(
        '|'.join(TARGET_KEYWORDS), case=False, na=False
    )
    return comps[mask].reset_index(drop=True)


def parse_time_str(time_str):
    """Convert 'MM:SS' string to total seconds. Returns None if invalid."""
    if not isinstance(time_str, str) or ':' not in time_str:
        return None
    try:
        parts = time_str.split(':')
        return int(parts[0]) * 60 + int(parts[1])
    except (ValueError, IndexError):
        return None


def get_keeper_from_freeze_frame(freeze_frame):
    """Try to get the opposing keeper from the shot freeze frame."""
    if not isinstance(freeze_frame, list):
        return None, None

    for player in freeze_frame:
        if not isinstance(player, dict):
            continue
        position = player.get('position', {})
        if position.get('name') == 'Goalkeeper' and not player.get('teammate', True):
            player_info = player.get('player', {})
            return player_info.get('id'), player_info.get('name')
    return None, None


def get_keeper_from_lineup(lineups, defending_team, penalty_minute, penalty_second):
    """
    Find the keeper from the defending team's lineup who was on the pitch
    at the penalty's timestamp. Handles keeper substitutions.
    """
    if defending_team not in lineups:
        return None, None

    lineup = lineups[defending_team]
    penalty_seconds = penalty_minute * 60 + (penalty_second or 0)

    candidates = []  # (keeper_id, keeper_name, from_sec, to_sec)

    for _, player in lineup.iterrows():
        positions = player.get('positions', [])
        if not isinstance(positions, list):
            continue

        for pos in positions:
            if pos.get('position') != 'Goalkeeper':
                continue
            from_sec = parse_time_str(pos.get('from')) or 0
            to_sec = parse_time_str(pos.get('to'))
            # to=None means played until end of match
            if to_sec is None:
                to_sec = float('inf')

            candidates.append((
                player.get('player_id'),
                player.get('player_name'),
                from_sec,
                to_sec,
            ))

    # Find the keeper whose time window contains the penalty
    for kid, kname, from_sec, to_sec in candidates:
        if from_sec <= penalty_seconds <= to_sec:
            return kid, kname

    # Fallback: return any keeper from the defending team if no match found
    if candidates:
        return candidates[0][0], candidates[0][1]

    return None, None


def extract_penalties_from_match(match_id, match_info):
    """Pull all penalty events from a single match."""
    try:
        events = sb.events(match_id=match_id)
    except Exception as e:
        print(f"  Skipping match {match_id} (events): {e}")
        return []

    if 'shot_type' not in events.columns:
        return []

    pens = events[(events['type'] == 'Shot') & (events['shot_type'] == 'Penalty')]
    if len(pens) == 0:
        return []

    # Fetch lineups ONCE per match (not per penalty) for efficiency
    try:
        lineups = sb.lineups(match_id=match_id)
    except Exception as e:
        print(f"  Skipping match {match_id} (lineups): {e}")
        lineups = {}

    home_team = match_info.get('home_team')
    away_team = match_info.get('away_team')

    rows = []
    for _, p in pens.iterrows():
        loc = p.get('location') if isinstance(p.get('location'), list) else [None, None]
        end_loc = p.get('shot_end_location') if isinstance(p.get('shot_end_location'), list) else [None, None, None]
        while len(end_loc) < 3:
            end_loc.append(None)

        # Step 1: try freeze frame
        keeper_id, keeper_name = get_keeper_from_freeze_frame(p.get('shot_freeze_frame'))
        keeper_source = 'freeze_frame' if keeper_name else None

        # Step 2: fall back to lineup
        if keeper_name is None:
            taker_team = p.get('team')
            defending_team = away_team if taker_team == home_team else home_team
            keeper_id, keeper_name = get_keeper_from_lineup(
                lineups, defending_team, p.get('minute', 0), p.get('second', 0)
            )
            if keeper_name:
                keeper_source = 'lineup'

        rows.append({
            # Match context
            'match_id': match_id,
            'competition': match_info.get('competition_name'),
            'season': match_info.get('season_name'),
            'home_team': home_team,
            'away_team': away_team,
            'home_score': match_info.get('home_score'),
            'away_score': match_info.get('away_score'),
            # Timing
            'period': p.get('period'),
            'minute': p.get('minute'),
            'second': p.get('second'),
            'is_shootout': p.get('period') == 5,
            # Taker
            'taker_id': p.get('player_id'),
            'taker_name': p.get('player'),
            'taker_team': p.get('team'),
            # Keeper
            'keeper_id': keeper_id,
            'keeper_name': keeper_name,
            'keeper_source': keeper_source,  # 'freeze_frame' or 'lineup'
            # Shot details
            'shot_outcome': p.get('shot_outcome'),
            'shot_body_part': p.get('shot_body_part'),
            'shot_technique': p.get('shot_technique'),
            'shot_xg': p.get('shot_statsbomb_xg'),
            # Locations
            'start_x': loc[0],
            'start_y': loc[1],
            'end_x': end_loc[0],
            'end_y': end_loc[1],
            'end_z': end_loc[2],
            # Outcome
            'is_goal': p.get('shot_outcome') == 'Goal',
        })

    return rows


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Fetching competitions...")
    comps = get_target_competitions()
    print(f"Targeting {len(comps)} competition-seasons")

    all_penalties = []

    for _, comp in tqdm(comps.iterrows(), total=len(comps), desc="Competitions"):
        comp_id = comp['competition_id']
        season_id = comp['season_id']
        comp_name = comp['competition_name']
        season_name = comp['season_name']

        try:
            matches = sb.matches(competition_id=comp_id, season_id=season_id)
        except Exception as e:
            print(f"\nSkipping {comp_name} {season_name}: {e}")
            continue

        for _, match in tqdm(
            matches.iterrows(),
            total=len(matches),
            desc=f"  {comp_name[:25]} {season_name}",
            leave=False,
        ):
            match_info = {
                'competition_name': comp_name,
                'season_name': season_name,
                'home_team': match.get('home_team'),
                'away_team': match.get('away_team'),
                'home_score': match.get('home_score'),
                'away_score': match.get('away_score'),
            }
            penalties = extract_penalties_from_match(match['match_id'], match_info)
            all_penalties.extend(penalties)

    df = pd.DataFrame(all_penalties)
    df.to_csv(OUTPUT_FILE, index=False)

    print(f"\n{'=' * 60}")
    print(f"DONE: Saved {len(df)} penalties to {OUTPUT_FILE}")
    print(f"{'=' * 60}")
    print(f"\nBreakdown:")
    print(f"  Goals:               {df['is_goal'].sum()}")
    print(f"  Shootouts:           {df['is_shootout'].sum()}")
    print(f"  In-game:             {(~df['is_shootout']).sum()}")
    print(f"\nKeeper identification:")
    print(f"  From freeze frame:   {(df['keeper_source'] == 'freeze_frame').sum()}")
    print(f"  From lineup:         {(df['keeper_source'] == 'lineup').sum()}")
    print(f"  Missing keeper:      {df['keeper_name'].isna().sum()}")
    print(f"\nBy competition:")
    print(df.groupby('competition').size().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()

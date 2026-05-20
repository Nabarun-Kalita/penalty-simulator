"""
PenaltySimulator — the prediction engine.

Combines the shot placement model and outcome model into a unified
predictor. Given a taker, keeper, and context, returns a full
prediction including zone distribution, goal probability, and
Monte Carlo simulation results.

Models are loaded from local disk in development; if they're not
present locally, they're downloaded from the HF Model Hub
(intended for deployment).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from scipy.stats import entropy as scipy_entropy
from huggingface_hub import hf_hub_download

from src.models.calibration import IsotonicMultiClassCalibrator  # noqa: F401  (needed for pickle loading)


# ============================================================
# CONSTANTS
# ============================================================

ZONES = ['TL', 'TC', 'TR', 'BL', 'BC', 'BR']
OUTCOME_CLASSES = ['GOAL', 'OVER', 'POST', 'SAVED', 'WIDE']
GAME_STATE_MAP = {'LEADING': 0, 'LEVEL': 1, 'TRAILING': 2, 'SHOOTOUT': 3, 'UNKNOWN': -1}
PRESSURE_MAP = {'CAN_WIN': 0, 'STANDARD': 1, 'MUST_SCORE': 2, 'N/A': -1, 'UNKNOWN': -1}

HF_MODEL_REPO = "nabs7/penalty-simulator-models"


def _resolve_model_path(local_path: str, filename: str) -> str:
    """
    Return a usable filesystem path for a model artifact.

    - If `local_path` exists on disk (local dev), return it.
    - Otherwise download `filename` from HF Model Hub and return the
      cached path. Subsequent calls hit the HF cache.
    """
    if os.path.exists(local_path):
        return local_path
    print(f"  Downloading {filename} from {HF_MODEL_REPO}...")
    cached = hf_hub_download(repo_id=HF_MODEL_REPO, filename=filename)
    print(f"  Cached at {cached}")
    return cached


class PenaltySimulator:
    def __init__(
        self,
        shot_model_path: str = "models/shot_placement_model.pkl",
        shot_calib_path: str = "models/shot_placement_calibrator.pkl",
        outcome_model_path: str = "models/outcome_model.pkl",
        outcome_calib_path: str = "models/outcome_calibrator.pkl",
        taker_profiles_path: str = "data/processed/taker_profiles.csv",
        keeper_profiles_path: str = "data/processed/keeper_profiles.csv",
        priors_path: str = "data/processed/priors.json",
        feature_metadata_path: str = "data/processed/feature_metadata.json",
    ):
        shot_bundle = joblib.load(
            _resolve_model_path(shot_model_path, "shot_placement_model.pkl")
        )
        self.shot_model = shot_bundle['model']
        self.shot_encoder = shot_bundle['zone_encoder']
        self.shot_feature_cols = shot_bundle['feature_cols']
        self.shot_calibrator = joblib.load(
            _resolve_model_path(shot_calib_path, "shot_placement_calibrator.pkl")
        )

        outcome_bundle = joblib.load(
            _resolve_model_path(outcome_model_path, "outcome_model.pkl")
        )
        self.outcome_model = outcome_bundle['model']
        self.outcome_encoder = outcome_bundle['encoder']
        self.outcome_feature_cols = outcome_bundle['feature_cols']
        self.outcome_calibrator = joblib.load(
            _resolve_model_path(outcome_calib_path, "outcome_calibrator.pkl")
        )

        self.takers = pd.read_csv(taker_profiles_path)
        self.keepers = pd.read_csv(keeper_profiles_path)
        with open(priors_path) as f:
            self.priors = json.load(f)
        with open(feature_metadata_path) as f:
            self.meta = json.load(f)

    def simulate(
        self, taker_id, keeper_id, is_shootout=False, shootout_pressure='N/A',
        game_state=None, score_diff_at_penalty=0, minute=60, shootout_kick_num=None,
        is_left_foot=None, is_right_foot=None, n_simulations=10000, random_state=None,
    ) -> dict:
        taker_row = self._lookup_taker(taker_id)
        keeper_row = self._lookup_keeper(keeper_id)

        if game_state is None:
            game_state = 'SHOOTOUT' if is_shootout else 'LEVEL'
        if is_left_foot is None and is_right_foot is None:
            if taker_row is not None and taker_row.get('preferred_foot') == 'Left':
                is_left_foot, is_right_foot = True, False
            else:
                is_left_foot, is_right_foot = False, True
        elif is_left_foot is None:
            is_left_foot = not is_right_foot
        elif is_right_foot is None:
            is_right_foot = not is_left_foot

        context = {
            'is_shootout': is_shootout, 'shootout_pressure': shootout_pressure,
            'game_state': game_state, 'score_diff_at_penalty': score_diff_at_penalty,
            'minute': minute,
            'shootout_kick_num': shootout_kick_num if shootout_kick_num is not None else (3 if is_shootout else -1),
            'is_left_foot': is_left_foot, 'is_right_foot': is_right_foot,
        }

        base_features = self._build_base_features(taker_row, keeper_row, context)
        zone_probs = self._predict_zone_distribution(base_features, taker_row=taker_row)
        outcome_by_zone = self._predict_outcomes_for_all_zones(base_features, taker_row, keeper_row)
        marginal_outcomes = self._marginalize(zone_probs, outcome_by_zone)
        p_goal = marginal_outcomes['GOAL']
        sim_results = self._monte_carlo(zone_probs, outcome_by_zone, n=n_simulations, random_state=random_state)

        return {
            'p_goal': float(p_goal),
            'zone_probs': {z: float(p) for z, p in zone_probs.items()},
            'outcome_by_zone': {
                z: {oc: float(p) for oc, p in zone_outcomes.items()}
                for z, zone_outcomes in outcome_by_zone.items()
            },
            'outcome_probs': {oc: float(p) for oc, p in marginal_outcomes.items()},
            'simulations': sim_results,
            'taker_name': taker_row['taker_name'] if taker_row is not None else f'Unknown ({taker_id})',
            'keeper_name': keeper_row['keeper_name'] if keeper_row is not None else f'Unknown ({keeper_id})',
            'taker_reliability': taker_row['reliability'] if taker_row is not None else 'low',
            'keeper_reliability': keeper_row['reliability'] if keeper_row is not None else 'low',
            'taker_total_penalties': int(taker_row['total_penalties']) if taker_row is not None else 0,
            'keeper_total_penalties': int(keeper_row['total_penalties_faced']) if keeper_row is not None else 0,
            'context': context,
        }

    def list_takers(self) -> pd.DataFrame:
        return self.takers[
            ['taker_id', 'taker_name', 'last_team', 'total_penalties', 'reliability']
        ].sort_values('total_penalties', ascending=False).reset_index(drop=True)

    def list_keepers(self) -> pd.DataFrame:
        return self.keepers[
            ['keeper_id', 'keeper_name', 'last_team', 'total_penalties_faced', 'reliability']
        ].sort_values('total_penalties_faced', ascending=False).reset_index(drop=True)

    def _lookup_taker(self, taker_id):
        match = self.takers[self.takers['taker_id'] == taker_id]
        return match.iloc[0].to_dict() if len(match) > 0 else None

    def _lookup_keeper(self, keeper_id):
        match = self.keepers[self.keepers['keeper_id'] == keeper_id]
        return match.iloc[0].to_dict() if len(match) > 0 else None

    def _build_base_features(self, taker_row, keeper_row, context: dict) -> dict:
        feats = {}
        if taker_row is not None and taker_row['total_penalties'] > 0:
            n_taker = int(taker_row['total_penalties'])
            feats['taker_total_penalties_loo'] = n_taker
            feats['taker_conversion_rate_loo'] = float(taker_row['conversion_rate'])
            feats['taker_shootout_conversion_rate_loo'] = float(taker_row['shootout_conversion_rate'])
            feats['taker_in_game_conversion_rate_loo'] = float(taker_row['in_game_conversion_rate'])
            feats['taker_must_score_conversion_rate_loo'] = float(taker_row['must_score_conversion_rate']) \
                if pd.notna(taker_row['must_score_conversion_rate']) else 0.50
            feats['taker_right_foot_pct_loo'] = float(taker_row['right_foot_pct'])
            feats['taker_left_foot_pct_loo'] = float(taker_row['left_foot_pct'])
            zone_probs_taker = []
            for z in ZONES:
                p = float(taker_row[f'zone_{z}_prob'])
                feats[f'taker_zone_{z}_prob_loo'] = p
                zone_probs_taker.append(p)
            feats['taker_zone_entropy_loo'] = float(scipy_entropy(np.array(zone_probs_taker) + 1e-12))
            feats['taker_career_age_years'] = 0
        else:
            feats['taker_total_penalties_loo'] = 0
            feats['taker_conversion_rate_loo'] = self.priors['conversion_rate']
            feats['taker_shootout_conversion_rate_loo'] = self.priors['shootout_conversion_rate']
            feats['taker_in_game_conversion_rate_loo'] = self.priors['in_game_conversion_rate']
            feats['taker_must_score_conversion_rate_loo'] = 0.50
            feats['taker_right_foot_pct_loo'] = self.priors['right_foot_rate']
            feats['taker_left_foot_pct_loo'] = self.priors['left_foot_rate']
            zone_dist = []
            for z in ZONES:
                p = self.priors['zone_distribution'][z]
                feats[f'taker_zone_{z}_prob_loo'] = p
                zone_dist.append(p)
            feats['taker_zone_entropy_loo'] = float(scipy_entropy(np.array(zone_dist) + 1e-12))
            feats['taker_career_age_years'] = 0

        if keeper_row is not None and keeper_row['total_penalties_faced'] > 0:
            feats['keeper_total_penalties_faced_loo'] = int(keeper_row['total_penalties_faced'])
            feats['keeper_save_rate_loo'] = float(keeper_row['save_rate'])
            feats['keeper_shootout_save_rate_loo'] = float(keeper_row['shootout_save_rate'])
            feats['keeper_in_game_save_rate_loo'] = float(keeper_row['in_game_save_rate'])
            for z in ZONES:
                feats[f'keeper_save_rate_{z}_loo'] = float(keeper_row[f'save_rate_{z}'])
        else:
            feats['keeper_total_penalties_faced_loo'] = 0
            feats['keeper_save_rate_loo'] = self.priors['save_rate']
            feats['keeper_shootout_save_rate_loo'] = 1 - self.priors['shootout_conversion_rate']
            feats['keeper_in_game_save_rate_loo'] = 1 - self.priors['in_game_conversion_rate']
            for z in ZONES:
                feats[f'keeper_save_rate_{z}_loo'] = self.priors['save_rate_by_zone'][z]

        feats['is_shootout_int'] = int(bool(context['is_shootout']))
        feats['shootout_pressure_code'] = PRESSURE_MAP.get(context['shootout_pressure'], -1)
        feats['game_state_code'] = GAME_STATE_MAP.get(context['game_state'], -1)
        feats['score_diff_at_penalty'] = context['score_diff_at_penalty']
        feats['minute'] = context['minute']
        feats['is_left_foot_int'] = int(bool(context['is_left_foot']))
        feats['is_right_foot_int'] = int(bool(context['is_right_foot']))
        feats['is_high_pressure'] = int(
            context['shootout_pressure'] == 'MUST_SCORE' or
            (context['game_state'] == 'TRAILING' and context['minute'] >= 80)
        )
        return feats

    def _build_outcome_features_for_zone(self, base_features: dict, zone: str) -> dict:
        feats = dict(base_features)
        feats['zone_is_corner'] = int(zone in ['TL', 'TR', 'BL', 'BR'])
        feats['zone_is_top'] = int(zone in ['TL', 'TC', 'TR'])
        feats['zone_is_left'] = int(zone in ['TL', 'BL'])
        feats['zone_is_right'] = int(zone in ['TR', 'BR'])
        feats['zone_is_center'] = int(zone in ['TC', 'BC'])
        feats['keeper_save_rate_at_zone_loo'] = base_features[f'keeper_save_rate_{zone}_loo']
        feats['taker_zone_prob_at_zone_loo'] = base_features[f'taker_zone_{zone}_prob_loo']
        feats['matchup_gap_loo'] = (
            feats['taker_zone_prob_at_zone_loo'] - feats['keeper_save_rate_at_zone_loo']
        )
        return feats

    def _predict_zone_distribution(self, base_features: dict, taker_row=None) -> dict:
        row = pd.DataFrame([base_features])[self.shot_feature_cols]
        proba_raw = self.shot_model.predict_proba(row)
        proba_cal = self.shot_calibrator.predict_proba(proba_raw)[0]

        model_probs = {
            self.shot_encoder.classes_[i]: float(proba_cal[i])
            for i in range(len(self.shot_encoder.classes_))
        }

        BLEND_THRESHOLD = 5
        BLEND_WEIGHT = 0.5
        if taker_row is not None and taker_row.get('total_penalties', 0) >= BLEND_THRESHOLD:
            profile_probs = {z: float(taker_row[f'zone_{z}_prob']) for z in ZONES}
            blended = {
                z: (1 - BLEND_WEIGHT) * model_probs[z] + BLEND_WEIGHT * profile_probs[z]
                for z in ZONES
            }
            total = sum(blended.values())
            return {z: p / total for z, p in blended.items()}

        return model_probs

    def _predict_outcomes_for_all_zones(self, base_features, taker_row, keeper_row) -> dict:
        rows = []
        for zone in ZONES:
            feats = self._build_outcome_features_for_zone(base_features, zone)
            rows.append(feats)
        df = pd.DataFrame(rows)[self.outcome_feature_cols]
        proba_raw = self.outcome_model.predict_proba(df)
        proba_cal = self.outcome_calibrator.predict_proba(proba_raw)
        return {
            ZONES[i]: {
                self.outcome_encoder.classes_[c]: float(proba_cal[i, c])
                for c in range(len(self.outcome_encoder.classes_))
            }
            for i in range(len(ZONES))
        }

    def _marginalize(self, zone_probs: dict, outcome_by_zone: dict) -> dict:
        marginal = {oc: 0.0 for oc in OUTCOME_CLASSES}
        for zone, zone_p in zone_probs.items():
            for oc, oc_p in outcome_by_zone[zone].items():
                marginal[oc] += zone_p * oc_p
        total = sum(marginal.values())
        if total > 0:
            marginal = {k: v / total for k, v in marginal.items()}
        return marginal

    def _monte_carlo(self, zone_probs, outcome_by_zone, n=10000, random_state=None) -> dict:
        rng = np.random.default_rng(random_state)
        zone_list = list(zone_probs.keys())
        zone_p = np.array([zone_probs[z] for z in zone_list])
        zone_p = zone_p / zone_p.sum()
        sampled_zones = rng.choice(zone_list, size=n, p=zone_p)

        outcome_list = OUTCOME_CLASSES
        outcomes_sampled = np.empty(n, dtype=object)
        for zone in zone_list:
            mask = sampled_zones == zone
            count = mask.sum()
            if count == 0:
                continue
            oc_p = np.array([outcome_by_zone[zone][oc] for oc in outcome_list])
            oc_p = oc_p / oc_p.sum()
            outcomes_sampled[mask] = rng.choice(outcome_list, size=count, p=oc_p)

        counts = {oc: int((outcomes_sampled == oc).sum()) for oc in outcome_list}
        zone_counts = {z: int((sampled_zones == z).sum()) for z in zone_list}
        return {
            'n': n,
            'outcome_counts': counts,
            'outcome_pct': {oc: counts[oc] / n for oc in outcome_list},
            'zone_counts': zone_counts,
            'zone_pct': {z: zone_counts[z] / n for z in zone_list},
        }

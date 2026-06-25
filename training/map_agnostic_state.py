"""Augment the base state with map-agnostic features.

Appends 7 computed features to the base state so the policy can generalize to
unknown towns: in_lane_center (1) + road-type one-hot (4) + visibility (1) +
oncoming-traffic flag (1).

Dimension note: the base state is 48-dim (Tier-1), so the augmented state is
48 + 7 = 55. (The Tier-2 brief said "42 -> 49"; that assumes the pre-Tier-1
state. The wrapper derives the size from ``config.state_dim`` so it stays
correct regardless.)
"""
import numpy as np

from utils.map_agnostic_features import compute_map_agnostic_features

_ROAD_TYPE_ONEHOT = {
    "straight":     [1.0, 0.0, 0.0, 0.0],
    "curve":        [0.0, 1.0, 0.0, 0.0],
    "intersection": [0.0, 0.0, 1.0, 0.0],
    "unknown":      [0.0, 0.0, 0.0, 1.0],
}
N_EXTRA_FEATURES = 7


class MapAgnosticStateWrapper:
    def __init__(self, config):
        self.config = config
        self.base_state_dim = config.state_dim
        self.augmented_state_dim = self.base_state_dim + N_EXTRA_FEATURES

    def augment_state(self, state, info):
        """Return the base state with map-agnostic features appended (np.float32)."""
        base = np.asarray(state, dtype=np.float32).reshape(-1)[:self.base_state_dim]
        feats = compute_map_agnostic_features(base, info, self.config)

        extra = [feats["in_lane_center"]]
        # 'sharp_curve' (and anything unmapped) falls back to the 'unknown' slot.
        extra.extend(_ROAD_TYPE_ONEHOT.get(feats["road_type"],
                                           _ROAD_TYPE_ONEHOT["unknown"]))
        extra.append(feats["visibility"])
        extra.append(float(feats["oncoming_traffic"]))

        return np.concatenate([base, np.asarray(extra, dtype=np.float32)])

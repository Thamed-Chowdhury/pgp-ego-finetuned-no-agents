"""
NuScenesEgoGraphsDoScenes: ego-graph dataset wired to the doScenes evaluation
protocol (one window per scene, anchor at sample index HISTORY_LEN, no
overlap, no reliance on the nuScenes prediction-challenge split files).

Use this for the doScenes test split where:
  - There is no prediction_challenge_split (the test split is not in the
    standard prediction challenge files).
  - We want exactly one (history, anchor, future) window per scene, with the
    anchor at scene.samples[HISTORY_LEN] (the 5th keyframe), matching the
    official doScenes dataloader.
"""

from datasets.nuScenes.nuScenes_ego_graphs import NuScenesEgoGraphs
from typing import Dict, List
from nuscenes.prediction import PredictHelper


HISTORY_LEN = 4    # 2 s @ 2 Hz
FUTURE_LEN = 12    # 6 s @ 2 Hz
MIN_SCENE_SAMPLES = HISTORY_LEN + 1 + FUTURE_LEN  # 17


class NuScenesEgoGraphsDoScenes(NuScenesEgoGraphs):
    """
    Identical to NuScenesEgoGraphs except token_list is built from every scene
    in helper.data.scene (sorted by scene name), one ego window per scene with
    anchor at samples[HISTORY_LEN]. Scenes shorter than MIN_SCENE_SAMPLES are
    skipped.
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        super().__init__(mode, data_dir, args, helper)
        # Override token list with doScenes-style selection
        self.token_list = self._build_doscenes_token_list()

    def _build_doscenes_token_list(self) -> List[str]:
        ego_tokens = []
        for s in sorted(self.helper.data.scene, key=lambda x: x['name']):
            samples = []
            t = s['first_sample_token']
            while t:
                samples.append(t)
                t = self.helper.data.get('sample', t)['next']
            if len(samples) >= MIN_SCENE_SAMPLES:
                anchor = samples[HISTORY_LEN]
                ego_tokens.append(f'ego_{anchor}')
        return ego_tokens

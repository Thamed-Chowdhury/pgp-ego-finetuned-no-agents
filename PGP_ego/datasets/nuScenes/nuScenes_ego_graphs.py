"""
NuScenesEgoGraphs: PGP graph dataset treating the ego vehicle as the prediction target.

The ego vehicle has no instance token or sample_annotation entry. Its trajectory is
read directly from the ego_pose table. Token format: "ego_{sample_token}".

Key overrides vs NuScenesGraphs:
  - __init__           : builds ego token list from prediction-challenge scenes
  - get_target_agent_global_pose : reads ego_pose via LIDAR_TOP → ego_pose_token
  - get_target_agent_representation: ego history from past sample chain
  - get_target_agent_future      : ego future from future sample chain
  - get_visited_edges            : uses ego future poses instead of helper API
  - save_data / load_data        : filenames use "ego_{s_t}.pickle"

All other map/surrounding-agent logic is inherited unchanged.
"""

from datasets.nuScenes.nuScenes_graphs import NuScenesGraphs
from nuscenes.eval.prediction.splits import get_prediction_challenge_split
from nuscenes.eval.common.utils import quaternion_yaw
from nuscenes.prediction.input_representation.static_layers import correct_yaw
from nuscenes.prediction import PredictHelper
from pyquaternion import Quaternion
import numpy as np
import os
import pickle
from typing import Dict, Tuple, List


class NuScenesEgoGraphs(NuScenesGraphs):
    """
    PGP graph dataset with the ego vehicle as prediction target.
    Inherits all map/surrounding-agent logic from NuScenesGraphs.
    """

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        super().__init__(mode, data_dir, args, helper)
        # Replace prediction-challenge token list with ego tokens derived from the
        # same set of scenes.
        self.token_list = self._build_ego_token_list(args['split'])

    # ------------------------------------------------------------------
    # Token list
    # ------------------------------------------------------------------

    def _build_ego_token_list(self, split: str) -> List[str]:
        """
        Returns non-overlapping ['ego_{s_t}', ...] for the split's scenes.

        A valid window requires:
          - at least t_h*2 keyframes of history before the sample (no zero-padding)
          - at least t_f*2 keyframes of future after the sample
        Windows are spaced t_f*2 keyframes apart so they do not overlap.
        """
        pred_tokens = get_prediction_challenge_split(split, dataroot=self.helper.data.dataroot)

        # Collect unique scenes from the prediction challenge split
        sample_tokens_in_split = set(t.split('_')[1] for t in pred_tokens)
        scenes_in_split = set(
            self.helper.data.get('sample', st)['scene_token']
            for st in sample_tokens_in_split
        )

        n_hist_needed   = int(self.t_h * 2)   # 4 keyframes of history  (2s @ 2Hz)
        n_future_needed = int(self.t_f * 2)   # 12 keyframes of future  (6s @ 2Hz)
        step            = n_future_needed      # non-overlapping: advance by t_f

        ego_tokens = []

        for scene_token in sorted(scenes_in_split):
            scene = self.helper.data.get('scene', scene_token)
            # Collect all keyframe sample tokens in chronological order
            samples = []
            s = scene['first_sample_token']
            while s:
                samples.append(s)
                s = self.helper.data.get('sample', s)['next']

            # Candidate start indices: first valid index is n_hist_needed,
            # last valid index is len(samples) - 1 - n_future_needed.
            # Step by n_future_needed to ensure non-overlapping windows.
            first = n_hist_needed
            last  = len(samples) - 1 - n_future_needed
            for i in range(first, last + 1, step):
                ego_tokens.append(f'ego_{samples[i]}')

        return ego_tokens

    # ------------------------------------------------------------------
    # Ego pose helpers
    # ------------------------------------------------------------------

    def _get_ego_pose_at_sample(self, sample_token: str) -> Dict:
        """Returns the ego_pose record for a given sample token (via LIDAR_TOP)."""
        sample = self.helper.data.get('sample', sample_token)
        sd = self.helper.data.get('sample_data', sample['data']['LIDAR_TOP'])
        return self.helper.data.get('ego_pose', sd['ego_pose_token'])

    def _ego_pose_to_global(self, ep: Dict) -> Tuple[float, float, float]:
        """Extracts (x, y, yaw) in global coordinates from an ego_pose record."""
        x, y = ep['translation'][:2]
        yaw = quaternion_yaw(Quaternion(ep['rotation']))
        yaw = correct_yaw(yaw)
        return x, y, yaw

    # ------------------------------------------------------------------
    # Overrides
    # ------------------------------------------------------------------

    def get_target_agent_global_pose(self, idx: int) -> Tuple[float, float, float]:
        """Ego global pose read from ego_pose table."""
        s_t = self.token_list[idx].split('_', 1)[1]
        ep = self._get_ego_pose_at_sample(s_t)
        return self._ego_pose_to_global(ep)

    def get_target_agent_representation(self, idx: int) -> np.ndarray:
        """
        Builds ego history tensor of shape [t_h*2 + 1, 5]:
          cols 0-1: (x, y) in ego-centric frame at current time
          cols 2-4: (velocity, acceleration, yaw_rate)
        """
        s_t = self.token_list[idx].split('_', 1)[1]
        origin = self.get_target_agent_global_pose(idx)

        # Walk backwards along sample chain to collect past poses
        n_hist = int(self.t_h * 2)
        past_global = []   # most-recent-first
        cur = s_t
        for _ in range(n_hist):
            sample = self.helper.data.get('sample', cur)
            if not sample['prev']:
                break
            cur = sample['prev']
            ep = self._get_ego_pose_at_sample(cur)
            past_global.append(self._ego_pose_to_global(ep))

        # Reverse so oldest-first, then append current (0, 0) at end
        past_global = past_global[::-1]   # oldest → newest past

        # Build [n_hist+1, 2] position array (last row = current = 0,0)
        xy = np.zeros((n_hist + 1, 2))
        for k, (gx, gy, _) in enumerate(past_global):
            lx, ly, _ = self.global_to_local(origin, (gx, gy, 0))
            offset = n_hist - len(past_global)   # zero-pad at front
            xy[offset + k] = [lx, ly]
        # xy[-1] stays (0, 0) — current position in its own frame

        # Motion states via finite differences (dt = 0.5 s at 2 Hz)
        dt = 0.5
        motion = np.zeros((n_hist + 1, 3))  # v, a, yaw_rate
        all_global = list(past_global) + [origin]

        # Velocity
        speeds = []
        for k in range(1, len(all_global)):
            dx = all_global[k][0] - all_global[k - 1][0]
            dy = all_global[k][1] - all_global[k - 1][1]
            speeds.append(np.sqrt(dx ** 2 + dy ** 2) / dt)

        # Yaw rate
        yaw_rates = []
        for k in range(1, len(all_global)):
            d_yaw = all_global[k][2] - all_global[k - 1][2]
            d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
            yaw_rates.append(d_yaw / dt)

        # Acceleration
        accels = []
        for k in range(1, len(speeds)):
            accels.append((speeds[k] - speeds[k - 1]) / dt)

        # Fill motion states array (aligned to last n_hist+1 positions)
        start = n_hist - len(all_global) + 1
        for k, v in enumerate(speeds):
            if start + k + 1 <= n_hist:
                motion[start + k + 1, 0] = v
        if speeds:
            motion[-1, 0] = speeds[-1]
        for k, a in enumerate(accels):
            if start + k + 2 <= n_hist:
                motion[start + k + 2, 1] = a
        if accels:
            motion[-1, 1] = accels[-1]
        for k, yr in enumerate(yaw_rates):
            if start + k + 1 <= n_hist:
                motion[start + k + 1, 2] = yr
        if yaw_rates:
            motion[-1, 2] = yaw_rates[-1]

        motion = np.nan_to_num(motion)
        hist = np.concatenate((xy, motion), axis=1)
        return hist

    def get_target_agent_future(self, idx: int) -> np.ndarray:
        """
        Ego future trajectory in ego-centric frame, shape [t_f*2, 2].
        """
        s_t = self.token_list[idx].split('_', 1)[1]
        origin = self.get_target_agent_global_pose(idx)

        n_future = int(self.t_f * 2)
        fut = np.zeros((n_future, 2))

        cur = s_t
        for k in range(n_future):
            sample = self.helper.data.get('sample', cur)
            if not sample['next']:
                break
            cur = sample['next']
            ep = self._get_ego_pose_at_sample(cur)
            gx, gy, _ = self._ego_pose_to_global(ep)
            lx, ly, _ = self.global_to_local(origin, (gx, gy, 0))
            fut[k] = [lx, ly]

        return fut

    def get_visited_edges(self, idx: int, lane_graph: Dict):
        """
        Computes ground-truth visited lane nodes/edges using the ego's future poses
        (replaces the helper.get_future_for_agent call in the parent).
        """
        node_feats = lane_graph['lane_node_feats']
        s_next = lane_graph['s_next']
        edge_type = lane_graph['edge_type']

        node_feat_lens = np.sum(1 - lane_graph['lane_node_masks'][:, :, 0], axis=1)
        node_poses = []
        for i, nf in enumerate(node_feats):
            if node_feat_lens[i] != 0:
                node_poses.append(nf[:int(node_feat_lens[i]), :3])

        # Build fine-grained future trajectory from ego_pose chain
        s_t = self.token_list[idx].split('_', 1)[1]
        origin = self.get_target_agent_global_pose(idx)
        n_future = int(self.t_f * 2)

        fut_xy_coarse = np.zeros((n_future, 2))
        cur = s_t
        for k in range(n_future):
            sample = self.helper.data.get('sample', cur)
            if not sample['next']:
                break
            cur = sample['next']
            ep = self._get_ego_pose_at_sample(cur)
            gx, gy, _ = self._ego_pose_to_global(ep)
            lx, ly, _ = self.global_to_local(origin, (gx, gy, 0))
            fut_xy_coarse[k] = [lx, ly]

        # Interpolate to higher resolution (same as parent)
        fut_interpolated = np.zeros((fut_xy_coarse.shape[0] * 10 + 1, 2))
        param_query = np.linspace(0, fut_xy_coarse.shape[0], fut_xy_coarse.shape[0] * 10 + 1)
        param_given = np.linspace(0, fut_xy_coarse.shape[0], fut_xy_coarse.shape[0] + 1)
        val_x = np.concatenate(([0], fut_xy_coarse[:, 0]))
        val_y = np.concatenate(([0], fut_xy_coarse[:, 1]))
        fut_interpolated[:, 0] = np.interp(param_query, param_given, val_x)
        fut_interpolated[:, 1] = np.interp(param_query, param_given, val_y)
        fut_xy = fut_interpolated

        # Compute yaw from trajectory
        fut_yaw = np.zeros(len(fut_xy))
        for n in range(1, len(fut_yaw)):
            fut_yaw[n] = -np.arctan2(fut_xy[n, 0] - fut_xy[n-1, 0], fut_xy[n, 1] - fut_xy[n-1, 1])

        # --- identical graph-walking logic from parent ---
        current_step = 0
        node_seq = np.zeros(self.traversal_horizon)
        evf = np.zeros_like(s_next)

        query_pose = np.asarray([fut_xy[0, 0], fut_xy[0, 1], fut_yaw[0]])
        current_node = self.assign_pose_to_node(node_poses, query_pose)
        node_seq[current_step] = current_node

        for n in range(1, len(fut_xy)):
            query_pose = np.asarray([fut_xy[n, 0], fut_xy[n, 1], fut_yaw[n]])
            dist_from_current = np.min(np.linalg.norm(node_poses[current_node][:, :2] - query_pose[:2], axis=1))

            padding = self.polyline_length * self.polyline_resolution / 2
            if self.map_extent[0] - padding <= query_pose[0] <= self.map_extent[1] + padding and \
                    self.map_extent[2] - padding <= query_pose[1] <= self.map_extent[3] + padding:

                if dist_from_current >= 1.5:
                    assigned_node = self.assign_pose_to_node(node_poses, query_pose)
                    if assigned_node != current_node:
                        if assigned_node in s_next[current_node]:
                            nbr_idx = np.where(s_next[current_node] == assigned_node)[0]
                            nbr_valid = np.where(edge_type[current_node] > 0)[0]
                            nbr_idx = np.intersect1d(nbr_idx, nbr_valid)
                            if edge_type[current_node, nbr_idx].any():
                                evf[current_node, nbr_idx] = 1
                        current_node = assigned_node
                        if current_step < self.traversal_horizon - 1:
                            current_step += 1
                            node_seq[current_step] = current_node
            else:
                break

        goal_node = current_node + self.max_nodes
        node_seq[current_step + 1:] = goal_node
        evf[current_node, -1] = 1

        return node_seq, evf

    def save_data(self, idx: int, data: Dict):
        """Saves pickle with 'ego_{s_t}' filename."""
        filename = os.path.join(self.data_dir, self.token_list[idx] + '.pickle')
        with open(filename, 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_data(self, idx: int) -> Dict:
        """Loads pickle with 'ego_{s_t}' filename."""
        filename = os.path.join(self.data_dir, self.token_list[idx] + '.pickle')
        if not os.path.isfile(filename):
            raise Exception(f'Could not find {filename}. Run dataset in extract_data mode first.')
        with open(filename, 'rb') as handle:
            data = pickle.load(handle)

        if self.random_flips:
            import torch
            if torch.randint(2, (1, 1)).squeeze().bool().item():
                data = self.flip_horizontal(data)

        return data

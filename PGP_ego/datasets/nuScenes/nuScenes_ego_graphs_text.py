"""
Text-conditioned wrapper over NuScenesEgoGraphs.

Same as the parent class for everything (token list, encoder/aggregator inputs,
GT). Additionally, on every load_data() call, fetches the cached Gemini
embedding for the scene's doScenes instruction and attaches it as
inputs['text_embedding'] (shape: (text_dim,) float32). Scenes without a
doScenes annotation get the zero vector cached in the embedding pickle, which
is a no-op under LVMRankedText's zero-init residual addition.

Constructor takes one extra arg in args:
    text_emb_pkl: path to embedding cache produced by embed_doscenes_gemini.py
"""
import os
import pickle
from typing import Dict, List

import numpy as np

from datasets.nuScenes.nuScenes_ego_graphs import NuScenesEgoGraphs
from nuscenes.prediction import PredictHelper


class NuScenesEgoGraphsText(NuScenesEgoGraphs):

    def __init__(self, mode: str, data_dir: str, args: Dict, helper: PredictHelper):
        super().__init__(mode, data_dir, args, helper)
        emb_pkl = args.get('text_emb_pkl')
        if not emb_pkl or not os.path.isfile(emb_pkl):
            raise FileNotFoundError(
                f"text_emb_pkl missing or not a file: {emb_pkl!r}. "
                f"Run embed_doscenes_gemini.py first.")
        with open(emb_pkl, 'rb') as f:
            cache = pickle.load(f)
        self._instr_emb        = cache['instr_embeddings']
        self._scene_to_instr   = cache['scene_to_instruction']
        self._empty_emb        = cache['empty_embedding']
        self._text_dim         = int(cache['dim'])

        # Build anchor_token -> text_emb once at startup.
        self._anchor_to_emb: Dict[str, np.ndarray] = {}
        n_hit, n_miss = 0, 0
        for tok in self.token_list:
            sample_token = tok.split('_', 1)[1]
            try:
                sample = self.helper.data.get('sample', sample_token)
                scene = self.helper.data.get('scene', sample['scene_token'])
                scene_num = int(scene['name'].split('-')[1])
                instr = self._scene_to_instr.get(scene_num, '')
            except Exception:
                instr = ''
            if instr and instr in self._instr_emb:
                self._anchor_to_emb[sample_token] = self._instr_emb[instr]
                n_hit += 1
            else:
                self._anchor_to_emb[sample_token] = self._empty_emb
                n_miss += 1
        print(f"[NuScenesEgoGraphsText] tokens={len(self.token_list)} "
              f"text-hit={n_hit} text-miss={n_miss} text_dim={self._text_dim}")

    def load_data(self, idx: int) -> Dict:
        data = super().load_data(idx)
        sample_token = self.token_list[idx].split('_', 1)[1]
        emb = self._anchor_to_emb.get(sample_token, self._empty_emb)
        # Store as plain ndarray; train_eval.utils.convert2tensors will tensorise it.
        data['inputs']['text_embedding'] = emb.astype(np.float32, copy=False)
        return data

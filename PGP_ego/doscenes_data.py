"""
doScenes natural-language-instruction loader.

Each nuScenes scene has 2-7 paraphrased instructions in
`doScenes-VLM-Planning-main/data/doScenes/entire_doscenes.csv`. The model
should learn to pick the GT-closest trajectory given any one of those
instructions, so we expose:

  - load_doscenes(csv_path) -> {scene_number(int): [instruction(str), ...]}
  - build_sample_to_instructions(nusc, sample_tokens, doscenes_map)
        -> [(sample_token, [instruction, ...]), ...]
"""

import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def load_doscenes(csv_path: str) -> Dict[int, List[str]]:
    """Read entire_doscenes.csv into {scene_number: [instructions...]}.

    Scene numbers are stored as floats in the CSV ("1.0", "36"). Empty rows
    are skipped. Order within a scene preserves CSV order.
    """
    out: Dict[int, List[str]] = defaultdict(list)
    with open(csv_path, encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sn = (row.get('Scene Number') or '').strip()
            instr = (row.get('Instruction') or '').strip()
            if not sn or not instr:
                continue
            try:
                scene_num = int(float(sn))
            except ValueError:
                continue
            out[scene_num].append(instr)
    return dict(out)


def scene_token_to_number(nusc, scene_token: str) -> int:
    """nuScenes scene names look like 'scene-0036' -> 36."""
    name = nusc.get('scene', scene_token)['name']
    return int(name.split('-')[-1])


def build_sample_to_instructions(
    nusc,
    sample_tokens: List[str],
    doscenes_map: Dict[int, List[str]],
) -> List[Tuple[str, List[str]]]:
    """For a list of sample_tokens, return [(sample_token, instructions)].

    Samples whose scene has no doScenes entry are dropped.
    """
    out = []
    for st in sample_tokens:
        sample = nusc.get('sample', st)
        scene_num = scene_token_to_number(nusc, sample['scene_token'])
        instrs = doscenes_map.get(scene_num)
        if not instrs:
            continue
        out.append((st, instrs))
    return out

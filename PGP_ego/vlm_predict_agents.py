"""
Predict surrounding-agent positions on the nuScenes v1.0-test split using
Gemini 2.5 Flash on the CAM_FRONT 2-second history (5 keyframes per scene).

For each scene the VLM sees 5 frames (t = -2.0, -1.5, -1.0, -0.5, 0.0 s)
and is asked to list visible road agents per frame with:
  * a stable id (re-used across frames for the same physical agent)
  * class ('vehicle' or 'pedestrian')
  * forward_m  (positive = ahead of ego)
  * right_m    (positive = to ego's right)

Output: vlm_agents/predictions.jsonl — one record per scene with the parsed
agent list.

Usage:
  python vlm_predict_agents.py \
    --subset_json /path/to/subset20_scenes.json \
    --images_root /path/to/test_samples \
    --out_jsonl  /path/to/vlm_agents/predictions.jsonl
"""

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

import google.genai as genai
from google.genai import types as gtypes

KEYS_FILE = '/teamspace/studios/this_studio/PGP_ego/Gemini_keys.txt'
MODEL_NAME = 'gemini-2.5-flash'
KEY_GAP_SEC = 14.0       # min seconds between calls on the same key
PROMPT_TIMES = ['-2.0s', '-1.5s', '-1.0s', '-0.5s', '0.0s']

SYSTEM_PROMPT = """You are an expert in autonomous-driving perception. You will be shown 5 frames from the FRONT CAMERA of an ego vehicle, captured 0.5 s apart (frames are labelled t=-2.0s, -1.5s, -1.0s, -0.5s, and 0.0s where t=0.0s is "now").

For EACH frame, identify the road agents (other vehicles, pedestrians, cyclists) that are visible AND on or immediately next to the road in front of the ego vehicle. Ignore parked far-side cars, distant background traffic, and any agent farther than 60 m ahead.

For each detected agent, estimate its position in the EGO VEHICLE frame at the moment of that frame:
  - forward_m : distance ahead of the ego, in metres (positive = ahead). Use lane-width = 3.5 m and typical car length = 4.5 m to calibrate scale.
  - right_m   : lateral offset, in metres (positive = to the right of ego; negative = to the left).

Assign each physical agent a stable integer id (1, 2, 3, ...) and re-use the same id across frames when it is the SAME physical agent. Classify as one of:
  - "vehicle"    (car, truck, bus, motorcycle, bicycle)
  - "pedestrian" (any person on foot)

Be conservative — only list agents you are reasonably sure about. Typical scenes will have 0–6 agents in view.

Reply with a STRICT JSON object of the form:
{
  "frames": [
    {"t": "-2.0s", "agents": [{"id": 1, "cls": "vehicle", "fwd": 12.5, "rgt": -1.8}, ...]},
    {"t": "-1.5s", "agents": [...]},
    {"t": "-1.0s", "agents": [...]},
    {"t": "-0.5s", "agents": [...]},
    {"t":  "0.0s", "agents": [...]}
  ]
}
Where cls is "vehicle" or "pedestrian", fwd is forward_m, rgt is right_m.
Output exactly one JSON object — no prose, no markdown fences."""


def load_keys(path: str) -> list[str]:
    with open(path) as f:
        return [k.strip() for k in f if k.strip()]


RESPONSE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "frames": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "t": {"type": "STRING"},
                    "agents": {
                        "type": "ARRAY",
                        "items": {
                            "type": "OBJECT",
                            "properties": {
                                "id": {"type": "INTEGER"},
                                "cls": {"type": "STRING"},
                                "fwd": {"type": "NUMBER"},
                                "rgt": {"type": "NUMBER"},
                            },
                            "required": ["id", "cls", "fwd", "rgt"],
                        },
                    },
                },
                "required": ["t", "agents"],
            },
        }
    },
    "required": ["frames"],
}


def call_gemini(clients, key_last_call, image_bytes_list, labels, retries: int = 4):
    last_err = "no_attempt"
    for attempt in range(retries * max(1, len(clients))):
        now = time.time()
        idx = max(range(len(clients)), key=lambda i: now - key_last_call[i])
        elapsed = now - key_last_call[idx]
        if elapsed < KEY_GAP_SEC:
            time.sleep(KEY_GAP_SEC - elapsed)
        key_last_call[idx] = time.time()

        contents = [SYSTEM_PROMPT]
        for label, img_bytes in zip(labels, image_bytes_list):
            contents.append(f"Frame t={label}:")
            contents.append(gtypes.Part.from_bytes(data=img_bytes, mime_type='image/jpeg'))

        try:
            cfg = gtypes.GenerateContentConfig(
                temperature=0.0,
                max_output_tokens=32000,
                response_mime_type='application/json',
                response_schema=RESPONSE_SCHEMA,
            )
            resp = clients[idx].models.generate_content(
                model=MODEL_NAME, contents=contents, config=cfg,
            )
            text = resp.text or ''
            text = re.sub(r'^```(?:json)?\s*|\s*```$', '', text.strip(), flags=re.MULTILINE)
            parsed = json.loads(text)
            # Normalize: convert short keys to canonical names
            norm = {"frames": []}
            for fr in parsed.get('frames', []):
                agents = []
                for a in fr.get('agents', []):
                    agents.append({
                        'id': int(a.get('id', a.get('agent_id', 0))),
                        'class': str(a.get('cls', a.get('class', 'vehicle'))).lower(),
                        'forward_m': float(a.get('fwd', a.get('forward_m', 0.0))),
                        'right_m': float(a.get('rgt', a.get('right_m', 0.0))),
                    })
                norm["frames"].append({'t': fr.get('t', ''), 'agents': agents})
            return norm, idx, None
        except json.JSONDecodeError as e:
            last_err = f"json_decode: {e} | raw={text[:300]!r}"
        except Exception as e:
            err = str(e)
            last_err = err
            wait = 6.0 if ('quota' in err.lower() or '429' in err) else 2.0
            time.sleep(wait)
    return None, -1, f"all retries failed; last_err={last_err[:200]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--subset_json', required=True)
    ap.add_argument('--images_root', required=True,
                    help='Dir containing samples/CAM_FRONT/*.jpg (e.g. test_samples)')
    ap.add_argument('--out_jsonl', required=True)
    ap.add_argument('--resume', action='store_true')
    args = ap.parse_args()

    with open(args.subset_json) as f:
        scenes = json.load(f)

    done_anchors = set()
    if args.resume and os.path.exists(args.out_jsonl):
        with open(args.out_jsonl) as f:
            for ln in f:
                try:
                    done_anchors.add(json.loads(ln)['anchor_token'])
                except Exception:
                    pass
        print(f"[resume] skipping {len(done_anchors)} already-done scenes")

    keys = load_keys(KEYS_FILE)
    print(f"Loaded {len(keys)} Gemini API keys")
    clients = [genai.Client(api_key=k) for k in keys]
    key_last_call = [0.0] * len(clients)

    os.makedirs(os.path.dirname(args.out_jsonl), exist_ok=True)
    n_ok = 0
    n_err = 0
    with open(args.out_jsonl, 'a') as fout:
        for i, sc in enumerate(scenes):
            if sc['anchor_token'] in done_anchors:
                continue
            img_bytes = []
            for cam_rel in sc['cam_files']:
                p = Path(args.images_root) / cam_rel
                if not p.exists():
                    print(f"  [skip] {sc['scene_name']}: missing image {p}")
                    img_bytes = None
                    break
                img_bytes.append(p.read_bytes())
            if img_bytes is None:
                continue

            t0 = time.time()
            parsed, key_idx, err = call_gemini(clients, key_last_call, img_bytes, PROMPT_TIMES)
            dt = time.time() - t0
            rec = {
                'scene_token': sc['scene_token'],
                'scene_name': sc['scene_name'],
                'anchor_token': sc['anchor_token'],
                'sample_tokens': sc['sample_tokens'],
                'cam_files': sc['cam_files'],
                'vlm_result': parsed,
                'vlm_error': err,
                'used_key_idx': key_idx,
                'wall_time_s': round(dt, 2),
            }
            fout.write(json.dumps(rec) + '\n')
            fout.flush()
            n_frames = len(parsed['frames']) if parsed else 0
            n_agents = sum(len(fr.get('agents', [])) for fr in (parsed['frames'] if parsed else []))
            tag = 'OK ' if parsed else 'ERR'
            print(f"[{i+1:3d}/{len(scenes)}] {tag} {sc['scene_name']:>20}  "
                  f"frames={n_frames}  agents_total={n_agents}  dt={dt:.1f}s  err={err}")
            if parsed:
                n_ok += 1
            else:
                n_err += 1

    print(f"\nDone. ok={n_ok} err={n_err}")


if __name__ == '__main__':
    main()

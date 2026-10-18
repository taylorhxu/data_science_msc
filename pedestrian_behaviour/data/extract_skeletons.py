
"""
Extract skeletons from JAAD video clips into per-track .npy files using ViTPose or
OpenPose (CPM, used as the OpenPose-equivalent in mmpose).

Reads the output of preprocess_jaad.py (bboxes/, frame_ids/, labels.csv)
and writes skeletons_vitpose/ or skeletons_openpose/ alongside them.

Output per track: <tid>.npy  shape (T, 17, 2)  — (x, y) in pixel coords

Models: ViTPose-B (recommended) or CPM, both on COCO, 17 keypoints.
    Weights auto-downloaded on first run via mmpose.

Usage:
    python3 data/extract_skeletons.py \
        --video_dir  /path/to/JAAD/JAAD_clips \
        --tracks_dir data/processed_jaad/train \
        --output_dir data/processed_jaad/train/skeletons_vitpose \
        --method vitpose --device cuda

    # 4th ablation condition — compare extraction quality
    python3 data/extract_skeletons.py --compare_only \
        --vitpose_dir  data/processed_jaad/train/skeletons_vitpose \
        --openpose_dir data/processed_jaad/train/skeletons_openpose
"""

import argparse
import os
import re
from pathlib import Path

import cv2
import numpy as np


def _auto_device():
    import torch
    if torch.cuda.is_available():
        return 'cuda'
    if torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def _mmpose_config(relative_path):
    """Resolve a config path relative to the mmpose .mim directory."""
    from mim.utils import get_installed_path
    mim_dir = os.path.join(get_installed_path('mmpose'), '.mim')
    return os.path.join(mim_dir, 'configs', relative_path)


# -- Model loaders ---------------------------------------------------------------

def load_vitpose(device='cuda'):
    from mmpose.apis import init_model, inference_topdown
    config = _mmpose_config(
        'body_2d_keypoint/topdown_heatmap/coco/'
        'td-hm_ViTPose-base_8xb64-210e_coco-256x192.py'
    )
    ckpt = (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
        'topdown_heatmap/coco/'
        'td-hm_ViTPose-base_8xb64-210e_coco-256x192-216eae50_20230314.pth'
    )
    return init_model(config, ckpt, device=device), inference_topdown


def load_openpose(device='cuda'):
    """CPM (Convolutional Pose Machines) — used as the OpenPose-equivalent in mmpose."""
    from mmpose.apis import init_model, inference_topdown
    config = _mmpose_config(
        'body_2d_keypoint/topdown_heatmap/coco/'
        'td-hm_cpm_8xb64-210e_coco-256x192.py'
    )
    ckpt = (
        'https://download.openmmlab.com/mmpose/v1/body_2d_keypoint/'
        'topdown_heatmap/coco/'
        'td-hm_cpm_8xb64-210e_coco-256x192-0e978875_20220920.pth'
    )
    return init_model(config, ckpt, device=device), inference_topdown


_LOADERS = {'vitpose': load_vitpose, 'openpose': load_openpose}


# -- Per-track extraction ----------------------------------------------------------

def extract_track(frames, bboxes_xywh, model, infer_fn):
    """
    Run top-down pose estimation on a sequence of cropped frames.
    Returns (T, 17, 2) float32 array; zeros where inference fails.
    """
    keypoints = []
    for frame, bbox in zip(frames, bboxes_xywh):
        x, y, w, h = bbox.astype(int)
        bbox_xyxy = np.array([[x, y, x + w, y + h]], dtype=np.float32)
        try:
            res = infer_fn(model, frame, bboxes=bbox_xyxy)
            kp  = res[0].pred_instances.keypoints[0][:, :2]
        except Exception:
            kp = np.zeros((17, 2), dtype=np.float32)
        keypoints.append(kp)
    return np.array(keypoints, dtype=np.float32)


# -- Main processing loop -----------------------------------------------------------

def _video_path(video_dir, vid):
    for ext in ('.mp4', '.avi', '.MP4', '.AVI'):
        p = os.path.join(video_dir, vid + ext)
        if os.path.exists(p):
            return p
    return None


def _parse_vid(tid):
    """Extract JAAD video id from track_id = '{video_XXXX}_{pid}'."""
    m = re.match(r'(video_\d+)_(.*)', tid)
    if m:
        return m.group(1)
    return tid.rsplit('_', 1)[0]


def process_all_tracks(video_dir, tracks_dir, output_dir, device, method='vitpose'):
    import pandas as pd

    os.makedirs(output_dir, exist_ok=True)

    labels_csv = os.path.join(tracks_dir, 'labels.csv')
    if not os.path.exists(labels_csv):
        raise FileNotFoundError(
            f"labels.csv not found in {tracks_dir}\n"
            "Run data/preprocess_jaad.py first."
        )

    df = pd.read_csv(labels_csv)
    track_ids = df['track_id'].tolist()
    print(f"Found {len(track_ids)} tracks in {tracks_dir}")

    print(f"Loading {method} on {device}...")
    model, infer_fn = _LOADERS[method](device)
    print("Model ready.")

    skipped = 0
    for tid in track_ids:
        out_path = os.path.join(output_dir, f'{tid}.npy')
        if os.path.exists(out_path):
            continue

        bbox_path = os.path.join(tracks_dir, 'bboxes',    f'{tid}.npy')
        fid_path  = os.path.join(tracks_dir, 'frame_ids', f'{tid}.npy')
        if not (os.path.exists(bbox_path) and os.path.exists(fid_path)):
            print(f"  [SKIP] Missing bbox/frame_ids for {tid}")
            skipped += 1
            continue

        bboxes    = np.load(bbox_path)    # (T, 4) x y w h
        frame_ids = np.load(fid_path)     # (T,)   int frame indices

        vid = _parse_vid(tid)
        vpath = _video_path(video_dir, vid)
        if vpath is None:
            print(f"  [SKIP] Video not found: {vid}")
            skipped += 1
            continue

        cap = cv2.VideoCapture(vpath)
        frames = []
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ret, frame = cap.read()
            frames.append(frame if ret else np.zeros((1, 1, 3), dtype=np.uint8))
        cap.release()

        kp = extract_track(frames, bboxes, model, infer_fn)
        np.save(out_path, kp)
        print(f"  Saved {tid}.npy  shape={kp.shape}")

    total = len(track_ids)
    print(f"Done. {total - skipped}/{total} tracks extracted -> {output_dir}")


# -- Quality comparison (4th ablation condition) -----------------------------------

def compare_quality(vitpose_dir, openpose_dir):
    """Print missing-keypoint rate for ViTPose vs OpenPose on their common tracks."""
    tids = [f.stem for f in Path(vitpose_dir).glob('*.npy')]
    vp_miss, op_miss = [], []

    for tid in tids:
        vp = os.path.join(vitpose_dir,  f'{tid}.npy')
        op = os.path.join(openpose_dir, f'{tid}.npy')
        if not (os.path.exists(vp) and os.path.exists(op)):
            continue
        vp_miss.append((np.load(vp) == 0).all(axis=-1).mean())
        op_miss.append((np.load(op) == 0).all(axis=-1).mean())

    if not vp_miss:
        print("No matching tracks found.")
        return

    print(f"Tracks compared: {len(vp_miss)}")
    print(f"ViTPose  missing keypoint rate: {np.mean(vp_miss):.4f}")
    print(f"OpenPose missing keypoint rate: {np.mean(op_miss):.4f}")


# -- CLI ----------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--video_dir',    type=str)
    p.add_argument('--tracks_dir',   type=str)
    p.add_argument('--output_dir',   type=str)
    p.add_argument('--method',       type=str, default='vitpose',
                   choices=['vitpose', 'openpose'])
    p.add_argument('--device',       type=str, default=None,
                   help='cpu | cuda | mps (default: auto-detect)')
    p.add_argument('--compare_only', action='store_true')
    p.add_argument('--vitpose_dir',  type=str)
    p.add_argument('--openpose_dir', type=str)
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.compare_only:
        compare_quality(args.vitpose_dir, args.openpose_dir)
    else:
        device = args.device or _auto_device()
        process_all_tracks(
            args.video_dir, args.tracks_dir, args.output_dir,
            device, method=args.method,
        )

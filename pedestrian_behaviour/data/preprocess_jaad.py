"""
Preprocess JAAD annotations into the directory layout expected by jaad_dataset.py:

    output_dir/
      {train,val,test}/
        labels.csv            columns: track_id, label, wait_time, scene_type,
                                       bb_expansion_rate
        bboxes/<tid>.npy      (T, 4)  [x, y, w, h]
        contexts/<tid>.npy    (T, 5)  [occlusion, traffic_light, vehicle_act,
                                       intersection, signalized]
        frame_ids/<tid>.npy   (T,)    original JAAD frame indices (for extract_skeletons.py)

Skeletons are NOT generated here — run extract_skeletons.py separately.

Feature notes:
  scene_type        0=mid-block (ND), 1=crosswalk (D) — from 'designated' attribute
  wait_time         seconds of continuous standing at end of observation window
  bb_expansion_rate mean frame-to-frame relative growth of bbox area; TTC proxy
                    (JAAD has no ego-vehicle speed, so this is the best available
                     substitute without running a depth/speed estimator)

  Head orientation variance and mean joint speed (features 4 & 5 from the PT
  feature list) are NOT computable from JAAD annotations — the appearance XMLs
  store only categorical pose direction flags, not keypoint coordinates.
  They require running extract_skeletons.py (ViTPose) on the raw video clips.

Usage:
    python3 data/preprocess_jaad.py --jaad_dir /path/to/JAAD --output_dir data/processed_jaad
"""

import sys
import argparse
import csv
import os

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from JAAD.jaad_data import JAAD

FPS = 30


def _scene_type(attributes):
    """1 = designated crosswalk (D), 0 = mid-block (ND)."""
    return int(attributes.get('designated', 0))


def _wait_time(action_seq):
    """Seconds of continuous standing at the end of the observation window.

    Counts only the unbroken standing run immediately before the decision
    point — not total standing frames — because a pedestrian who walked,
    paused, and then waited is in a different risk state than one who has
    stood still since clip start.
    """
    count = 0
    for a in reversed(action_seq):
        if a == 0:   # standing
            count += 1
        else:
            break
    return count / FPS


def _bb_expansion_rate(bbox_xywh):
    """Mean frame-to-frame relative growth of bbox area.

    Positive → pedestrian expanding in frame (vehicle approaching).
    Used as a TTC proxy when ego-vehicle speed is unavailable.
    Returns 0.0 for sequences shorter than 2 frames.
    """
    if len(bbox_xywh) < 2:
        return 0.0
    areas = bbox_xywh[:, 2] * bbox_xywh[:, 3]   # w * h
    diffs = np.diff(areas) / (areas[:-1] + 1e-6)
    return float(np.mean(diffs))


def _build_context(frame_ids, occlusion, traffic_annots, vehicle_annots, attributes):
    """
    Per-frame context array (T, 5):
        0  occlusion      0=none, 1=part, 2=full
        1  traffic_light  0=n/a, 1=red, 2=green
        2  vehicle_act    0-4 (stopped … accelerating)
        3  intersection   0/1  (static per track, repeated)
        4  signalized     0/1  (static per track, repeated, clamped to binary)
    """
    inter = int(attributes.get('intersection', 0))
    sig   = min(int(attributes.get('signalized', 0)), 1)

    rows = []
    for fid, occ in zip(frame_ids, occlusion):
        tf = traffic_annots.get(fid, {})
        tl = int(tf.get('traffic_light', 0))
        va = int(vehicle_annots.get(fid, 0))
        rows.append([occ, tl, va, inter, sig])
    return np.array(rows, dtype=np.float32)


def process_split(jaad, split, output_dir, annot_database):
    split_dir = os.path.join(output_dir, split)
    bbox_dir    = os.path.join(split_dir, 'bboxes')
    ctx_dir     = os.path.join(split_dir, 'contexts')
    fid_dir     = os.path.join(split_dir, 'frame_ids')
    os.makedirs(bbox_dir, exist_ok=True)
    os.makedirs(ctx_dir,  exist_ok=True)
    os.makedirs(fid_dir,  exist_ok=True)

    seq = jaad.generate_data_trajectory_sequence(
        split,
        fstride=1,
        sample_type='beh',
        seq_type='intention',
        min_track_size=15,
    )

    video_ids_split = set(jaad._get_video_ids_split(split))

    rows = []
    n_saved = 0

    for i, pid_seq in enumerate(seq['pid']):
        pid      = pid_seq[0][0]
        boxes    = seq['bbox'][i]         # list of [x1,y1,x2,y2]
        intent   = seq['intent'][i]       # list of [0/1]
        occ_seq  = seq['occlusion'][i]

        label = int(intent[-1][0])

        # Locate which video this pedestrian belongs to
        vid = None
        for v in video_ids_split:
            if pid in annot_database.get(v, {}).get('ped_annotations', {}):
                vid = v
                break
        if vid is None:
            continue

        ped_annot   = annot_database[vid]['ped_annotations'][pid]
        frame_ids   = ped_annot['frames']
        traffic     = annot_database[vid]['traffic_annotations']
        vehicle     = annot_database[vid]['vehicle_annotations']
        attributes  = ped_annot.get('attributes', {})
        behavior    = ped_annot.get('behavior', {})

        # Trim frame_ids to match the intention sequence length
        # (_get_intention cuts at decision_point; boxes reflects that trim)
        T = len(boxes)
        frame_ids_trimmed = frame_ids[:T]
        occ_trimmed       = occ_seq[:T]
        action_seq        = behavior.get('action', [0] * T)[:T]

        # bbox: convert [x1,y1,x2,y2] → [x,y,w,h]
        bbox_arr = np.array(boxes, dtype=np.float32)
        bbox_xywh = np.stack([
            bbox_arr[:, 0],
            bbox_arr[:, 1],
            bbox_arr[:, 2] - bbox_arr[:, 0],
            bbox_arr[:, 3] - bbox_arr[:, 1],
        ], axis=1)

        ctx_arr = _build_context(
            frame_ids_trimmed, occ_trimmed, traffic, vehicle, attributes)

        wait_time        = _wait_time(action_seq)
        scene_type       = _scene_type(attributes)
        bb_expansion_rate = _bb_expansion_rate(bbox_xywh)

        tid = f'{vid}_{pid}'
        np.save(os.path.join(bbox_dir, f'{tid}.npy'), bbox_xywh)
        np.save(os.path.join(ctx_dir,  f'{tid}.npy'), ctx_arr)
        np.save(os.path.join(fid_dir,  f'{tid}.npy'),
                np.array(frame_ids_trimmed, dtype=np.int32))

        rows.append({
            'track_id':        tid,
            'label':           label,
            'wait_time':       round(wait_time, 4),
            'scene_type':      scene_type,
            'bb_expansion_rate': round(bb_expansion_rate, 6),
        })
        n_saved += 1

    csv_path = os.path.join(split_dir, 'labels.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(
            f, fieldnames=['track_id', 'label', 'wait_time', 'scene_type',
                           'bb_expansion_rate'])
        writer.writeheader()
        writer.writerows(rows)

    print(f'[{split}] saved {n_saved} tracks → {split_dir}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jaad_dir',   required=True,
                        help='Path to cloned JAAD repo (contains annotations/)')
    parser.add_argument('--output_dir', required=True,
                        help='Where to write processed splits')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    jaad = JAAD(data_path=args.jaad_dir)
    print('Building annotation database (cached after first run)…')
    db = jaad.generate_database()

    for split in ('train', 'val', 'test'):
        process_split(jaad, split, args.output_dir, db)

    print('Done.')


if __name__ == '__main__':
    main()

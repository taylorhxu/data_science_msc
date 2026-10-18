"""
JAAD Dataset Loader

Skeletons are extracted with ViTPose (see data/extract_skeletons.py).

Observation window: 60 frames (~2 seconds at 30 fps).

Directory layout expected after preprocessing:
    data_dir/
      {split}/
        labels.csv
        skeletons_vitpose/   <track_id>.npy  (T, 17, 2)
        bboxes/              <track_id>.npy  (T, 4)
        contexts/            <track_id>.npy  (T, C)

Run data/preprocess_jaad.py and data/extract_skeletons.py to generate these.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

NUM_JOINTS = 17
OBS_FRAMES = 60   # ~2 seconds at 30 fps


# ── JAAD ───────────────────────────────────────────────────────────────────

class JADDataset(Dataset):

    def __init__(self, data_dir, split='train', obs_len=OBS_FRAMES, transform=None,
                 skeleton_src='vitpose'):
        super().__init__()
        self.data_dir     = data_dir
        self.split        = split
        self.obs_len      = obs_len
        self.transform    = transform
        self.skeleton_src = skeleton_src
        self.samples      = self._load_samples()
        print(f"[JAAD/{split}] {len(self.samples)} samples | obs_len={obs_len} | skeleton_src={skeleton_src}")

    def _load_samples(self):
        import pandas as pd
        path = os.path.join(self.data_dir, self.split, 'labels.csv')
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Labels not found: {path}\nRun data/preprocess_jaad.py first.")
        df     = pd.read_csv(path)
        skel_d = f'skeletons_{self.skeleton_src}'
        samples = []
        for _, row in df.iterrows():
            tid = row['track_id']
            p = {
                'skeleton': os.path.join(self.data_dir, self.split, skel_d,   f'{tid}.npy'),
                'bbox':     os.path.join(self.data_dir, self.split, 'bboxes', f'{tid}.npy'),
                'context':  os.path.join(self.data_dir, self.split, 'contexts',f'{tid}.npy'),
            }
            if not all(os.path.exists(v) for v in p.values()):
                continue
            samples.append({
                'track_id':        str(tid),
                'label':           int(row['label']),
                'wait_time':       float(row.get('wait_time',        0.0)),
                'scene_type':      int(row.get('scene_type',          0)),
                'bb_expansion_rate': float(row.get('bb_expansion_rate', 0.0)),
                **p,
            })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s        = self.samples[idx]
        skeleton = _fix_len(np.load(s['skeleton']), self.obs_len)
        bbox     = _fix_len(np.load(s['bbox']),     self.obs_len)
        context  = _fix_len(np.load(s['context']),  self.obs_len)

        sample = {
            'skeleton':    torch.from_numpy(skeleton.astype(np.float32)),
            'bbox':        torch.from_numpy(bbox.astype(np.float32)),
            'context':     torch.from_numpy(context.astype(np.float32)),
            'pt_features': torch.from_numpy(_pt_features(
                               s['wait_time'], s['bb_expansion_rate'],
                               s['scene_type'], skeleton)),  # risk_proxy = bb_expansion_rate
            'label':       torch.tensor(s['label'], dtype=torch.float32),
            'track_id':    s['track_id'],
        }
        return self.transform(sample) if self.transform else sample


# ── Shared helpers ────────────────────────────────────────────────────────────

def _pt_features(wait_time, risk_proxy, scene_type, skeleton):
    """
    5-dim PT input feature vector:
        0  wait_time    seconds standing before decision point
        1  risk_proxy   bb_expansion_rate (TTC proxy, no ego-vehicle speed)
        2  scene_type   0=mid-block, 1=designated crosswalk
        3  head_orient_var  variance of nose-to-neck x-displacement across frames
                            (nose=joint 0; neck≈midpoint of shoulders, joints 5&6)
        4  mean_joint_speed mean frame-to-frame displacement across all 17 joints
    """
    if skeleton.shape[1] >= 6:
        neck_x = (skeleton[:, 5, 0] + skeleton[:, 6, 0]) / 2.0
        head_orient_var = float(np.var(skeleton[:, 0, 0] - neck_x))
    else:
        head_orient_var = 0.0

    mean_joint_speed = (
        float(np.mean(np.linalg.norm(np.diff(skeleton, axis=0), axis=-1)))
        if skeleton.shape[0] >= 2 else 0.0
    )

    return np.array([
        float(wait_time),
        float(risk_proxy),
        float(scene_type),
        head_orient_var,
        mean_joint_speed,
    ], dtype=np.float32)


def _fix_len(arr, length):
    """Trim from end or zero-pad start to exactly `length` frames."""
    T = arr.shape[0]
    if T >= length:
        return arr[-length:]
    pad = (length - T,) + arr.shape[1:]
    return np.concatenate([np.zeros(pad, dtype=arr.dtype), arr], axis=0)


def get_dataloader(data_dir, split, dataset='jaad', batch_size=32,
                   num_workers=4, **kwargs):
    """Factory: returns a DataLoader over the JAAD dataset."""
    ds = JADDataset(data_dir, split=split, **kwargs)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == 'train'),
    )

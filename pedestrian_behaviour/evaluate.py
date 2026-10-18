"""
Evaluation and Ablation — PTBehaviour

Four conditions:
    full           ST-GCN + two-stage PT (general + individual adaptation)
    general_only   PT with Δθ_i = 0, uses θ* only
    no_pt          PT gating disabled entirely
    openpose       run separately: re-train/evaluate with OpenPose skeletons

Metrics: Accuracy, AUC, F1, Precision, Recall + per-individual consistency C.

Usage:
    python experiments/evaluate.py \
        --data_dir data/processed_jaad \
        --checkpoint experiments/run_01/best_model.pt \
        --skeleton_src vitpose --device cuda
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jaad_dataset import get_dataloader
from pt_behaviour import PTBehaviour


# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_inputs(batch, device):
    pt        = batch['pt_features'].to(device)
    wait_time = pt[:, 0]
    ttc       = pt[:, 1]
    return (
        torch.sigmoid(wait_time),
        torch.sigmoid(-ttc + 1.0).clamp(0.01, 0.99),
        torch.sigmoid(ttc * 2.0).clamp(0.01, 0.99),
    )


@torch.no_grad()
def run_condition(model, loader, device, mode='full'):
    """
    Args:
        mode: 'full' | 'general_only' | 'no_pt'
    """
    model.eval()
    probs_all, labels_all, ids_all = [], [], []
    params_all = {k: [] for k in ['alpha', 'beta', 'lam', 'gamma']}
    general_profile = None

    for batch in loader:
        skeleton    = batch['skeleton'].to(device)
        pt_features = batch['pt_features'].to(device)
        labels      = batch['label']
        B           = skeleton.shape[0]
        gain, loss_v, col_p = risk_inputs(batch, device)

        if mode == 'no_pt':
            H       = model.backbone(skeleton)
            utility = torch.zeros(B, device=device)
            H_gated = model.gate(H, utility)
            logit   = model.classifier(H_gated).squeeze(-1)
            params  = {k: torch.zeros(B) for k in params_all}

        elif mode == 'general_only':
            H  = model.backbone(skeleton)
            gp = model.pt_module.get_general_profile()
            params  = {k: gp[k].expand(B) for k in params_all}
            utility = model.pt_module._compute_utility(params, gain, loss_v, col_p)
            H_gated = model.gate(H, utility)
            logit   = model.classifier(H_gated).squeeze(-1)

        else:  # full
            logit, utility, params, gp = model(
                skeleton, pt_features, gain, loss_v, col_p)
            general_profile = {k: v.item() for k, v in gp.items()}

        probs_all.extend(torch.sigmoid(logit).cpu().numpy())
        labels_all.extend(labels.numpy())
        ids_all.extend(batch['track_id'])
        for k in params_all:
            v = params[k]
            params_all[k].extend(
                v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v)

    return (
        np.array(probs_all),
        np.array(labels_all),
        {k: np.array(v) for k, v in params_all.items()},
        ids_all,
        general_profile,
    )


def compute_metrics(probs, labels, thr=0.5):
    preds = (probs > thr).astype(int)
    return {
        'accuracy':  round(float(accuracy_score(labels, preds)),                  4),
        'auc':       round(float(roc_auc_score(labels, probs)),                   4),
        'f1':        round(float(f1_score(labels, preds,       zero_division=0)), 4),
        'precision': round(float(precision_score(labels, preds,zero_division=0)), 4),
        'recall':    round(float(recall_score(labels, preds,   zero_division=0)), 4),
    }


def consistency_metric(probs, track_ids):
    """
    Per-individual prediction variance across multiple clips.
    Lower variance = more consistent individual risk profiles.
    Returns None if no pedestrian appears more than once.
    """
    ped = defaultdict(list)
    for p, tid in zip(probs, track_ids):
        ped['_'.join(str(tid).split('_')[:2])].append(float(p))
    variances = [np.var(v) for v in ped.values() if len(v) > 1]
    if not variances:
        return None
    return {
        'mean_variance':   round(float(np.mean(variances)),   5),
        'median_variance': round(float(np.median(variances)), 5),
        'n_individuals':   len(variances),
    }


def print_param_analysis(params, labels, general):
    print("\n── PT Parameter Analysis ──")
    if general:
        print("General profile θ*:")
        for k, v in general.items():
            print(f"  {k:6s}: {v:.4f}")
    print("\nIndividual θ_i: crossing vs not-crossing")
    print(f"{'Param':<8} {'Crossing (mean±std)':<24} {'Not crossing (mean±std)':<24} {'Diff'}")
    print("─" * 70)
    for k, vals in params.items():
        if len(vals) == 0:
            continue
        c  = vals[labels == 1]
        nc = vals[labels == 0]
        if len(c) == 0 or len(nc) == 0:
            continue
        print(f"{k:<8} {c.mean():.3f}±{c.std():.3f}               "
              f"{nc.mean():.3f}±{nc.std():.3f}               "
              f"{c.mean()-nc.mean():+.4f}")


# ── Main ──────────────────────────────────────────────────────────────────────

def evaluate(args):
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt  = torch.load(args.checkpoint, map_location=device)
    model = PTBehaviour(num_frames=60).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f"Checkpoint epoch {ckpt['epoch']}, val AUC {ckpt['val_auc']:.4f}")
    if 'general_profile' in ckpt:
        print(f"Saved θ*: {ckpt['general_profile']}")

    loader = get_dataloader(
        args.data_dir, 'test',
        dataset=args.dataset,
        skeleton_src=args.skeleton_src,
        batch_size=64,
    )

    conditions = [
        ('full',         'Full model (ST-GCN + two-stage PT)'),
        ('general_only', 'Ablation: general profile only (Δθ=0)'),
        ('no_pt',        'Ablation: no PT gating'),
    ]

    results = {}
    for mode, label in conditions:
        print(f"\n── {label}")
        probs, labels, params, ids, gp = run_condition(model, loader, device, mode)
        m    = compute_metrics(probs, labels)
        cons = consistency_metric(probs, ids)

        results[mode] = {'metrics': m, 'consistency': cons}

        print(f"  Acc:{m['accuracy']:.4f}  AUC:{m['auc']:.4f}  "
              f"F1:{m['f1']:.4f}  Prec:{m['precision']:.4f}  Rec:{m['recall']:.4f}")
        if cons:
            print(f"  Consistency variance: {cons['mean_variance']:.5f} "
                  f"(n={cons['n_individuals']} individuals)")

        if mode == 'full':
            print_param_analysis(params, labels, gp)

    # Summary table
    print("\n── Ablation Summary ──")
    print(f"{'Condition':<40} {'AUC':<8} {'F1':<8} {'Acc':<8} {'Consist.'}")
    print("─" * 72)
    for mode, label in conditions:
        m    = results[mode]['metrics']
        cons = results[mode]['consistency']
        c    = f"{cons['mean_variance']:.5f}" if cons else "N/A"
        print(f"{label:<40} {m['auc']:<8.4f} {m['f1']:<8.4f} {m['accuracy']:<8.4f} {c}")

    out = os.path.join(os.path.dirname(args.checkpoint), 'eval_results.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     type=str, required=True)
    p.add_argument('--checkpoint',   type=str, required=True)
    p.add_argument('--dataset',      type=str, default='jaad', choices=['jaad'])
    p.add_argument('--skeleton_src', type=str, default='vitpose', choices=['vitpose','openpose'])
    p.add_argument('--device',       type=str, default='cuda')
    return p.parse_args()


if __name__ == '__main__':
    evaluate(parse_args())

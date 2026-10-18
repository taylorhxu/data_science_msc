"""
Training Script — PTBehaviour

Loss:
    L = L_BCE + mu * L_adapt

    L_BCE    weighted binary cross-entropy (JAAD class imbalance)
    L_adapt  regularises individual PT params toward general profile θ*
    mu       annealed from 1.0 → 0.05 during training

The general profile θ* and adaptation network are jointly optimised.
mu annealing encourages the general profile to stabilise first before
allowing individual deviations to grow.

Reproducibility and class imbalance:
    --seed                 seeds numpy / torch (CPU + CUDA) at the start of
                            training (default 42).
    --oversample_exponent   builds a WeightedRandomSampler over the training
                            set from inverse class-frequency weights raised
                            to this exponent (default 0.9, the value selected
                            in the 12-seed sweep). 0.0 disables oversampling
                            (plain shuffle); 0.5 is mild balancing; 1.0 is
                            full inverse-frequency balancing.

Usage:
    python train.py \
        --data_dir data/processed_jaad \
        --dataset jaad \
        --output_dir experiments/run_01 \
        --epochs 100 --batch_size 32 --device cuda \
        --seed 42 --oversample_exponent 0.9
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jaad_dataset import get_dataloader
from pt_behaviour import PTBehaviour


# ── Helpers ───────────────────────────────────────────────────────────────────

def risk_inputs(batch, device):
    """
    Derive (gain_value, loss_value, collision_prob) from the observable PT
    feature vector, since JAAD provides no direct annotations for these.

    pt_features dims:
        0  wait_time    → gain_value:     longer wait = higher gain from crossing
        1  risk_proxy   → loss_value and collision_prob:
                          bb_expansion_rate (TTC proxy; JAAD has no ego-vehicle speed)
    """
    pt          = batch['pt_features'].to(device)
    wait_time   = pt[:, 0]
    risk_proxy  = pt[:, 1]
    return (
        torch.sigmoid(wait_time),
        torch.sigmoid(-risk_proxy + 1.0).clamp(0.01, 0.99),
        torch.sigmoid(risk_proxy * 2.0).clamp(0.01, 0.99),
    )


def mu_anneal(epoch, total, start=1.0, end=0.05):
    t = epoch / max(total - 1, 1)
    return start * (1 - t) + end * t


def class_weights(loader, device):
    n_pos = n_neg = 0
    for b in loader:
        n_pos += (b['label'] == 1).sum().item()
        n_neg += (b['label'] == 0).sum().item()
    w = torch.tensor([n_neg / max(n_pos, 1)], device=device)
    print(f"  crossing: {n_pos}  not-crossing: {n_neg}  pos_weight: {w.item():.3f}")
    return w


def log_general(model):
    gp = model.get_general_profile()
    print(
        f"  θ*  α:{gp['alpha'].item():.3f}  "
        f"β:{gp['beta'].item():.3f}  "
        f"λ:{gp['lam'].item():.3f}  "
        f"γ:{gp['gamma'].item():.3f}"
    )


def make_oversampled_loader(train_loader, exponent, batch_size):
    """
    Rebuild the training DataLoader with a WeightedRandomSampler over
    inverse class-frequency weights raised to `exponent`.

    Mirrors the Colab notebook cell (iter8_oversample090_full100ep) exactly:

        train_labels  = np.array([s['label'] for s in jaad_loaders['train'].dataset.samples])
        class_counts  = np.bincount(train_labels)
        sample_weights_90 = (1.0 / class_counts**0.9)[train_labels]
        sampler_seed  = WeightedRandomSampler(weights=sample_weights_90,
                                               num_samples=len(sample_weights_90),
                                               replacement=True)
        train_loader_seed = DataLoader(
            jaad_loaders['train'].dataset, batch_size=BATCH_SIZE, sampler=sampler_seed,
            num_workers=4, pin_memory=True, drop_last=True,
        )

    get_dataloader() (jaad_dataset.py) doesn't expose a sampler parameter, so
    — same as the notebook — we bypass it here and construct the DataLoader
    directly over the already-instantiated dataset object, swapping the
    default `shuffle=True` for the weighted sampler (the two are mutually
    exclusive in DataLoader).
    """
    dataset = train_loader.dataset
    train_labels = np.array([s['label'] for s in dataset.samples])
    class_counts = np.bincount(train_labels)
    sample_weights = (1.0 / class_counts ** exponent)[train_labels]
    print(f"  oversample_exponent={exponent}  sample_weights[:3]={sample_weights[:3]}")

    sampler = WeightedRandomSampler(
        weights=sample_weights, num_samples=len(sample_weights), replacement=True,
    )
    return DataLoader(
        dataset, batch_size=batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, drop_last=True,
    )


# ── Train / eval loops ────────────────────────────────────────────────────────

def train_epoch(model, loader, opt, criterion, device, mu):
    model.train()
    total_loss, logits_all, labels_all = 0.0, [], []

    for batch in loader:
        skeleton    = batch['skeleton'].to(device)
        pt_features = batch['pt_features'].to(device)
        labels      = batch['label'].to(device)
        gain, loss_v, col_p = risk_inputs(batch, device)

        opt.zero_grad()
        logit, _, params, _ = model(skeleton, pt_features, gain, loss_v, col_p)
        loss = criterion(logit, labels) + mu * model.pt_module.adaptation_regularisation_loss(params)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        total_loss += loss.item()
        logits_all.extend(logit.detach().cpu().numpy())
        labels_all.extend(labels.detach().cpu().numpy())

    probs  = torch.sigmoid(torch.tensor(logits_all)).numpy()
    labels = np.array(labels_all)
    preds  = (probs > 0.5).astype(int)
    return {
        'loss': total_loss / len(loader),
        'acc':  float(accuracy_score(labels, preds)),
        'auc':  float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'f1':   float(f1_score(labels, preds, zero_division=0)),
    }


@torch.no_grad()
def eval_epoch(model, loader, criterion, device, mu):
    model.eval()
    total_loss, logits_all, labels_all = 0.0, [], []
    params_all = {k: [] for k in ['alpha', 'beta', 'lam', 'gamma']}

    for batch in loader:
        skeleton    = batch['skeleton'].to(device)
        pt_features = batch['pt_features'].to(device)
        labels      = batch['label'].to(device)
        gain, loss_v, col_p = risk_inputs(batch, device)

        logit, _, params, _ = model(skeleton, pt_features, gain, loss_v, col_p)
        loss = criterion(logit, labels) + mu * model.pt_module.adaptation_regularisation_loss(params)

        total_loss += loss.item()
        logits_all.extend(logit.cpu().numpy())
        labels_all.extend(labels.cpu().numpy())
        for k in params_all:
            params_all[k].extend(params[k].cpu().numpy())

    probs  = torch.sigmoid(torch.tensor(logits_all)).numpy()
    labels = np.array(labels_all)
    preds  = (probs > 0.5).astype(int)
    return {
        'loss':   total_loss / len(loader),
        'acc':    float(accuracy_score(labels, preds)),
        'auc':    float(roc_auc_score(labels, probs)) if len(np.unique(labels)) > 1 else 0.0,
        'f1':     float(f1_score(labels, preds, zero_division=0)),
        'params': {k: float(np.mean(v)) for k, v in params_all.items()},
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def train(args):
    # Seed CPU / CUDA RNGs first so data loading, sampling and model init
    # are all reproducible for a given --seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}  |  Dataset: {args.dataset}  |  Seed: {args.seed}")
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nLoading data...")
    train_loader = get_dataloader(args.data_dir, 'train', dataset=args.dataset,
                                  batch_size=args.batch_size, skeleton_src=args.skeleton_src)
    val_loader   = get_dataloader(args.data_dir, 'val',   dataset=args.dataset,
                                  batch_size=args.batch_size, skeleton_src=args.skeleton_src)

    if args.oversample_exponent > 0:
        print(f"\nBuilding oversampled train loader (exponent={args.oversample_exponent})...")
        train_loader = make_oversampled_loader(train_loader, args.oversample_exponent, args.batch_size)

    pw        = class_weights(train_loader, device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)

    model = PTBehaviour(embed_dim=args.embed_dim,
                                dropout=args.dropout, num_frames=60).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    log_general(model)

    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sch = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-5)

    best_auc, history = 0.0, []

    for epoch in range(1, args.epochs + 1):
        mu      = mu_anneal(epoch - 1, args.epochs)
        train_m = train_epoch(model, train_loader, opt, criterion, device, mu)
        val_m   = eval_epoch(model,  val_loader,   criterion, device, mu)
        sch.step()

        p = val_m['params']
        print(
            f"Ep {epoch:3d}/{args.epochs}  mu={mu:.3f}  "
            f"Train loss:{train_m['loss']:.4f} auc:{train_m['auc']:.3f}  |  "
            f"Val loss:{val_m['loss']:.4f} auc:{val_m['auc']:.3f} "
            f"f1:{val_m['f1']:.3f}  "
            f"α:{p['alpha']:.3f} β:{p['beta']:.3f} λ:{p['lam']:.3f} γ:{p['gamma']:.3f}"
        )
        log_general(model)
        history.append({'epoch': epoch, 'mu': mu, 'train': train_m, 'val': val_m})

        if val_m['auc'] > best_auc:
            best_auc = val_m['auc']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_auc': best_auc,
                'args': vars(args),
                'general_profile': {k: v.item() for k, v in model.get_general_profile().items()},
            }, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✓ Best AUC {best_auc:.4f} saved.")

    with open(os.path.join(args.output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nFinal general profile θ*:")
    log_general(model)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir',     type=str,   default='data/processed_jaad')
    p.add_argument('--output_dir',   type=str,   default='experiments/run_01')
    p.add_argument('--dataset',      type=str,   default='jaad', choices=['jaad'])
    p.add_argument('--skeleton_src', type=str,   default='vitpose', choices=['vitpose', 'openpose'])
    p.add_argument('--epochs',       type=int,   default=100)
    p.add_argument('--batch_size',   type=int,   default=32)
    p.add_argument('--lr',           type=float, default=1e-3)
    p.add_argument('--embed_dim',    type=int,   default=256)
    p.add_argument('--dropout',      type=float, default=0.2)
    p.add_argument('--device',       type=str,   default='cuda')
    p.add_argument('--seed',         type=int,   default=42,
                   help='seeds numpy / torch (CPU + CUDA) for reproducibility')
    p.add_argument('--oversample_exponent', type=float, default=0.9,
                   help='exponent applied to inverse class frequency when building the '
                        'WeightedRandomSampler for the train loader (0 disables oversampling; '
                        '0.5=mild, 1.0=full inverse-frequency balancing)')
    return p.parse_args()


if __name__ == '__main__':
    train(parse_args())

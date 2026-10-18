"""
Prospect Theory Module — Two-Stage Risk Profile

Implements the general-to-individual risk profile mechanism:

  TRAINING:   Learn a general risk profile θ* = (α*, β*, λ*, γ*)
              capturing population-level behaviour across all training
              pedestrians. Stored as globally shared nn.Parameter values.

  INFERENCE:  Given a ~2-second input clip, an adaptation network predicts
              a residual Δθ_i. Individual risk profile:
                  θ_i = θ* + Δθ_i
              Δθ_i is regularised toward zero — individual profiles stay
              anchored to the learned population prior.

CPT parameter ranges (Tversky & Kahneman, 1992):
    α, β  ∈ [0.50, 1.00]   gain/loss curvature     (empirical mean 0.88)
    λ     ∈ [1.00, 4.00]   loss aversion            (empirical mean 2.25)
    γ     ∈ [0.40, 1.00]   probability distortion   (empirical mean 0.65)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

PT_PRIORS = {'alpha': 0.88, 'beta': 0.88, 'lam': 2.25, 'gamma': 0.65}
PT_RANGES = {
    'alpha': (0.50, 1.00),
    'beta':  (0.50, 1.00),
    'lam':   (1.00, 4.00),
    'gamma': (0.40, 1.00),
}


def _inv_sig(y, lo, hi):
    """Inverse sigmoid: map y ∈ (lo, hi) → unconstrained real."""
    p = max(min((y - lo) / (hi - lo), 0.9999), 0.0001)
    return math.log(p / (1 - p))


def _scale(x, lo, hi):
    """Sigmoid scale from (-inf, inf) → (lo, hi)."""
    return lo + (hi - lo) * torch.sigmoid(x)


class ProspectTheoryModule(nn.Module):
    """
    Two-stage CPT module.

    Components:
        general_*        globally shared nn.Parameter for each CPT param
        adaptation_net   MLP predicting per-sample residual Δθ from PT features
    """

    def __init__(self, input_dim=5, hidden_dim=32):
        super().__init__()

        # General profile — initialised near CPT literature priors
        self.general_alpha = nn.Parameter(torch.tensor(_inv_sig(PT_PRIORS['alpha'], *PT_RANGES['alpha'])))
        self.general_beta  = nn.Parameter(torch.tensor(_inv_sig(PT_PRIORS['beta'],  *PT_RANGES['beta'])))
        self.general_lam   = nn.Parameter(torch.tensor(_inv_sig(PT_PRIORS['lam'],   *PT_RANGES['lam'])))
        self.general_gamma = nn.Parameter(torch.tensor(_inv_sig(PT_PRIORS['gamma'], *PT_RANGES['gamma'])))

        # Adaptation network — outputs small residuals Δθ_i
        self.adaptation_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        # Near-zero (not exactly zero) init: individual profiles start close to the
        # general profile, but a Linear layer with an exactly-zero weight matrix blocks
        # ALL gradient from reaching earlier layers (dL/d(input) = dL/d(output) @ W^T = 0
        # when W=0), permanently freezing the rest of the adaptation network. A small
        # non-zero weight keeps Delta-theta_i ~ 0 at init while still letting gradients
        # flow back from step one.
        nn.init.normal_(self.adaptation_net[-1].weight, std=1e-3)
        nn.init.zeros_(self.adaptation_net[-1].bias)

    def get_general_profile(self):
        """Return constrained θ* as a dict of scalars."""
        return {
            'alpha': _scale(self.general_alpha, *PT_RANGES['alpha']),
            'beta':  _scale(self.general_beta,  *PT_RANGES['beta']),
            'lam':   _scale(self.general_lam,   *PT_RANGES['lam']),
            'gamma': _scale(self.general_gamma, *PT_RANGES['gamma']),
        }

    def get_individual_profile(self, pt_features):
        """
        Compute θ_i = θ* + Δθ_i for a batch.

        Args:
            pt_features (B, input_dim)
        Returns:
            dict of (B,) tensors: alpha, beta, lam, gamma
        """
        general   = self.get_general_profile()
        raw_delta = self.adaptation_net(pt_features)  # (B, 4)

        # Scale delta to ±20% of each parameter range
        delta_scale = torch.tensor([
            (PT_RANGES['alpha'][1] - PT_RANGES['alpha'][0]) * 0.2,
            (PT_RANGES['beta'][1]  - PT_RANGES['beta'][0])  * 0.2,
            (PT_RANGES['lam'][1]   - PT_RANGES['lam'][0])   * 0.2,
            (PT_RANGES['gamma'][1] - PT_RANGES['gamma'][0]) * 0.2,
        ], device=pt_features.device)

        delta = torch.tanh(raw_delta) * delta_scale   # (B, 4)

        return {
            'alpha': (general['alpha'] + delta[:, 0]).clamp(*PT_RANGES['alpha']),
            'beta':  (general['beta']  + delta[:, 1]).clamp(*PT_RANGES['beta']),
            'lam':   (general['lam']   + delta[:, 2]).clamp(*PT_RANGES['lam']),
            'gamma': (general['gamma'] + delta[:, 3]).clamp(*PT_RANGES['gamma']),
        }

    def forward(self, pt_features, gain_value, loss_value, collision_prob):
        """
        Full forward pass.

        Args:
            pt_features    (B, 5)  observable pedestrian features
            gain_value     (B,)    normalised time saved by crossing
            loss_value     (B,)    collision severity estimate
            collision_prob (B,)    estimated P(collision | gap)

        Returns:
            utility (B,)   CPT prospect value of crossing
            params  dict   individual (B,) tensors: alpha, beta, lam, gamma
            general dict   population-level scalars
        """
        params  = self.get_individual_profile(pt_features)
        utility = self._compute_utility(params, gain_value, loss_value, collision_prob)
        return utility, params, self.get_general_profile()

    def _compute_utility(self, params, gain_value, loss_value, collision_prob):
        """U = w(p_success)*V(gain) + w(p_collision)*V(-loss)"""
        alpha = params['alpha']
        beta  = params['beta']
        lam   = params['lam']
        gamma = params['gamma']

        gain  = gain_value.clamp(min=1e-6)
        loss  = loss_value.clamp(min=1e-6)
        p_col = collision_prob.clamp(1e-6, 1 - 1e-6)
        p_suc = (1 - p_col).clamp(1e-6, 1 - 1e-6)

        v_gain = gain.pow(alpha)
        v_loss = -lam * loss.pow(beta)
        w_gain = self._w(p_suc, gamma)
        w_col  = self._w(p_col, gamma)

        return w_gain * v_gain + w_col * v_loss

    def _w(self, p, gamma):
        """Tversky-Kahneman probability weighting: p^γ / (p^γ+(1-p)^γ)^(1/γ)"""
        pg    = p.pow(gamma)
        qg    = (1 - p).pow(gamma)
        denom = (pg + qg).pow(1.0 / gamma.clamp(min=1e-3))
        return pg / (denom + 1e-8)

    def adaptation_regularisation_loss(self, params):
        """
        Pull individual params toward the learned general profile.
        Keeps individual deviations small and interpretable.
        """
        general = self.get_general_profile()
        return sum(
            F.mse_loss(params[k], general[k].detach().expand_as(params[k]))
            for k in ['alpha', 'beta', 'lam', 'gamma']
        )

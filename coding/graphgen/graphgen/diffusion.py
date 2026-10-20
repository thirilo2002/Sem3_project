"""
Mixed-type (continuous + categorical) tabular diffusion model, TabDDPM-style.

Each node type's feature row is split into:
  - continuous columns  -> standard Gaussian DDPM (eps-prediction)
  - categorical columns -> multinomial diffusion (D3PM "uniform" transition,
    trained with the simplified x0-prediction / cross-entropy objective that
    TabDDPM uses)

Both share one small MLP denoiser and are trained jointly with one optimizer
step per batch. An optional conditioning vector (e.g. a cluster-id embedding
from fk_sampler.py, or a parent node's embedding) is concatenated into the
denoiser input -- this is how "generate claim attributes conditioned on which
beneficiary/provider/physician it belongs to" is implemented.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .schema import NodeTypeSpec


# ---------------------------------------------------------------------------
# Time embedding + denoiser network
# ---------------------------------------------------------------------------

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) long timesteps
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=t.device).float() / max(half - 1, 1)
        )
        args = t.float()[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb


class Denoiser(nn.Module):
    """Single MLP that denoises the concatenated [continuous | categorical
    one-hot] vector, conditioned on timestep and an optional context vector.
    Outputs eps-prediction for the continuous block and per-column logits for
    each categorical column (predicting x0, not eps, for categoricals).
    """

    def __init__(
        self,
        n_continuous: int,
        cat_num_classes: List[int],
        time_emb_dim: int = 64,
        cond_dim: int = 0,
        hidden: int = 256,
        n_layers: int = 4,
    ):
        super().__init__()
        self.n_continuous = n_continuous
        self.cat_num_classes = cat_num_classes
        self.cat_total_dim = sum(cat_num_classes)
        in_dim = n_continuous + self.cat_total_dim

        self.time_emb = SinusoidalTimeEmbedding(time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim), nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        total_in = in_dim + time_emb_dim + cond_dim
        layers = [nn.Linear(total_in, hidden), nn.SiLU()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.SiLU()]
        self.backbone = nn.Sequential(*layers)

        self.cont_head = nn.Linear(hidden, n_continuous) if n_continuous > 0 else None
        self.cat_heads = nn.ModuleList(
            [nn.Linear(hidden, k) for k in cat_num_classes]
        )

    def forward(
        self,
        x_cont: torch.Tensor,        # (B, n_continuous), possibly empty
        x_cat_onehot: torch.Tensor,  # (B, sum(cat_num_classes)), possibly empty
        t: torch.Tensor,             # (B,)
        cond: Optional[torch.Tensor] = None,  # (B, cond_dim)
    ) -> Tuple[Optional[torch.Tensor], List[torch.Tensor]]:
        temb = self.time_mlp(self.time_emb(t))
        parts = [p for p in [x_cont, x_cat_onehot, temb] if p.shape[-1] > 0]
        if cond is not None and cond.shape[-1] > 0:
            parts.append(cond)
        h = torch.cat(parts, dim=-1)
        h = self.backbone(h)
        eps_cont = self.cont_head(h) if self.cont_head is not None else None
        cat_logits = [head(h) for head in self.cat_heads]
        return eps_cont, cat_logits


# ---------------------------------------------------------------------------
# Diffusion schedules
# ---------------------------------------------------------------------------

def linear_beta_schedule(T: int, beta_start=1e-4, beta_end=2e-2) -> torch.Tensor:
    return torch.linspace(beta_start, beta_end, T)


@dataclass
class DiffusionSchedule:
    T: int
    betas: torch.Tensor
    alphas: torch.Tensor
    alphas_cumprod: torch.Tensor

    @staticmethod
    def make(T: int = 1000) -> "DiffusionSchedule":
        betas = linear_beta_schedule(T)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        return DiffusionSchedule(T, betas, alphas, alphas_cumprod)

    def to(self, device):
        self.betas = self.betas.to(device)
        self.alphas = self.alphas.to(device)
        self.alphas_cumprod = self.alphas_cumprod.to(device)
        return self


# ---------------------------------------------------------------------------
# Mixed-type diffusion model
# ---------------------------------------------------------------------------

class MixedTypeDiffusion(nn.Module):
    """Diffusion model over one node type's feature table.

    x_cont: (N, n_continuous) float tensor, assumed standardized (zero mean,
        unit variance) by the caller -- see `standardize_continuous`.
    x_cat:  (N, n_categorical) long tensor of integer class ids, one column
        per categorical feature, each in [0, num_classes_i).
    """

    def __init__(self, spec: NodeTypeSpec, T: int = 1000, cond_dim: int = 0,
                 hidden: int = 256, n_layers: int = 4):
        super().__init__()
        self.spec = spec
        self.cat_num_classes = [spec.columns.categorical_idx[i] for i in sorted(spec.columns.categorical_idx)]
        self.n_continuous = spec.n_continuous
        self.schedule = DiffusionSchedule.make(T)
        self.cond_dim = cond_dim
        self.denoiser = Denoiser(
            n_continuous=self.n_continuous,
            cat_num_classes=self.cat_num_classes,
            cond_dim=cond_dim,
            hidden=hidden,
            n_layers=n_layers,
        )

    def to(self, device):
        super().to(device)
        self.schedule.to(device)
        return self

    # ---- forward (noising) processes ----

    def _q_sample_cont(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        a_bar = self.schedule.alphas_cumprod[t][:, None]
        eps = torch.randn_like(x0)
        xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
        return xt, eps

    def _q_sample_cat_onehot(self, x0_onehot: torch.Tensor, t: torch.Tensor, k: int) -> torch.Tensor:
        # uniform (D3PM-style) multinomial diffusion for one categorical column
        a_bar = self.schedule.alphas_cumprod[t][:, None]
        probs = a_bar * x0_onehot + (1 - a_bar) / k
        xt = torch.distributions.OneHotCategorical(probs=probs).sample()
        return xt

    # ---- training loss ----

    def loss(self, x_cont: torch.Tensor, x_cat: torch.Tensor,
              cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_cont: (B, n_continuous) standardized floats, possibly n_continuous=0.
        x_cat: (B, n_categorical) long class ids, possibly n_categorical=0.
        """
        device = next(self.parameters()).device
        B = x_cont.shape[0] if self.n_continuous > 0 else x_cat.shape[0]
        t = torch.randint(0, self.schedule.T, (B,), device=device)

        loss = 0.0

        if self.n_continuous > 0:
            xt_cont, eps_true = self._q_sample_cont(x_cont, t)
        else:
            xt_cont = x_cont  # (B, 0)
            eps_true = None

        xt_cat_onehots = []
        x0_cat_targets = []
        for i, k in enumerate(self.cat_num_classes):
            x0_oh = F.one_hot(x_cat[:, i], num_classes=k).float()
            xt_oh = self._q_sample_cat_onehot(x0_oh, t, k)
            xt_cat_onehots.append(xt_oh)
            x0_cat_targets.append(x_cat[:, i])
        xt_cat = torch.cat(xt_cat_onehots, dim=-1) if xt_cat_onehots else torch.zeros(B, 0, device=device)

        eps_pred, cat_logits = self.denoiser(xt_cont, xt_cat, t, cond)

        if eps_true is not None:
            loss = loss + F.mse_loss(eps_pred, eps_true)

        for logits, target in zip(cat_logits, x0_cat_targets):
            loss = loss + F.cross_entropy(logits, target)

        return loss

    # ---- sampling (reverse process) ----

    @torch.no_grad()
    def sample(self, n: int, cond: Optional[torch.Tensor] = None, device=None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_cont, x_cat): generated standardized continuous features
        and integer-encoded categorical features. Caller is responsible for
        un-standardizing x_cont back to the original scale.
        """
        device = device or next(self.parameters()).device
        T = self.schedule.T

        x_cont = torch.randn(n, self.n_continuous, device=device) if self.n_continuous > 0 \
            else torch.zeros(n, 0, device=device)
        x_cat_onehot = [
            torch.distributions.OneHotCategorical(probs=torch.full((n, k), 1.0 / k, device=device)).sample()
            for k in self.cat_num_classes
        ]

        for step in reversed(range(T)):
            t = torch.full((n,), step, device=device, dtype=torch.long)
            xt_cat = torch.cat(x_cat_onehot, dim=-1) if x_cat_onehot else torch.zeros(n, 0, device=device)
            eps_pred, cat_logits = self.denoiser(x_cont, xt_cat, t, cond)

            # continuous reverse step (standard DDPM posterior mean)
            if self.n_continuous > 0:
                beta_t = self.schedule.betas[step]
                alpha_t = self.schedule.alphas[step]
                a_bar_t = self.schedule.alphas_cumprod[step]
                mean = (1.0 / torch.sqrt(alpha_t)) * (
                    x_cont - (beta_t / torch.sqrt(1 - a_bar_t)) * eps_pred
                )
                if step > 0:
                    noise = torch.randn_like(x_cont)
                    x_cont = mean + torch.sqrt(beta_t) * noise
                else:
                    x_cont = mean

            # categorical reverse step: use predicted x0 distribution to
            # resample x_{t-1} from the D3PM uniform posterior
            new_onehots = []
            for i, k in enumerate(self.cat_num_classes):
                x0_probs = F.softmax(cat_logits[i], dim=-1)  # (n, k)
                if step > 0:
                    a_bar_t = self.schedule.alphas_cumprod[step]
                    a_bar_tm1 = self.schedule.alphas_cumprod[step - 1]
                    beta_t = self.schedule.betas[step]
                    xt_oh = x_cat_onehot[i]
                    # D3PM uniform posterior q(x_{t-1}|x_t,x0) closed form,
                    # approximated by mixing predicted-x0 with current xt
                    # weighted by how much signal remains.
                    post = a_bar_tm1 * x0_probs + (1 - a_bar_tm1) / k
                    post = post * ((1 - beta_t) * xt_oh + beta_t / k).sum(-1, keepdim=True)  # renorm hook
                    post = post / post.sum(-1, keepdim=True).clamp_min(1e-8)
                    sampled = torch.distributions.OneHotCategorical(probs=post).sample()
                else:
                    sampled = torch.distributions.OneHotCategorical(probs=x0_probs).sample()
                new_onehots.append(sampled)
            x_cat_onehot = new_onehots

        x_cat = torch.stack([oh.argmax(dim=-1) for oh in x_cat_onehot], dim=-1) if x_cat_onehot \
            else torch.zeros(n, 0, dtype=torch.long, device=device)
        return x_cont, x_cat


# ---------------------------------------------------------------------------
# Standardization helpers for continuous columns
# ---------------------------------------------------------------------------

@dataclass
class Standardizer:
    mean: torch.Tensor
    std: torch.Tensor

    @staticmethod
    def fit(x_cont: torch.Tensor) -> "Standardizer":
        if x_cont.shape[-1] == 0:
            return Standardizer(torch.zeros(0), torch.ones(0))
        mean = x_cont.mean(dim=0)
        std = x_cont.std(dim=0).clamp_min(1e-6)
        return Standardizer(mean, std)

    def transform(self, x_cont: torch.Tensor) -> torch.Tensor:
        if x_cont.shape[-1] == 0:
            return x_cont
        return (x_cont - self.mean.to(x_cont.device)) / self.std.to(x_cont.device)

    def inverse(self, x_cont: torch.Tensor) -> torch.Tensor:
        if x_cont.shape[-1] == 0:
            return x_cont
        return x_cont * self.std.to(x_cont.device) + self.mean.to(x_cont.device)

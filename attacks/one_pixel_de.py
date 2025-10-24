# attacks/one_pixel_de.py
from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ---- constants (avoid "magic numbers") ----
_CHANNELS = 3
_TENSOR_NDIMS = 3


@dataclass
class AttackParams:
    pop_size: int = 400
    F: float = 0.5  # mutation scale
    Cr: float = 0.9  # binomial crossover prob
    max_gens: int = 100
    es_target: float = 0.90  # targeted early-stop: p(target) >= es_target
    es_nonadv: float = 0.05  # untargeted early-stop: p(true) <= es_nonadv
    seed: int | None = None
    device: str = "mps"


@dataclass
class AttackResult:
    success: bool
    adv_label: int
    adv_conf: float
    gens_used: int
    queries: int
    best_tuple: tuple[int, int, float, float, float]  # (x,y,r,g,b)
    best_fitness: float
    pred_before: int
    conf_before: float


@dataclass
class _FinalizeMeta:
    """Bundle meta to keep _finalize within Ruff's arg-count rule."""

    best_fitness: float
    pred_before: int
    conf_before: float
    queries: int
    gens_used: int


def _set_seeds(seed: int | None) -> None:
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _cifar10_normalize(img: torch.Tensor) -> torch.Tensor:
    """Default CIFAR-10 normalization (0..1 -> normalized)."""
    mean = torch.tensor([0.4914, 0.4822, 0.4465], dtype=img.dtype, device=img.device)[
        :, None, None
    ]
    std = torch.tensor([0.2470, 0.2435, 0.2616], dtype=img.dtype, device=img.device)[
        :, None, None
    ]
    return (img - mean) / std


class OnePixelDEAttack:
    """
    Differential Evolution (DE/rand/1/bin) one-pixel attack.
    Works in pre-normalized 0..1 RGB space; `preprocess` applies dataset normalization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        preprocess: Callable[[torch.Tensor], torch.Tensor] | None = None,
        device: str | None = None,
        params: AttackParams | None = None,
    ) -> None:
        self.model = model.eval()
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.params = params or AttackParams(device=self.device)
        self.preprocess = preprocess or _cifar10_normalize
        _set_seeds(self.params.seed)

    @torch.no_grad()
    def run(
        self,
        image_01: torch.Tensor,  # [3,H,W], float32 in [0,1]
        true_label: int,
        mode: str = "untargeted",
        target_label: int | None = None,
    ) -> AttackResult:
        assert (
            image_01.ndim == _TENSOR_NDIMS and image_01.shape[0] == _CHANNELS
        ), "image must be [3,H,W]"
        assert mode in {"untargeted", "targeted"}
        if mode == "targeted":
            assert target_label is not None, "target_label required for targeted mode"

        base_img = image_01.to(self.device, dtype=torch.float32).clone()
        _, H, W = base_img.shape

        pred_before, conf_before = self._predict_top1(base_img)

        target_idx = target_label if mode == "targeted" else true_label

        P = self.params.pop_size
        pop = self._init_population(P, H, W, device=self.device)  # [P,5]
        fit = self._evaluate_population(base_img, pop, target_idx, mode)  # [P]
        queries = P

        # Early stop on initial population
        if self._early_stop_condition(fit, mode):
            i_best = torch.argmax(fit).item()
            meta = _FinalizeMeta(
                best_fitness=fit[i_best].item(),
                pred_before=pred_before,
                conf_before=conf_before,
                queries=queries,
                gens_used=0,
            )
            return self._finalize(base_img, pop[i_best], meta)

        best_fit, i_best = torch.max(fit, dim=0)

        gens_used = 0
        for g in range(self.params.max_gens):
            gens_used = g + 1
            trial = self._de_trial(pop)  # [P,5]
            fit_trial = self._evaluate_population(base_img, trial, target_idx, mode)
            queries += P

            improved = fit_trial > fit
            pop[improved] = trial[improved]
            fit[improved] = fit_trial[improved]

            curr_best_fit, curr_i_best = torch.max(fit, dim=0)
            if curr_best_fit > best_fit:
                best_fit = curr_best_fit
                i_best = curr_i_best

            if self._early_stop_condition(fit, mode):
                break

        meta = _FinalizeMeta(
            best_fitness=best_fit.item(),
            pred_before=pred_before,
            conf_before=conf_before,
            queries=queries,
            gens_used=gens_used,
        )
        return self._finalize(base_img, pop[i_best], meta)

    # ---------- internals ----------

    @torch.no_grad()
    def _predict_top1(self, img01: torch.Tensor) -> tuple[int, float]:
        x = self.preprocess(img01[None, ...])  # [1,3,H,W]
        logits = self.model(x)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.item(), conf.item()

    def _init_population(self, P: int, H: int, W: int, device: str) -> torch.Tensor:
        # x,y uniform over image; rgb ~ Normal(0.5, 0.5) clipped to [0,1]
        xy = torch.empty(P, 2, device=device).uniform_(0, 1)
        xy[:, 0] *= W - 1
        xy[:, 1] *= H - 1
        rgb = torch.empty(P, 3, device=device).normal_(0.5, 0.5).clamp_(0.0, 1.0)
        return torch.cat([xy, rgb], dim=1)  # [P,5]

    def _de_trial(self, pop: torch.Tensor) -> torch.Tensor:
        """One generation: mutation (rand/1) + binomial crossover."""
        P, D = pop.shape  # D=5
        idx = torch.arange(P, device=pop.device)

        # Choose r1,r2,r3 distinct and != i
        r1 = (idx + torch.randint(1, P, (P,), device=pop.device)) % P
        r2 = (idx + torch.randint(1, P - 1, (P,), device=pop.device) + 1) % P
        r3 = (idx + torch.randint(1, P - 2, (P,), device=pop.device) + 2) % P

        donor = pop[r1] + self.params.F * (pop[r2] - pop[r3])  # mutation

        # Binomial crossover
        cross_mask = torch.rand(P, D, device=pop.device) < self.params.Cr
        j_rand = torch.randint(
            0, D, (P,), device=pop.device
        )  # ensure at least one donor dim
        cross_mask[idx, j_rand] = True
        trial = torch.where(cross_mask, donor, pop)

        # Clamp RGB now; x,y clamped/rounded on application
        trial[:, 2:5].clamp_(0.0, 1.0)
        return trial

    @torch.no_grad()
    def _evaluate_population(
        self,
        base_img01: torch.Tensor,  # [3,H,W] in 0..1
        pop: torch.Tensor,  # [P,5] (x,y,r,g,b) floats
        target_idx: int,
        mode: str,
    ) -> torch.Tensor:
        """Vectorized population evaluation (one forward pass). Returns fitness to MAXIMIZE."""
        P = pop.shape[0]
        _, H, W = base_img01.shape
        batch = base_img01.repeat(P, 1, 1, 1)  # [P,3,H,W]

        # Round/clamp x,y -> ints; clamp rgb
        x = pop[:, 0].round().clamp_(0, W - 1).to(torch.int64)
        y = pop[:, 1].round().clamp_(0, H - 1).to(torch.int64)
        rgb = pop[:, 2:5].clamp(0.0, 1.0)

        # Vectorized pixel write
        batch[torch.arange(P, device=batch.device), :, y, x] = rgb

        x_in = self.preprocess(batch)
        logits = self.model(x_in)
        probs = F.softmax(logits, dim=1)

        fitness = (
            probs[:, target_idx] if mode == "targeted" else (1.0 - probs[:, target_idx])
        )
        return fitness

    def _success_from_fitness(self, fitness: torch.Tensor, mode: str) -> torch.Tensor:
        if mode == "targeted":
            return fitness >= self.params.es_target
        return fitness >= (1.0 - self.params.es_nonadv)  # 1 - p(true) >= 1 - es_nonadv

    def _early_stop_condition(self, fitness: torch.Tensor, mode: str) -> bool:
        return bool(self._success_from_fitness(fitness, mode).any())

    @torch.no_grad()
    def _finalize(
        self,
        base_img01: torch.Tensor,
        best_tuple: torch.Tensor,  # [5]
        meta: _FinalizeMeta,
    ) -> AttackResult:
        _, H, W = base_img01.shape
        x = int(torch.round(best_tuple[0]).clamp(0, W - 1).item())
        y = int(torch.round(best_tuple[1]).clamp(0, H - 1).item())
        rgb = best_tuple[2:5].clamp(0.0, 1.0)

        adv = base_img01.clone()
        adv[:, y, x] = rgb
        adv_label, adv_conf = self._predict_top1(adv)

        success = adv_label != meta.pred_before
        return AttackResult(
            success=success,
            adv_label=adv_label,
            adv_conf=adv_conf,
            gens_used=meta.gens_used,
            queries=meta.queries,
            best_tuple=(x, y, float(rgb[0]), float(rgb[1]), float(rgb[2])),
            best_fitness=meta.best_fitness,
            pred_before=meta.pred_before,
            conf_before=meta.conf_before,
        )

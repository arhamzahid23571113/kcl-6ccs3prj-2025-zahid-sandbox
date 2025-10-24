from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from attacks.one_pixel_de import (
    _CHANNELS,
    _TENSOR_NDIMS,
    AttackParams,
    AttackResult,
    OnePixelDEAttack,
)
from explanations.rise import RISE


@dataclass
class RiseGuideParams:
    temperature: float = 2.0
    mutate_prob: float = 0.0
    masks: int = 500
    s: int = 7
    p: float = 0.5
    batch: int = 64


class OnePixelDERISEGuided(OnePixelDEAttack):
    def __init__(
        self,
        model: torch.nn.Module,
        preprocess,
        device: str | None,
        params: AttackParams,
        guide: RiseGuideParams | None = None,
    ) -> None:
        super().__init__(model=model, preprocess=preprocess, device=device, params=params)
        self.guide = guide or RiseGuideParams()
        self._rise = RISE(
            model=self.model,
            n_masks=self.guide.masks,
            s=self.guide.s,
            p=self.guide.p,
            batch=self.guide.batch,
        )
        self._prior_weights: torch.Tensor | None = None
        self._prior_hw: torch.Tensor | None = None

    @torch.no_grad()
    def _build_prior(self, img01: torch.Tensor, target_class: int) -> None:
        x_in = self.preprocess(img01[None, ...])
        sal = self._rise.generate(x_in, target_class).squeeze(0).squeeze(0)
        sal = (sal + 1e-12).pow(self.guide.temperature)
        flat = sal.reshape(-1)
        self._prior_weights = (flat / flat.sum()).to(img01.device)

    @torch.no_grad()
    def _sample_xy_from_prior(self, P: int, H: int, W: int) -> torch.Tensor:
        assert self._prior_weights is not None
        idx = torch.multinomial(self._prior_weights, num_samples=P, replacement=True)
        y = torch.div(idx, W, rounding_mode="floor")
        x = idx % W
        return torch.stack([x, y], dim=1).to(dtype=torch.float32, device=self._prior_weights.device)

    def _init_population(self, P: int, H: int, W: int, device: str) -> torch.Tensor:
        if self._prior_weights is not None:
            xy = self._sample_xy_from_prior(P, H, W)
        else:
            xy = torch.empty(P, 2, device=device).uniform_(0, 1)
            xy[:, 0] *= W - 1
            xy[:, 1] *= H - 1
        rgb = torch.empty(P, 3, device=device).normal_(0.5, 0.5).clamp_(0.0, 1.0)
        return torch.cat([xy, rgb], dim=1)

    def _de_trial(self, pop: torch.Tensor) -> torch.Tensor:
        trial = super()._de_trial(pop)
        if self._prior_weights is None or self.guide.mutate_prob <= 0.0:
            return trial
        P, _ = trial.shape
        device = trial.device
        mask = torch.rand(P, device=device) < self.guide.mutate_prob
        if mask.any():
            assert self._prior_hw is not None
            H = int(self._prior_hw[0].item())
            W = int(self._prior_hw[1].item())
            xy_new = self._sample_xy_from_prior(int(mask.sum().item()), H, W)
            trial[mask, 0:2] = xy_new
        return trial

    @torch.no_grad()
    def run(
        self,
        image_01: torch.Tensor,
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
        target_idx_for_rise = pred_before if mode == "untargeted" else target_label  # type: ignore[arg-type]
        self._build_prior(base_img, target_idx_for_rise)
        self._prior_hw = torch.tensor([H, W], device=self.device)

        target_idx = target_label if mode == "targeted" else true_label

        P = self.params.pop_size
        pop = self._init_population(P, H, W, device=self.device)
        fit = self._evaluate_population(base_img, pop, target_idx, mode)
        queries = P

        if self._early_stop_condition(fit, mode):
            i_best = torch.argmax(fit).item()
            return self._finalize(
                base_img,
                pop[i_best],
                best_fitness=fit[i_best].item(),
                pred_before=pred_before,
                conf_before=conf_before,
                queries=queries,
                gens_used=0,
            )

        best_fit, i_best = torch.max(fit, dim=0)
        gens_used = 0
        for g in range(self.params.max_gens):
            gens_used = g + 1
            trial = self._de_trial(pop)
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

        return self._finalize(
            base_img,
            pop[i_best],
            best_fitness=best_fit.item(),
            pred_before=pred_before,
            conf_before=conf_before,
            queries=queries,
            gens_used=gens_used,
        )

    @torch.no_grad()
    def _finalize(  # noqa: PLR0913
        self,
        base_img01: torch.Tensor,
        best_tuple: torch.Tensor,
        *,
        best_fitness: float,
        pred_before: int,
        conf_before: float,
        queries: int,
        gens_used: int,
    ) -> AttackResult:
        _, H, W = base_img01.shape
        x = int(torch.round(best_tuple[0]).clamp(0, W - 1).item())
        y = int(torch.round(best_tuple[1]).clamp(0, H - 1).item())
        rgb = best_tuple[2:5].clamp(0.0, 1.0)

        adv = base_img01.clone()
        adv[:, y, x] = rgb
        x_in = self.preprocess(adv[None, ...])
        logits = self.model(x_in)
        probs = F.softmax(logits, dim=1)
        adv_conf, adv_label = torch.max(probs, dim=1)

        success = int(adv_label.item()) != pred_before
        return AttackResult(
            success=bool(success),
            adv_label=int(adv_label.item()),
            adv_conf=float(adv_conf.item()),
            gens_used=gens_used,
            queries=queries,
            best_tuple=(x, y, float(rgb[0]), float(rgb[1]), float(rgb[2])),
            best_fitness=best_fitness,
            pred_before=pred_before,
            conf_before=conf_before,
        )

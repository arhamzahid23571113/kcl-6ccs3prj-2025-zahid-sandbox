from __future__ import annotations

import random
from contextlib import nullcontext
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# -------------------- constants --------------------
_TENSOR_NDIMS = 3
_CHANNELS = 3
_RGB_DIMS = 3


@dataclass
class MTParams:
    pop_size: int = 192
    max_gens: int = 40
    F: float = 0.5
    Cr: float = 0.9
    es_target: float = 0.90
    xy_topk: int = 32
    xy_mutate_prob: float = 0.10
    seed: int | None = 1337
    device: str = "mps"
    # If True, allow AMP on CUDA (never on MPS/CPU).
    use_amp: bool = True


@dataclass
class MTResult:
    success_mask: torch.Tensor  # [T] bool
    best_tuple_per_tgt: list[tuple[int, int, float, float, float]]  # (x,y,r,g,b)
    adv_label_per_tgt: list[int]
    adv_conf_per_tgt: list[float]
    gens_used: int
    queries: int
    pred_before: int
    conf_before: float


class OnePixelDEMultiTarget:
    """
    Multi-target, prior-guided DE for one-pixel attack.

    Speedups:
      * Vectorized evaluation over (targets x population).
      * (x,y) restricted to top-K coords (union across targets).
      * AMP only on CUDA.
      * RGB mutates with DE; (x,y) occasionally resampled from per-target categorical prior.
      * Early stop per target.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str,
        mean: torch.Tensor,  # [3]
        std: torch.Tensor,  # [3]
        params: MTParams,
    ) -> None:
        self.model = model.eval()
        self.device = device
        self.mean = mean.to(device).view(3, 1, 1)
        self.std = std.to(device).view(3, 1, 1)
        self.p = params
        if params.seed is not None:
            random.seed(params.seed)
            torch.manual_seed(params.seed)

    # ------------ utils ------------
    def _amp_ctx(self):
        """AMP context: enabled on CUDA only."""
        if self.p.use_amp and (self.device == "cuda"):
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()

    @torch.no_grad()
    def _preprocess01_to_norm(self, x01: torch.Tensor) -> torch.Tensor:
        # x01 [3,H,W] -> normalized
        return (x01 - self.mean) / self.std

    @torch.no_grad()
    def _predict_top1_norm(self, x_norm: torch.Tensor) -> tuple[int, float]:
        # x_norm [1,3,H,W] already normalized
        with self._amp_ctx():
            logits = self.model(x_norm)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return int(pred.item()), float(conf.item())

    @torch.no_grad()
    def _predict_batch_top1_norm(
        self, x_norm_batch: torch.Tensor
    ) -> tuple[list[int], list[float]]:
        # x_norm_batch [B,3,H,W]
        with self._amp_ctx():
            logits = self.model(x_norm_batch)
        probs = F.softmax(logits, dim=1)
        conf, pred = torch.max(probs, dim=1)
        return pred.int().tolist(), conf.float().tolist()

    # ------------ coordinate pool ------------
    @torch.no_grad()
    def build_coord_pool(
        self,
        H: int,
        W: int,
        per_target_cam: list[torch.Tensor],  # each [1,1,H,W], 0..1
        topk: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          coords [K,2] int64 as (x,y)
          weights_per_tgt [T,K] normalized categorical weights (>= eps)
        """
        eps = 1e-8

        # union of top-k per target
        flat_coords: list[tuple[int, int]] = []
        for cam in per_target_cam:
            sal = cam.view(-1)
            k = min(topk, sal.numel())
            top_idx = torch.topk(sal, k=k, dim=0).indices
            for idx in top_idx.tolist():
                y = idx // W
                x = idx % W
                flat_coords.append((x, y))
        # deduplicate while preserving order
        seen = set()
        uniq: list[tuple[int, int]] = []
        for xy in flat_coords:
            if xy not in seen:
                seen.add(xy)
                uniq.append(xy)
        if not uniq:
            uniq = [(W // 2, H // 2)]
        if len(uniq) > topk:
            uniq = uniq[:topk]
        K = len(uniq)
        coords = torch.tensor(uniq, device=self.device, dtype=torch.long)  # [K,2]

        # per-target weights from cams at those coords
        weights = []
        for cam in per_target_cam:
            vals = cam[0, 0, coords[:, 1], coords[:, 0]].clamp(min=0.0)  # [K]
            s = vals.sum()
            vals = (
                torch.full_like(vals, 1.0 / K) if float(s.item()) <= eps else (vals / s)
            )
            weights.append(vals)
        weights_per_tgt = torch.stack(weights, dim=0)  # [T,K]
        return coords, weights_per_tgt

    # ------------ core DE (RGB only) ------------
    @torch.no_grad()
    def run(  # noqa: PLR0913, PLR0915
        self,
        img01: torch.Tensor,  # [3,H,W], float32 [0,1]
        true_label: int,
        target_labels: list[int],  # e.g., all classes except true
        coord_pool: torch.Tensor,  # [K,2] long (x,y)
        coord_weights: torch.Tensor,  # [T,K] float (categorical per target)
        mode: str = "targeted",
    ) -> MTResult:
        assert img01.ndim == _TENSOR_NDIMS and img01.shape[0] == _CHANNELS
        assert mode == "targeted", "MT runner is for targeted sweep"
        T = len(target_labels)
        _, H, W = img01.shape  # noqa: F841  (kept for clarity)

        # base normalized image once
        base_norm = self._preprocess01_to_norm(img01).to(self.device)  # [3,H,W]
        pred_before, conf_before = self._predict_top1_norm(base_norm.unsqueeze(0))

        # populations per target: (idx, r, g, b) where rgb in [0,1]
        P = self.p.pop_size
        idx = torch.multinomial(coord_weights, num_samples=P, replacement=True)  # [T,P]
        rgb = torch.empty(T, P, _RGB_DIMS, device=self.device).uniform_(0.0, 1.0)
        pop = torch.cat([idx.unsqueeze(-1).to(torch.float32), rgb], dim=-1)  # [T,P,4]

        best = pop.clone()
        best_fit = self._evaluate_fitness_batch(
            base_norm, best, coord_pool, target_labels
        )  # [T,P]
        queries = T * P

        hit_tgt = (best_fit >= self.p.es_target).any(dim=1)  # [T]
        if hit_tgt.all():
            gens_used = 0
            return self._finalize(
                base_norm,
                best,
                best_fit,
                coord_pool,
                target_labels,
                hit_tgt,
                pred_before,
                conf_before,
                gens_used,
                queries,
            )

        gens_used = 0
        rng = torch.Generator(device=self.device)
        if self.p.seed is not None:
            rng.manual_seed(self.p.seed)

        for g in range(self.p.max_gens):
            gens_used = g + 1
            # --- DE on RGB (3 dims) ---
            idxs = torch.arange(P, device=self.device)
            r1 = (
                idxs + torch.randint(1, P, (P,), generator=rng, device=self.device)
            ) % P
            r2 = (
                idxs
                + torch.randint(1, P - 1, (P,), generator=rng, device=self.device)
                + 1
            ) % P
            r3 = (
                idxs
                + torch.randint(1, P - 2, (P,), generator=rng, device=self.device)
                + 2
            ) % P
            r1 = r1.view(1, P).expand(T, P)
            r2 = r2.view(1, P).expand(T, P)
            r3 = r3.view(1, P).expand(T, P)

            donor_rgb = best[torch.arange(T).unsqueeze(1), r1, 1:] + self.p.F * (
                best[torch.arange(T).unsqueeze(1), r2, 1:]
                - best[torch.arange(T).unsqueeze(1), r3, 1:]
            )
            donor_rgb = donor_rgb.clamp(0.0, 1.0)

            # binomial crossover (ensure at least one donor dim per (t,p))
            cross_mask = (
                torch.rand(T, P, _RGB_DIMS, device=self.device, generator=rng)
                < self.p.Cr
            )
            j_rand = torch.randint(
                0, _RGB_DIMS, (T, P), generator=rng, device=self.device
            )
            t_idx = torch.arange(T, device=self.device)[:, None].expand(T, P)
            p_idx = torch.arange(P, device=self.device)[None, :].expand(T, P)
            cross_mask[t_idx, p_idx, j_rand] = True
            trial_rgb = torch.where(cross_mask, donor_rgb, best[:, :, 1:]).clamp_(
                0.0, 1.0
            )

            # occasionally mutate coordinate index by sampling per-target categorical prior
            trial_idx = best[:, :, 0]
            mutate_mask = (
                torch.rand(T, P, device=self.device, generator=rng)
                < self.p.xy_mutate_prob
            )
            if mutate_mask.any():
                for t in range(T):
                    m = mutate_mask[t]
                    if m.any():
                        to_sample = int(m.sum().item())
                        new_idx = torch.multinomial(
                            coord_weights[t].clamp(min=1e-8),
                            num_samples=to_sample,
                            replacement=True,
                        )
                        trial_idx[t, m] = new_idx.to(trial_idx.dtype)

            trial = torch.cat([trial_idx.unsqueeze(-1), trial_rgb], dim=-1)  # [T,P,4]

            # evaluate
            fit_trial = self._evaluate_fitness_batch(
                base_norm, trial, coord_pool, target_labels
            )  # [T,P]
            queries += T * P

            improved = fit_trial > best_fit
            best[improved] = trial[improved]
            best_fit[improved] = fit_trial[improved]

            # early stop per target
            hit_tgt = hit_tgt | (best_fit >= self.p.es_target).any(dim=1)
            if hit_tgt.all():
                break

        return self._finalize(
            base_norm,
            best,
            best_fit,
            coord_pool,
            target_labels,
            hit_tgt,
            pred_before,
            conf_before,
            gens_used,
            queries,
        )

    @torch.no_grad()
    def _evaluate_fitness_batch(
        self,
        base_norm: torch.Tensor,  # [3,H,W]
        pop: torch.Tensor,  # [T,P,4] (idx,r,g,b)
        coord_pool: torch.Tensor,  # [K,2] (x,y)
        target_labels: list[int],
    ) -> torch.Tensor:
        T, P, _ = pop.shape
        K = coord_pool.size(0)
        _, H, W = base_norm.shape

        idx = pop[:, :, 0].long().clamp(0, K - 1)  # [T,P]
        rgb01 = pop[:, :, 1:].clamp(0.0, 1.0)  # [T,P,3]
        rgb_norm = (rgb01 - self.mean.view(1, 1, 3)) / self.std.view(1, 1, 3)

        xy = coord_pool[idx]  # [T,P,2]
        x = xy[..., 0].view(-1)  # [T*P]
        y = xy[..., 1].view(-1)  # [T*P]
        rgbn = rgb_norm.view(-1, _RGB_DIMS)  # [T*P,3]

        # build batch [T*P, 3, H, W] from base_norm
        batch = base_norm.unsqueeze(0).expand(T * P, -1, -1, -1).clone()
        ar = torch.arange(T * P, device=batch.device)
        batch[ar, :, y, x] = rgbn

        with self._amp_ctx():
            logits = self.model(batch)  # [T*P, C]
        probs = logits.softmax(dim=1)
        tgt = (
            torch.as_tensor(target_labels, device=batch.device, dtype=torch.long)
            .view(T, 1)
            .expand(T, P)
            .reshape(-1, 1)
        )  # [T*P,1]
        fit = probs.gather(1, tgt).view(T, P)  # [T,P]
        return fit

    @torch.no_grad()
    def _finalize(  # noqa: PLR0913
        self,
        base_norm: torch.Tensor,
        best: torch.Tensor,  # [T,P,4]
        best_fit: torch.Tensor,  # [T,P]
        coord_pool: torch.Tensor,  # [K,2]
        target_labels: list[int],
        hit_tgt: torch.Tensor,  # [T] bool
        pred_before: int,
        conf_before: float,
        gens_used: int,
        queries: int,
    ) -> MTResult:
        T, P, _ = best.shape
        K = coord_pool.size(0)  # noqa: F841

        # pick the best individual per target
        i_best = torch.argmax(best_fit, dim=1)  # [T]
        rows = best[torch.arange(T, device=best.device), i_best]  # [T,4]
        idx = rows[:, 0].long().clamp(0, K - 1)
        rgb01 = rows[:, 1:].clamp(0.0, 1.0)  # [T,3]

        xy = coord_pool[idx]  # [T,2]
        x_list = xy[:, 0].tolist()
        y_list = xy[:, 1].tolist()
        best_tuple = [
            (
                int(x_list[i]),
                int(y_list[i]),
                float(rgb01[i, 0]),
                float(rgb01[i, 1]),
                float(rgb01[i, 2]),
            )
            for i in range(T)
        ]

        # batch compute adv labels / confs for the chosen tuples
        xn = (rgb01.view(T, _RGB_DIMS, 1, 1) - self.mean) / self.std  # [T,3,1,1]
        batch = base_norm.unsqueeze(0).expand(T, -1, -1, -1).clone()  # [T,3,H,W]
        ar = torch.arange(T, device=batch.device)
        batch[ar, :, xy[:, 1], xy[:, 0]] = xn.view(T, _RGB_DIMS)  # set the chosen pixel
        with self._amp_ctx():
            logits = self.model(batch)
        probs = logits.softmax(dim=1)
        adv_conf, adv_label = torch.max(probs, dim=1)

        return MTResult(
            success_mask=hit_tgt.bool(),
            best_tuple_per_tgt=best_tuple,
            adv_label_per_tgt=adv_label.int().tolist(),
            adv_conf_per_tgt=adv_conf.float().tolist(),
            gens_used=gens_used,
            queries=queries,
            pred_before=pred_before,
            conf_before=conf_before,
        )

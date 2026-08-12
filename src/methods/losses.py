import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal loss for imbalanced classification with optional masking."""

    def __init__(self, gamma=2.0, alpha=None, reduction='mean'):
        super().__init__()
        self.gamma = float(gamma)
        self.alpha = alpha
        self.reduction = reduction

    def _resolve_alpha(self, target, device):
        if self.alpha is None:
            return None
        if self.alpha == 'balanced':
            if target.numel() == 0:
                return None
            classes = torch.unique(target)
            counts = torch.bincount(target.to(dtype=torch.long), minlength=int(classes.max().item()) + 1)
            counts = counts[counts > 0]
            if counts.numel() == 0:
                return None
            # Capped for the same reason as MAX_POS_WEIGHT in experiments_supervised.py's
            # _build_loss_criterion: on extreme class imbalance (e.g. IBM HI-Small, where this
            # ratio reaches ~1959.8), an uncapped weight can saturate the gradient and contribute
            # to NaN blowups rather than just correcting for class frequency.
            weights = torch.clamp(counts.max() / counts, max=50.0)
            if classes.numel() > 0:
                weight_map = torch.zeros(int(classes.max().item()) + 1, dtype=torch.float32, device=device)
                for idx, cls in enumerate(classes.tolist()):
                    weight_map[int(cls)] = weights[idx]
                return weight_map
            return weights
        if isinstance(self.alpha, (list, tuple)):
            return torch.tensor(self.alpha, dtype=torch.float32, device=device)
        if isinstance(self.alpha, torch.Tensor):
            return self.alpha.to(device=device, dtype=torch.float32)
        return torch.tensor(float(self.alpha), dtype=torch.float32, device=device)

    def forward(self, logits, target, mask=None):
        if logits.ndim != 2:
            raise ValueError(f"Expected logits with shape [N, C], got {tuple(logits.shape)}")
        if target.ndim != 1:
            raise ValueError(f"Expected target with shape [N], got {tuple(target.shape)}")
        if logits.shape[0] != target.shape[0]:
            raise ValueError(f"Logits and target batch size mismatch: {logits.shape[0]} vs {target.shape[0]}")

        if mask is not None:
            mask = mask.to(device=logits.device, dtype=torch.bool)
            if mask.ndim != 1:
                raise ValueError(f"Expected mask with shape [N], got {tuple(mask.shape)}")
            if mask.shape[0] != target.shape[0]:
                raise ValueError(f"Mask and target size mismatch: {mask.shape[0]} vs {target.shape[0]}")
            valid = mask
            if not valid.any():
                return logits.sum() * 0.0
            logits = logits[valid]
            target = target[valid]

        if target.numel() == 0:
            return logits.sum() * 0.0

        log_probs = F.log_softmax(logits, dim=-1)
        target_indices = target.to(device=logits.device, dtype=torch.long)
        log_pt = log_probs[torch.arange(target_indices.numel(), device=logits.device), target_indices]
        pt = torch.exp(log_pt)
        ce = -log_pt
        alpha = self._resolve_alpha(target_indices, logits.device)
        if alpha is not None:
            alpha_t = alpha[target_indices]
            if alpha_t.ndim == 0:
                alpha_t = alpha_t.unsqueeze(0)
            ce = alpha_t * ce
        loss = ce * ((1.0 - pt) ** self.gamma)
        if self.reduction == 'mean':
            return loss.mean()
        if self.reduction == 'sum':
            return loss.sum()
        if self.reduction == 'none':
            return loss
        raise ValueError(f"Unsupported reduction: {self.reduction}")

import torch
from torch import nn
import torch.nn.functional as F


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class ScaleShiftInvariantLoss(nn.Module):
    """
    Scale and shift invariant loss for non-metric (relative) depth estimation.

    This loss aligns the predicted depth to the ground truth using least squares
    to find optimal scale and shift, then computes the L1 loss on aligned predictions.

    This allows training without requiring absolute depth values, only relative
    depth ordering matters.
    """
    def __init__(self, alpha=0.5, scales=4, reduction='batch-based'):
        """
        Args:
            alpha: Weight for gradient matching loss (0-1). Higher values emphasize
                   edge/gradient matching over pixel-wise loss.
            scales: Number of scales for multi-scale gradient loss
            reduction: 'batch-based' averages over batch, 'image-based' averages per image
        """
        super().__init__()
        self.alpha = alpha
        self.scales = scales
        self.reduction = reduction

    def compute_scale_and_shift(self, pred, target, mask):
        """
        Compute optimal scale and shift to align prediction to target.
        Uses least squares: target = scale * pred + shift

        Args:
            pred: Predicted depth (B, H, W) or (N,) flattened
            target: Ground truth depth (B, H, W) or (N,) flattened
            mask: Valid pixel mask (B, H, W) or (N,) flattened

        Returns:
            scale, shift: Optimal alignment parameters
        """
        # Flatten if needed
        if pred.dim() > 1:
            pred = pred.reshape(-1)
            target = target.reshape(-1)
            mask = mask.reshape(-1)

        # Only use valid pixels
        pred_valid = pred[mask]
        target_valid = target[mask]

        if pred_valid.numel() < 2:
            return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)

        # Least squares solution for: target = scale * pred + shift
        # [pred, 1] @ [scale, shift]^T = target
        # Using normal equations: A^T A x = A^T b
        pred_mean = pred_valid.mean()
        target_mean = target_valid.mean()

        pred_centered = pred_valid - pred_mean
        target_centered = target_valid - target_mean

        # Compute scale
        pred_var = (pred_centered ** 2).sum()
        if pred_var < 1e-8:
            scale = torch.tensor(1.0, device=pred.device)
        else:
            scale = (pred_centered * target_centered).sum() / pred_var

        # Compute shift
        shift = target_mean - scale * pred_mean

        # Clamp scale to reasonable range to avoid instability
        scale = scale.clamp(min=1e-3, max=1e3)

        return scale, shift

    def gradient_loss(self, pred, target, mask):
        """
        Compute multi-scale gradient matching loss.

        This encourages the model to preserve depth edges and discontinuities.
        """
        total_loss = 0.0

        for scale in range(self.scales):
            step = 2 ** scale

            # Compute gradients in x and y directions
            pred_dx = pred[:, :, :, step:] - pred[:, :, :, :-step]
            pred_dy = pred[:, :, step:, :] - pred[:, :, :-step, :]

            target_dx = target[:, :, :, step:] - target[:, :, :, :-step]
            target_dy = target[:, :, step:, :] - target[:, :, :-step, :]

            mask_dx = mask[:, :, :, step:] & mask[:, :, :, :-step]
            mask_dy = mask[:, :, step:, :] & mask[:, :, :-step, :]

            # L1 loss on gradients
            if mask_dx.sum() > 0:
                total_loss += torch.abs(pred_dx[mask_dx] - target_dx[mask_dx]).mean()
            if mask_dy.sum() > 0:
                total_loss += torch.abs(pred_dy[mask_dy] - target_dy[mask_dy]).mean()

        return total_loss / (2 * self.scales)

    def forward(self, pred, target, valid_mask):
        """
        Compute scale-shift invariant loss.

        Args:
            pred: Predicted depth (B, H, W), values in (0, 1) from sigmoid
            target: Ground truth depth (B, H, W), any positive values
            valid_mask: Boolean mask of valid pixels (B, H, W)

        Returns:
            Combined loss value
        """
        valid_mask = valid_mask.detach()

        # Ensure we have batch dimension
        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        batch_size = pred.shape[0]
        total_loss = 0.0

        # Process each image in batch
        aligned_preds = []
        for b in range(batch_size):
            pred_b = pred[b]
            target_b = target[b]
            mask_b = valid_mask[b]

            if mask_b.sum() < 10:
                # Not enough valid pixels, skip
                aligned_preds.append(pred_b)
                continue

            # Compute optimal scale and shift
            scale, shift = self.compute_scale_and_shift(pred_b, target_b, mask_b)

            # Align prediction
            aligned_pred = scale * pred_b + shift
            aligned_preds.append(aligned_pred)

            # Pixel-wise L1 loss on aligned predictions
            pixel_loss = torch.abs(aligned_pred[mask_b] - target_b[mask_b]).mean()
            total_loss += pixel_loss

        # Average over batch
        pixel_loss = total_loss / batch_size

        # Gradient loss (on aligned predictions)
        aligned_pred_batch = torch.stack(aligned_preds, dim=0).unsqueeze(1)  # (B, 1, H, W)
        target_batch = target.unsqueeze(1)  # (B, 1, H, W)
        mask_batch = valid_mask.unsqueeze(1)  # (B, 1, H, W)

        grad_loss = self.gradient_loss(aligned_pred_batch, target_batch, mask_batch)

        # Combined loss
        loss = (1 - self.alpha) * pixel_loss + self.alpha * grad_loss

        return loss


class RankingLoss(nn.Module):
    """
    Pairwise ranking loss for relative depth estimation.

    This loss only requires relative depth ordering (which pixel is closer/farther),
    not absolute depth values. Useful when training on data with ordinal depth annotations.
    """
    def __init__(self, margin=0.0, sample_ratio=0.1, max_samples=10000):
        """
        Args:
            margin: Margin for ranking loss (depth difference threshold)
            sample_ratio: Ratio of pixel pairs to sample (for efficiency)
            max_samples: Maximum number of pairs to sample per image
        """
        super().__init__()
        self.margin = margin
        self.sample_ratio = sample_ratio
        self.max_samples = max_samples

    def forward(self, pred, target, valid_mask):
        """
        Compute ranking loss.

        Args:
            pred: Predicted depth (B, H, W)
            target: Ground truth depth (B, H, W)
            valid_mask: Boolean mask of valid pixels (B, H, W)

        Returns:
            Ranking loss value
        """
        valid_mask = valid_mask.detach()

        if pred.dim() == 2:
            pred = pred.unsqueeze(0)
            target = target.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)

        batch_size = pred.shape[0]
        total_loss = 0.0
        valid_count = 0

        for b in range(batch_size):
            pred_b = pred[b]
            target_b = target[b]
            mask_b = valid_mask[b]

            # Get valid pixel indices
            valid_indices = torch.where(mask_b.flatten())[0]
            n_valid = valid_indices.shape[0]

            if n_valid < 2:
                continue

            # Sample pairs
            n_pairs = min(int(n_valid * (n_valid - 1) / 2 * self.sample_ratio), self.max_samples)
            if n_pairs < 1:
                n_pairs = min(n_valid * (n_valid - 1) // 2, self.max_samples)

            # Random sample indices
            idx1 = torch.randint(0, n_valid, (n_pairs,), device=pred.device)
            idx2 = torch.randint(0, n_valid, (n_pairs,), device=pred.device)

            # Ensure different indices
            same_mask = idx1 == idx2
            idx2[same_mask] = (idx2[same_mask] + 1) % n_valid

            # Get pixel values
            flat_pred = pred_b.flatten()
            flat_target = target_b.flatten()

            pred_i = flat_pred[valid_indices[idx1]]
            pred_j = flat_pred[valid_indices[idx2]]
            target_i = flat_target[valid_indices[idx1]]
            target_j = flat_target[valid_indices[idx2]]

            # Compute ranking loss
            # If target_i > target_j (i is farther), pred_i should be > pred_j
            target_diff = target_i - target_j
            pred_diff = pred_i - pred_j

            # Loss when ordering is wrong
            # positive target_diff means i should have larger depth
            loss = torch.relu(-target_diff.sign() * pred_diff + self.margin)

            total_loss += loss.mean()
            valid_count += 1

        if valid_count == 0:
            return torch.tensor(0.0, device=pred.device, requires_grad=True)

        return total_loss / valid_count

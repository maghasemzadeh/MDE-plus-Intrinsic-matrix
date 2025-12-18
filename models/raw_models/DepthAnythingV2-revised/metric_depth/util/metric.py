import torch


def eval_depth(pred, target):
    assert pred.shape == target.shape

    thresh = torch.max((target / pred), (pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = pred - target
    diff_log = torch.log(pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log , 2)))

    log10 = torch.mean(torch.abs(torch.log10(pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10':log10.item(), 'silog':silog.item()}


def compute_scale_and_shift(pred, target):
    """
    Compute optimal scale and shift to align prediction to target using least squares.
    Solves: target = scale * pred + shift

    Args:
        pred: Predicted depth (flattened, valid pixels only)
        target: Ground truth depth (flattened, valid pixels only)

    Returns:
        scale, shift: Optimal alignment parameters
    """
    if pred.numel() < 2:
        return torch.tensor(1.0, device=pred.device), torch.tensor(0.0, device=pred.device)

    pred_mean = pred.mean()
    target_mean = target.mean()

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    # Compute scale
    pred_var = (pred_centered ** 2).sum()
    if pred_var < 1e-8:
        scale = torch.tensor(1.0, device=pred.device)
    else:
        scale = (pred_centered * target_centered).sum() / pred_var

    # Compute shift
    shift = target_mean - scale * pred_mean

    # Clamp scale to reasonable range
    scale = scale.clamp(min=1e-3, max=1e3)

    return scale, shift


def eval_depth_scale_aligned(pred, target):
    """
    Evaluate depth with scale-shift alignment for basic (non-metric) models.

    First aligns the prediction to ground truth using least squares,
    then computes standard depth metrics on aligned predictions.

    Args:
        pred: Predicted depth (flattened tensor of valid pixels)
        target: Ground truth depth (flattened tensor of valid pixels)

    Returns:
        Dictionary of metrics computed on aligned predictions
    """
    assert pred.shape == target.shape

    # Compute optimal scale and shift
    scale, shift = compute_scale_and_shift(pred, target)

    # Align prediction
    aligned_pred = scale * pred + shift

    # Ensure positive values for log-based metrics
    aligned_pred = aligned_pred.clamp(min=1e-6)
    target = target.clamp(min=1e-6)

    # Compute metrics on aligned predictions
    thresh = torch.max((target / aligned_pred), (aligned_pred / target))

    d1 = torch.sum(thresh < 1.25).float() / len(thresh)
    d2 = torch.sum(thresh < 1.25 ** 2).float() / len(thresh)
    d3 = torch.sum(thresh < 1.25 ** 3).float() / len(thresh)

    diff = aligned_pred - target
    diff_log = torch.log(aligned_pred) - torch.log(target)

    abs_rel = torch.mean(torch.abs(diff) / target)
    sq_rel = torch.mean(torch.pow(diff, 2) / target)

    rmse = torch.sqrt(torch.mean(torch.pow(diff, 2)))
    rmse_log = torch.sqrt(torch.mean(torch.pow(diff_log, 2)))

    log10 = torch.mean(torch.abs(torch.log10(aligned_pred) - torch.log10(target)))
    silog = torch.sqrt(torch.pow(diff_log, 2).mean() - 0.5 * torch.pow(diff_log.mean(), 2))

    return {'d1': d1.item(), 'd2': d2.item(), 'd3': d3.item(), 'abs_rel': abs_rel.item(), 'sq_rel': sq_rel.item(),
            'rmse': rmse.item(), 'rmse_log': rmse_log.item(), 'log10': log10.item(), 'silog': silog.item(),
            'scale': scale.item(), 'shift': shift.item()}
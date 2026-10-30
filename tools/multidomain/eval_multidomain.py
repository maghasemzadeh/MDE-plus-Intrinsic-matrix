#!/usr/bin/env python3
"""Evaluate one metric-depth model on several held-out test sets.

Absolute metric depth, no per-image scale or shift alignment, prediction at
the DA2 inference resolution (lower-bound 518, multiple of 14) interpolated
back to GT resolution. Per-image metrics are saved so that paired tests
between runs (camera-aware model vs K-blind control) use identical images
and masks.

Test sets (datasets/multidomain.py EVAL_READERS):
  nyu            eigen 654, rawDepths, eigen crop, 1e-3..10 m
  kitti          DA2 eigen 652, no crop, 1e-3..80 m, per-date rectified K
  cityscapes     official val 500, SGM disparity GT, hood rows masked, ..80 m
  diode_indoor   DIODE val indoor 325, 1e-3..10 m
  diode_outdoor  DIODE val outdoor 446, 1e-3..80 m
  middlebury     2014 '-perfect' scenes (23), MiddEval3 half resolution, ..20 m
  sunrgbd_*      SUN RGB-D subset, 200 frames per sensor (kv1 B3DO, kv2,
                 realsense, xtion; NYU frames excluded), raw depth, ..10 m

Models:
  --checkpoint X       a train.py checkpoint (ReLU-head class; intrinsics use
                       read from its config)
  --released NAME      released DA2 metric vitl: hypersim (max 20 m) or
                       vkitti (max 80 m), original sigmoid*max_depth head
Diagnostics (camera-aware models only):
  --k-scale s          feed K with fx, fy multiplied by s (wrong focal)
  --withhold-k         run the camera-aware model without K
Camera shift:
  --center-zoom s      evaluate on the central 1/s window of every test image
                       (K's principal point shifted accordingly): the same
                       scenes seen by a camera with an s-times narrower FoV
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from datasets.multidomain import EVAL_READERS  # noqa: E402

MODEL_CONFIGS = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
}


def depth_metrics(pred, gt):
    """DA2 util/metric.py eval_depth definitions on valid pixels; predictions
    are floored at 1e-3 m so zero outputs count as errors instead of being
    dropped."""
    p = np.maximum(pred, 1e-3).astype(np.float64)
    g = gt.astype(np.float64)
    thresh = np.maximum(g / p, p / g)
    diff, diff_log = p - g, np.log(p) - np.log(g)
    return {
        'd1': float((thresh < 1.25).mean()),
        'd2': float((thresh < 1.25 ** 2).mean()),
        'd3': float((thresh < 1.25 ** 3).mean()),
        'abs_rel': float(np.mean(np.abs(diff) / g)),
        'sq_rel': float(np.mean(diff ** 2 / g)),
        'rmse': float(np.sqrt(np.mean(diff ** 2))),
        'rmse_log': float(np.sqrt(np.mean(diff_log ** 2))),
        'log10': float(np.mean(np.abs(np.log10(p) - np.log10(g)))),
        'silog': float(np.sqrt(max(np.mean(diff_log ** 2) - 0.5 * np.mean(diff_log) ** 2, 0.0))),
        'n_valid': int(g.size),
    }


class _Items(Dataset):
    def __init__(self, reader, max_items=None):
        self.reader = reader
        self.n = len(reader) if max_items is None else min(max_items, len(reader))

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        rgb, depth, K = self.reader.load(i)
        return {'bgr': cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), 'depth': depth,
                'K': K.astype(np.float32), 'id': self.reader.item_id(i)}


def load_model(args, device):
    if args.released:
        # The released metric package dir has no __init__.py, so a plain
        # `import depth_anything_v2` resolves to the revised (ReLU-head)
        # regular package instead. Mount it under its own package name.
        import importlib
        import inspect
        import types
        pkg = types.ModuleType('da2_released_metric')
        pkg.__path__ = [os.path.join(ROOT, 'models', 'raw_models', 'DepthAnythingV2',
                                     'metric_depth', 'depth_anything_v2')]
        sys.modules['da2_released_metric'] = pkg
        DepthAnythingV2 = importlib.import_module('da2_released_metric.dpt').DepthAnythingV2
        src = inspect.getsourcefile(DepthAnythingV2)
        assert os.sep.join(['DepthAnythingV2', 'metric_depth', 'depth_anything_v2']) in src, src
        # same fused attention as the revised package (numerically equivalent,
        # avoids materialising N x N maps at KITTI/Cityscapes resolutions)
        att = importlib.import_module('da2_released_metric.dinov2_layers.attention')

        def _sdpa_forward(self, x, attn_bias=None):
            B, N, C = x.shape
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            x = torch.nn.functional.scaled_dot_product_attention(qkv[0], qkv[1], qkv[2], scale=self.scale)
            return self.proj(x.transpose(1, 2).reshape(B, N, C))

        att.Attention.forward = _sdpa_forward
        att.MemEffAttention.forward = _sdpa_forward
        max_depth = {'hypersim': 20.0, 'vkitti': 80.0}[args.released]
        ckpt = os.path.join(ROOT, 'models', 'raw_models', 'DepthAnythingV2', 'checkpoints',
                            f'depth_anything_v2_metric_{args.released}_{args.encoder}.pth')
        model = DepthAnythingV2(**MODEL_CONFIGS[args.encoder], max_depth=max_depth)
        model.load_state_dict(torch.load(ckpt, map_location='cpu'))
        meta = {'checkpoint': ckpt, 'released': args.released, 'uses_intrinsics': False,
                'encoder': args.encoder}
        return model.to(device).eval(), False, meta

    sys.path.insert(0, os.path.join(ROOT, 'models', 'raw_models', 'DepthAnythingV2-revised'))
    from depth_anything_v2.dpt import DepthAnythingV2
    import inspect
    src = inspect.getsourcefile(DepthAnythingV2)
    assert os.sep.join(['DepthAnythingV2-revised', 'depth_anything_v2', 'dpt.py']) in src, src
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    assert cfg['model_type'] == 'metric', cfg
    uses_k = bool(cfg.get('use_camera_intrinsics'))
    model = DepthAnythingV2(**MODEL_CONFIGS[cfg['encoder']], use_camera_intrinsics=uses_k,
                            cam_token_inject_layer=cfg.get('cam_token_inject_layer'))
    model.load_state_dict(ckpt['model'], strict=True)
    meta = {'checkpoint': os.path.abspath(args.checkpoint), 'epoch': ckpt.get('epoch'),
            'uses_intrinsics': uses_k, 'train_config': cfg, 'encoder': cfg['encoder'],
            'canonical_focal': float(cfg.get('canonical_focal') or 0.0)}
    if uses_k:
        meta['cam_gate'] = float(model.cam_encoder.output_gate.item())
    return model.to(device).eval(), uses_k, meta


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--checkpoint')
    g.add_argument('--released', choices=['hypersim', 'vkitti'])
    p.add_argument('--encoder', default='vitl')
    p.add_argument('--datasets', default='nyu,kitti,cityscapes,diode_indoor,diode_outdoor,middlebury')
    p.add_argument('--out-dir', required=True)
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--max-items', type=int, default=None)
    p.add_argument('--k-scale', type=float, default=1.0)
    p.add_argument('--withhold-k', action='store_true')
    p.add_argument('--center-zoom', type=float, default=1.0)
    p.add_argument('--jpeg-quality', type=int, default=0,
                   help='re-encode every test image as JPEG at this quality first (0 = off)')
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, uses_k, meta = load_model(args, device)
    feed_k = uses_k and not args.withhold_k
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"model: {meta['checkpoint']} | intrinsics: {uses_k} | feed K: {feed_k} "
          f"| k_scale: {args.k_scale}", flush=True)

    overview = {}
    names = list(EVAL_READERS) if args.datasets == 'all' else \
        [d.strip() for d in args.datasets.split(',') if d.strip()]
    for name in names:
        reader = EVAL_READERS[name]()
        lo, hi = reader.eval_range
        loader = DataLoader(_Items(reader, args.max_items), batch_size=None, shuffle=False,
                            num_workers=args.workers)
        per_image, t0 = [], time.time()
        for s in loader:
            bgr, gt = np.asarray(s['bgr']), np.asarray(s['depth'], dtype=np.float32)
            K = np.asarray(s['K'], dtype=np.float32).copy()
            region = reader.eval_mask(gt.shape)
            if args.center_zoom > 1.0:
                H, W = gt.shape
                wh, ww = int(round(H / args.center_zoom)), int(round(W / args.center_zoom))
                y0, x0 = (H - wh) // 2, (W - ww) // 2
                bgr = np.ascontiguousarray(bgr[y0:y0 + wh, x0:x0 + ww])
                gt = np.ascontiguousarray(gt[y0:y0 + wh, x0:x0 + ww])
                region = region[y0:y0 + wh, x0:x0 + ww]
                K[0, 2] -= x0
                K[1, 2] -= y0
            K[0, 0] *= args.k_scale
            K[1, 1] *= args.k_scale
            if args.jpeg_quality:
                ok, buf = cv2.imencode('.jpg', bgr, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            with torch.no_grad():
                if feed_k:
                    pred = model.infer_image(bgr, args.input_size, intrinsics=K)
                else:
                    pred = model.infer_image(bgr, args.input_size)
            pred = np.asarray(pred, dtype=np.float32)
            if meta.get('canonical_focal') and feed_k:
                # canonical-camera output -> metric: x f_net / F, where the
                # inference resize maps the image short side to input_size
                pred = pred * (K[1, 1] * args.input_size / min(gt.shape) / meta['canonical_focal'])
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(gt) & (gt > lo) & (gt < hi) & region
            if valid.sum() < 10:
                continue
            m = depth_metrics(pred[valid], gt[valid])
            m['id'] = s['id']
            m['pred_median'] = float(np.median(pred[valid]))
            m['gt_median'] = float(np.median(gt[valid]))
            per_image.append(m)
        keys = ['abs_rel', 'rmse', 'd1', 'd2', 'd3', 'sq_rel', 'rmse_log', 'log10', 'silog']
        summary = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
        summary['n_images'] = len(per_image)
        summary['median_scale_ratio'] = float(np.median(
            [m['pred_median'] / m['gt_median'] for m in per_image]))
        overview[name] = summary
        out = {'dataset': name, 'eval_range': [lo, hi], 'summary': summary,
               'model': meta, 'k_scale': args.k_scale, 'withhold_k': args.withhold_k,
               'center_zoom': args.center_zoom, 'input_size': args.input_size,
               'per_image': per_image}
        with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
            json.dump(out, f, indent=1)
        print(f"{name:14s} n={summary['n_images']:4d} abs_rel={summary['abs_rel']:.4f} "
              f"rmse={summary['rmse']:.3f} d1={summary['d1']:.4f} "
              f"scale={summary['median_scale_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
        json.dump({'model': meta, 'k_scale': args.k_scale, 'withhold_k': args.withhold_k,
                   'center_zoom': args.center_zoom, 'results': overview}, f, indent=1)
    print('EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()

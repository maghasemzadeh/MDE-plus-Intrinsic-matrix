#!/usr/bin/env python3
"""Evaluate published metric-depth models under the multi-domain protocol.

Same readers, masks, depth caps, metrics and JSON schema as eval_multidomain.py,
so results/multidomain/related_* can be compared per image with md1_P / md1_C /
released_*. No per-image alignment. Each model uses its own published
pre-processing. Models that can accept a known camera receive the true K, the
same information the proposed model gets, so they are not handicapped:

  zoedepth   Intel/zoedepth-nyu-kitti (transformers); image only
  depthpro   apple/DepthPro-hf (transformers); true focal replaces its estimate
  metric3d   Metric3D v2 ViT-L (torch.hub); true K, canonical-camera de-scaling
  unidepth   UniDepthV2 ViT-L (UniDepth repo); true K
  da3metric  Depth Anything 3 DA3METRIC-LARGE; focal * output / 300

--dump-manifest also writes predictions for the images locked by
make_qualitative_md.py; with --only-manifest nothing else is evaluated. The same
flags accept --checkpoint / --released, which go through
eval_multidomain.load_model unchanged, so every column of a figure comes from
one code path.

Run on lingotube with the venv active and, for the comparators,
PYTHONPATH=~/related_models/pkgs.
"""
import argparse
import json
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from datasets.multidomain import EVAL_READERS  # noqa: E402
from eval_multidomain import depth_metrics, load_model  # noqa: E402

RELATED_ROOT = os.path.expanduser(os.environ.get('RELATED_ROOT', '~/related_models'))


class ZoeDepth:
    def __init__(self, device):
        from transformers import AutoImageProcessor, ZoeDepthForDepthEstimation
        name = 'Intel/zoedepth-nyu-kitti'
        self.proc = AutoImageProcessor.from_pretrained(name)
        self.model = ZoeDepthForDepthEstimation.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb, K):
        inp = self.proc(images=rgb, return_tensors='pt').to(self.device)
        out = self.model(**inp)
        post = self.proc.post_process_depth_estimation(out, source_sizes=[rgb.shape[:2]])
        return post[0]['predicted_depth'].float().cpu().numpy()


class DepthPro:
    """Depth = fx_true_at_output / (width * canonical inverse depth).

    Equivalent to the post-processing in transformers, with the estimated
    field of view replaced by the known focal length (Apple's `f_px` option).
    """

    def __init__(self, device):
        from transformers import DepthProForDepthEstimation, DepthProImageProcessorFast
        name = 'apple/DepthPro-hf'
        self.proc = DepthProImageProcessorFast.from_pretrained(name)
        self.model = DepthProForDepthEstimation.from_pretrained(name).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb, K):
        inp = self.proc(images=rgb, return_tensors='pt').to(self.device)
        out = self.model(**inp)
        H, W = rgb.shape[:2]
        canonical = out.predicted_depth  # B x h x w, canonical inverse depth
        inverse = torch.nn.functional.interpolate(
            canonical.unsqueeze(1), size=(H, W), mode='bilinear', align_corners=False)[0, 0]
        # DepthPro: inverse_metric = canonical_inverse * width / f_px
        inverse = inverse * W / float(K[0, 0])
        depth = 1.0 / inverse.clamp(min=1e-4, max=1e4)
        return depth.float().cpu().numpy()


class Metric3D:
    """Pre/post-processing from the official hubconf.py __main__ example."""
    INPUT = (616, 1064)
    MEAN = torch.tensor([123.675, 116.28, 103.53]).view(3, 1, 1)
    STD = torch.tensor([58.395, 57.12, 57.375]).view(3, 1, 1)

    def __init__(self, device):
        self.model = torch.hub.load(os.path.join(RELATED_ROOT, 'Metric3D'), 'metric3d_vit_large',
                                    source='local', pretrain=True).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb, K):
        h, w = rgb.shape[:2]
        scale = min(self.INPUT[0] / h, self.INPUT[1] / w)
        img = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)
        fx = float(K[0, 0]) * scale
        ph, pw = img.shape[:2]
        pad_h, pad_w = self.INPUT[0] - ph, self.INPUT[1] - pw
        top, left = pad_h // 2, pad_w // 2
        img = cv2.copyMakeBorder(img, top, pad_h - top, left, pad_w - left, cv2.BORDER_CONSTANT,
                                 value=[123.675, 116.28, 103.53])
        x = (torch.from_numpy(img.transpose(2, 0, 1)).float() - self.MEAN) / self.STD
        pred, _, _ = self.model.inference({'input': x[None].to(self.device)})
        pred = pred[0, 0, top:top + ph, left:left + pw]
        pred = torch.nn.functional.interpolate(pred[None, None], size=(h, w), mode='bilinear',
                                               align_corners=False)[0, 0]
        return (pred * fx / 1000.0).clamp(0, 300).float().cpu().numpy()


class UniDepth:
    def __init__(self, device):
        sys.path.insert(0, os.path.join(RELATED_ROOT, 'UniDepth'))
        from unidepth.models import UniDepthV2
        self.model = UniDepthV2.from_pretrained('lpiccinelli/unidepth-v2-vitl14').to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, rgb, K):
        x = torch.from_numpy(rgb.transpose(2, 0, 1)).to(self.device)
        out = self.model.infer(x, torch.from_numpy(np.asarray(K, dtype=np.float32)).to(self.device))
        return out['depth'][0, 0].float().cpu().numpy()


class DA3Metric:
    def __init__(self, device):
        sys.path.insert(0, os.path.join(ROOT, 'models', 'raw_models', 'Depth-Anything-3', 'src'))
        from depth_anything_3.api import DepthAnything3
        self.model = DepthAnything3.from_pretrained('depth-anything/DA3METRIC-LARGE').to(device).eval()

    def __call__(self, rgb, K):
        h, w = rgb.shape[:2]
        out = self.model.inference([rgb], process_res=504, process_res_method='upper_bound_resize')
        pred = np.asarray(out.depth[0], dtype=np.float32)
        ph, pw = pred.shape[:2]
        # DA3 metric convention: depth[m] = focal[px, at the processed size] * output / 300
        focal = 0.5 * (K[0, 0] * pw / w + K[1, 1] * ph / h)
        return cv2.resize(pred * float(focal) / 300.0, (w, h), interpolation=cv2.INTER_LINEAR)


ADAPTERS = {'zoedepth': ZoeDepth, 'depthpro': DepthPro, 'metric3d': Metric3D,
            'unidepth': UniDepth, 'da3metric': DA3Metric}


def build_predictor(args, device):
    """Returns (rgb, K) -> depth, and a metadata dict."""
    if args.model:
        return ADAPTERS[args.model](device), {'related': args.model}
    model, uses_k, meta = load_model(args, device)

    def predict(rgb, K):
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            if uses_k:
                return model.infer_image(bgr, args.input_size, intrinsics=K)
            return model.infer_image(bgr, args.input_size)
    return predict, {k: meta[k] for k in ('checkpoint', 'uses_intrinsics') if k in meta}


def center_zoom(rgb, gt, region, K, s):
    """Identical geometry to eval_multidomain.py --center-zoom."""
    H, W = gt.shape
    wh, ww = int(round(H / s)), int(round(W / s))
    y0, x0 = (H - wh) // 2, (W - ww) // 2
    K = K.copy()
    K[0, 2] -= x0
    K[1, 2] -= y0
    return (np.ascontiguousarray(rgb[y0:y0 + wh, x0:x0 + ww]),
            np.ascontiguousarray(gt[y0:y0 + wh, x0:x0 + ww]),
            region[y0:y0 + wh, x0:x0 + ww], K)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument('--model', choices=sorted(ADAPTERS))
    g.add_argument('--checkpoint')
    g.add_argument('--released', choices=['hypersim', 'vkitti'])
    p.add_argument('--encoder', default='vitl')
    p.add_argument('--input-size', type=int, default=518)
    p.add_argument('--datasets', default='nyu,cityscapes,sunrgbd_kv1,sunrgbd_kv2')
    p.add_argument('--out-dir', help='per-image JSON output (not needed with --only-manifest)')
    p.add_argument('--center-zoom', type=float, default=1.0)
    p.add_argument('--dump-manifest', help='make_qualitative_md.py manifest')
    p.add_argument('--dump-dir', help='where <dataset>/<index>[_zoom].npz predictions go')
    p.add_argument('--only-manifest', action='store_true')
    p.add_argument('--max-items', type=int)
    args = p.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    predict, meta = build_predictor(args, device)
    manifest = json.load(open(args.dump_manifest)) if args.dump_manifest else None
    zoomed = args.center_zoom > 1.0
    overview = {}
    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)

    for name in [d.strip() for d in args.datasets.split(',') if d.strip()]:
        reader = EVAL_READERS[name]()
        lo, hi = reader.eval_range
        locked = set(manifest['sets'].get(name, {}).get('indices', [])) if manifest else set()
        indices = sorted(locked) if args.only_manifest else range(len(reader))
        if args.max_items:
            indices = list(indices)[:args.max_items]
        per_image, t0 = [], time.time()
        for i in indices:
            rgb, gt, K = reader.load(i)
            gt = np.asarray(gt, dtype=np.float32)
            K = np.asarray(K, dtype=np.float32)
            region = reader.eval_mask(gt.shape)
            if zoomed:
                rgb, gt, region, K = center_zoom(rgb, gt, region, K, args.center_zoom)
            pred = np.asarray(predict(rgb, K), dtype=np.float32)
            if pred.shape != gt.shape:
                pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            valid = np.isfinite(gt) & (gt > lo) & (gt < hi) & region
            if valid.sum() < 10:
                continue
            m = depth_metrics(pred[valid], gt[valid])
            m['id'] = reader.item_id(i)
            m['pred_median'] = float(np.median(pred[valid]))
            m['gt_median'] = float(np.median(gt[valid]))
            per_image.append(m)
            if i in locked and args.dump_dir:
                d = os.path.join(args.dump_dir, name)
                os.makedirs(d, exist_ok=True)
                suffix = f'_zoom{args.center_zoom:g}' if zoomed else ''
                np.savez_compressed(os.path.join(d, f'{i}{suffix}.npz'),
                                    pred=pred.astype(np.float16), abs_rel=m['abs_rel'],
                                    d1=m['d1'], rmse=m['rmse'], id=m['id'])
        keys = ['abs_rel', 'rmse', 'd1', 'd2', 'd3', 'sq_rel', 'rmse_log', 'log10', 'silog']
        summary = {k: float(np.mean([m[k] for m in per_image])) for k in keys}
        summary['n_images'] = len(per_image)
        summary['median_scale_ratio'] = float(np.median(
            [m['pred_median'] / m['gt_median'] for m in per_image]))
        overview[name] = summary
        if args.out_dir:
            with open(os.path.join(args.out_dir, f'{name}.json'), 'w') as f:
                json.dump({'dataset': name, 'eval_range': [lo, hi], 'summary': summary,
                           'model': meta, 'center_zoom': args.center_zoom,
                           'per_image': per_image}, f, indent=1)
        print(f"{name:14s} n={summary['n_images']:4d} abs_rel={summary['abs_rel']:.4f} "
              f"rmse={summary['rmse']:.3f} d1={summary['d1']:.4f} "
              f"scale={summary['median_scale_ratio']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    if args.out_dir:
        with open(os.path.join(args.out_dir, 'summary.json'), 'w') as f:
            json.dump({'model': meta, 'center_zoom': args.center_zoom, 'results': overview}, f, indent=1)
    print('EVAL_DONE', flush=True)


if __name__ == '__main__':
    main()

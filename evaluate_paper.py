"""
Reproduce the Depth Anything V2 paper evaluation protocol.

Paper targets (Table 2, zero-shot relative depth, released *basic* checkpoints;
AbsRel lower / d1 higher is better):

    encoder   KITTI            NYU-D
    vits      0.078 / 0.936    0.053 / 0.973
    vitb      0.078 / 0.939    0.049 / 0.976
    vitl      0.074 / 0.946    0.045 / 0.979

Notes:
- The paper's *metric* NYU/KITTI table (Table 4) uses in-domain fine-tuned
  checkpoints that were never released, so it cannot be reproduced with the
  public hypersim/vkitti metric checkpoints. Those can still be evaluated
  here with --model-type metric as a zero-shot sanity check.
- Protocol: predictions are made at input_size (518, lower-bound aspect
  preserving resize) and interpolated back to GT resolution (the model
  wrapper does this). For the basic model, per-image scale+shift alignment
  is done in inverse-depth space (affine-invariant protocol). GT is capped
  to [min_depth, max_depth] (KITTI: 0.001-80m, NYU: 0.001-10m) and the
  standard evaluation crop is applied (KITTI: garg crop, NYU: eigen crop).
- KITTI split: the official DA2 split is 652 eigen-val images
  (metric_depth/dataset/splits/kitti/val.txt) with images from KITTI raw
  and GT from data_depth_annotated/val. Pass --kitti-filelist with a local
  copy of val.txt (paths remapped to your KITTI location).

Examples:
    # NYU, basic vitl (paper target 0.045 / 0.979)
    python evaluate_paper.py --dataset nyu --model-type basic --encoder vitl

    # KITTI, basic vitl (paper target 0.074 / 0.946)
    python evaluate_paper.py --dataset kitti --model-type basic --encoder vitl \
        --kitti-filelist datasets/raw_data/kitti/da2_val.txt

    # Sanity check the released outdoor metric model on KITTI
    python evaluate_paper.py --dataset kitti --model-type metric --encoder vitl \
        --kitti-filelist datasets/raw_data/kitti/da2_val.txt
"""

import os
import sys
import json
import argparse
from datetime import datetime

import numpy as np
import cv2
from tqdm import tqdm

from datasets import DatasetConfig, NYUDataset, KITTIDataset
from models import DepthAnythingV2Wrapper
from src.metrics import compute_depth_metrics


# Paper Table 2 (zero-shot relative depth) targets: {dataset: {encoder: (abs_rel, d1)}}
PAPER_TARGETS = {
    'kitti': {'vits': (0.078, 0.936), 'vitb': (0.078, 0.939), 'vitl': (0.074, 0.946), 'vitg': (0.075, 0.948)},
    'nyu':   {'vits': (0.053, 0.973), 'vitb': (0.049, 0.976), 'vitl': (0.045, 0.979), 'vitg': (0.044, 0.979)},
}

# Standard evaluation depth ranges (meters)
EVAL_RANGES = {
    'kitti': (0.001, 80.0),
    'nyu': (0.001, 10.0),
}

# Default evaluation crops
DEFAULT_CROPS = {
    'kitti': 'garg',
    'nyu': 'eigen',
}


def crop_mask(shape, crop: str, dataset: str) -> np.ndarray:
    """Return a boolean mask that is True inside the evaluation crop."""
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if crop == 'none':
        mask[:] = True
    elif crop == 'garg':
        # Garg/eigen KITTI crop (fractions of image size), as in BTS/ZoeDepth
        mask[int(0.40810811 * h):int(0.99189189 * h),
             int(0.03594771 * w):int(0.96405229 * w)] = True
    elif crop == 'eigen':
        if dataset == 'nyu':
            # NYU eigen crop is defined in absolute 480x640 pixels
            if (h, w) == (480, 640):
                mask[45:471, 41:601] = True
            else:
                mask[int(45 / 480 * h):int(471 / 480 * h),
                     int(41 / 640 * w):int(601 / 640 * w)] = True
        else:
            mask[int(0.3324324 * h):int(0.91351351 * h),
                 int(0.03594771 * w):int(0.96405229 * w)] = True
    else:
        raise ValueError(f"Unknown crop: {crop}")
    return mask


def build_dataset(args):
    config_kwargs = dict(
        dataset_path=args.dataset_path,
        max_items=args.max_items,
    )
    if args.dataset == 'nyu':
        config = DatasetConfig(split=args.split or 'test', **config_kwargs)
        return NYUDataset(config)
    else:
        if args.kitti_filelist:
            split = f'filelist:{args.kitti_filelist}'
        elif args.split:
            split = args.split
        else:
            split = 'val_selection'
            print("⚠️  No --kitti-filelist given: falling back to split "
                  "'val_selection' (1000 images). This is NOT the paper's "
                  "652-image eigen-val split; numbers will not be directly "
                  "comparable to the paper.")
        config = DatasetConfig(split=split, **config_kwargs)
        return KITTIDataset(config)


def main():
    parser = argparse.ArgumentParser(description='DA2 paper-protocol evaluation')
    parser.add_argument('--dataset', required=True, choices=['nyu', 'kitti'])
    parser.add_argument('--dataset-path', default=None)
    parser.add_argument('--split', default=None,
                        help="Dataset split. NYU default: 'test' (eigen 654). "
                             "KITTI: use --kitti-filelist instead.")
    parser.add_argument('--kitti-filelist', default=None,
                        help='Local copy of the DA2 kitti val.txt (652 lines: image_path depth_path)')
    parser.add_argument('--model-type', default='basic', choices=['basic', 'metric'],
                        help='basic = relative model with affine alignment (paper Table 2); '
                             'metric = released hypersim/vkitti metric model, no alignment')
    parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--checkpoint', default=None, help='Explicit checkpoint path override')
    parser.add_argument('--input-size', type=int, default=518)
    parser.add_argument('--device', default=None)
    parser.add_argument('--crop', default='auto', choices=['auto', 'none', 'eigen', 'garg'],
                        help="Evaluation crop. auto = garg for KITTI, eigen for NYU")
    parser.add_argument('--clip-pred', default='auto', choices=['auto', 'true', 'false'],
                        help="Clip predictions to the eval depth range after alignment "
                             "(MiDaS/eigen protocol). auto = true for basic, false for metric "
                             "(matches the official DA2 metric eval which does not clip)")
    parser.add_argument('--min-depth-eval', type=float, default=None)
    parser.add_argument('--max-depth-eval', type=float, default=None)
    parser.add_argument('--max-items', type=int, default=None)
    parser.add_argument('--output-json', default=None,
                        help='Where to save results (default: results/paper_eval/<auto>.json)')
    args = parser.parse_args()

    min_depth, max_depth = EVAL_RANGES[args.dataset]
    if args.min_depth_eval is not None:
        min_depth = args.min_depth_eval
    if args.max_depth_eval is not None:
        max_depth = args.max_depth_eval
    crop = DEFAULT_CROPS[args.dataset] if args.crop == 'auto' else args.crop
    if args.clip_pred == 'auto':
        clip_pred = args.model_type == 'basic'
    else:
        clip_pred = args.clip_pred == 'true'

    # Build model
    model_config = {
        'model_type': args.model_type,
        'encoder': args.encoder,
        'checkpoint_path': args.checkpoint,
        'device': args.device,
    }
    if args.model_type == 'metric':
        # Steer checkpoint auto-selection: outdoor (vkitti) for KITTI, indoor (hypersim) for NYU
        model_config['max_depth'] = 80.0 if args.dataset == 'kitti' else 20.0
    model = DepthAnythingV2Wrapper(model_config)
    is_metric = model.is_metric()
    outputs_inverse_depth = model.outputs_inverse_depth()

    if args.model_type == 'basic' and is_metric:
        print("❌ Requested basic model but loaded checkpoint is metric. "
              "Check the checkpoint path/name.")
        sys.exit(1)
    if args.model_type == 'metric' and not is_metric:
        print("❌ Requested metric model but loaded checkpoint is basic. "
              "Check the checkpoint path/name.")
        sys.exit(1)

    dataset = build_dataset(args)
    items = dataset.find_items()
    if len(items) == 0:
        print("❌ No items found — check dataset path/split.")
        sys.exit(1)

    print(f"\n{'=' * 78}")
    print(f"DA2 paper-protocol evaluation")
    print(f"  Model:      {model.get_model_name()} ({'metric' if is_metric else 'basic/relative'})")
    print(f"  Checkpoint: {model.get_checkpoint_path()}")
    print(f"  Dataset:    {args.dataset} ({len(items)} items)")
    print(f"  Depth eval range: [{min_depth}, {max_depth}] m | crop: {crop} | clip_pred: {clip_pred}")
    if not is_metric:
        print(f"  Alignment:  per-image scale+shift in inverse-depth space "
              f"(outputs_inverse_depth={outputs_inverse_depth})")
    print(f"{'=' * 78}\n")

    per_image = []
    skipped = 0
    for item in tqdm(items, desc=f"eval {args.dataset}", unit="img"):
        # Load image
        if dataset.is_cached_dataset():
            image = dataset.load_image(item)
        else:
            image = cv2.imread(item.image_path)
        if image is None:
            skipped += 1
            continue

        gt = dataset.load_gt_depth(item.gt_path, item)
        gt = np.squeeze(np.asarray(gt, dtype=np.float32))

        pred = model.infer_image(image, input_size=args.input_size)
        pred = np.squeeze(np.asarray(pred, dtype=np.float32))
        if pred.shape != gt.shape:
            # Model output should already be at image resolution; resize if not
            pred = cv2.resize(pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Evaluation mask: depth caps + crop (NaN-out GT outside)
        gt_eval = gt.copy()
        gt_eval[~np.isfinite(gt_eval)] = np.nan
        gt_eval[(gt_eval < min_depth) | (gt_eval > max_depth)] = np.nan
        gt_eval[~crop_mask(gt_eval.shape, crop, args.dataset)] = np.nan

        m = compute_depth_metrics(
            pred, gt_eval,
            is_metric_model=is_metric,
            outputs_inverse_depth=outputs_inverse_depth,
            pred_clip=(min_depth, max_depth) if clip_pred else None,
        )
        if m['n_valid'] < 10 or not np.isfinite(m['abs_rel']):
            skipped += 1
            continue
        m['item_id'] = item.item_id
        per_image.append(m)

    if not per_image:
        print("❌ No images produced valid metrics.")
        sys.exit(1)

    metric_keys = ['d1', 'd2', 'd3', 'abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'log10', 'silog']
    summary = {k: float(np.mean([m[k] for m in per_image])) for k in metric_keys}
    summary['n_images'] = len(per_image)
    summary['n_skipped'] = skipped

    print(f"\n{'=' * 78}")
    print(f"RESULTS ({len(per_image)} images, {skipped} skipped)")
    print(f"{'-' * 78}")
    print(', '.join(f"{k}" for k in metric_keys))
    print(', '.join(f"{summary[k]:.4f}" for k in metric_keys))
    print(f"{'-' * 78}")

    target = None
    if not is_metric and crop == DEFAULT_CROPS[args.dataset]:
        target = PAPER_TARGETS[args.dataset].get(model.encoder)
    if target:
        t_absrel, t_d1 = target
        print(f"Paper target ({model.encoder}, {args.dataset}, Table 2): "
              f"AbsRel {t_absrel:.3f} / d1 {t_d1:.3f}")
        print(f"This run:                              "
              f"AbsRel {summary['abs_rel']:.3f} / d1 {summary['d1']:.3f}")
        print(f"Difference:                            "
              f"AbsRel {summary['abs_rel'] - t_absrel:+.3f} / d1 {summary['d1'] - t_d1:+.3f}")
    print(f"{'=' * 78}\n")

    out_path = args.output_json
    if out_path is None:
        os.makedirs('results/paper_eval', exist_ok=True)
        stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        out_path = (f"results/paper_eval/{args.dataset}_{args.model_type}_"
                    f"{model.encoder}_crop-{crop}_{stamp}.json")
    else:
        os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump({
            'summary': summary,
            'paper_target': {'abs_rel': target[0], 'd1': target[1]} if target else None,
            'config': {
                'dataset': args.dataset,
                'split': args.split,
                'kitti_filelist': args.kitti_filelist,
                'model': model.get_model_name(),
                'checkpoint': model.get_checkpoint_path(),
                'model_type': args.model_type,
                'encoder': model.encoder,
                'input_size': args.input_size,
                'min_depth_eval': min_depth,
                'max_depth_eval': max_depth,
                'crop': crop,
                'clip_pred': bool(clip_pred),
                'outputs_inverse_depth': bool(outputs_inverse_depth),
            },
            'per_image': per_image,
        }, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == '__main__':
    main()

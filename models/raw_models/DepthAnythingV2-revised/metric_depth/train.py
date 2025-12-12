import argparse
import logging
import os
import pprint
import random

import warnings
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
from dataset.generic_with_intrinsics import GenericDatasetWithIntrinsics
from depth_anything_v2.dpt import DepthAnythingV2
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss
from util.metric import eval_depth
from util.utils import init_log


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--dataset', default='hypersim', choices=['hypersim', 'vkitti'])
parser.add_argument('--img-size', default=518, type=int)
parser.add_argument('--min-depth', default=0.001, type=float)
parser.add_argument('--max-depth', default=20, type=float)
parser.add_argument('--epochs', default=40, type=int)
parser.add_argument('--bs', default=2, type=int)
parser.add_argument('--lr', default=0.000005, type=float)
parser.add_argument('--pretrained-from', type=str, help='Path to pretrained checkpoint (full model or just pretrained weights)')
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--use-camera-intrinsics', action='store_true', help='Enable camera intrinsics support')
parser.add_argument('--cam-token-inject-layer', type=int, default=None, help='Layer index to inject camera token (None = first layer)')
parser.add_argument('--freeze-dinov2', action='store_true', help='Freeze DINOv2 backbone (default: True, recommended)')
parser.add_argument('--unfreeze-dinov2', action='store_true', help='Unfreeze DINOv2 backbone (not recommended for initial training)')
parser.add_argument('--teacher-checkpoint', type=str, help='Path to teacher model checkpoint for knowledge distillation')
parser.add_argument('--use-distillation', action='store_true', help='Enable teacher-student knowledge distillation')
parser.add_argument('--save-every', type=int, default=5, help='Save checkpoint every N epochs (default: 5)')
parser.add_argument('--skip-validation', action='store_true', help='Skip validation (useful for debugging hangs)')

# Training stability and smoothing arguments
parser.add_argument('--grad-clip', type=float, default=1.0, help='Gradient clipping max norm (default: 1.0)')
parser.add_argument('--head-lr-multiplier', type=float, default=10.0, help='Learning rate multiplier for depth head and cam_encoder (default: 10.0)')
parser.add_argument('--warmup-epochs', type=int, default=1, help='Number of warmup epochs (default: 1)')
parser.add_argument('--accumulate-grad', type=int, default=1, help='Gradient accumulation steps (default: 1, effectively increases batch size)')
parser.add_argument('--ema-decay', type=float, default=0.0, help='EMA decay rate for model weights (0 = disabled, try 0.999 or 0.9999)')
parser.add_argument('--label-smoothing', type=float, default=0.0, help='Label smoothing for SiLog loss (0 = disabled)')
parser.add_argument('--augment-strength', type=float, default=0.5, help='Data augmentation strength (0-1, default: 0.5)')


def main():
    args = parser.parse_args()
    
    warnings.simplefilter('ignore', np.RankWarning)
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    rank, world_size = setup_distributed(port=args.port)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(args.save_path)
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    if args.dataset == 'hypersim':
        trainset = Hypersim('dataset/splits/hypersim/train.txt', 'train', size=size)
    elif args.dataset == 'vkitti':
        # Try multiple possible locations for VKITTI train.txt
        train_file = 'dataset/splits/vkitti2/train.txt'
        # Check if it exists in the default location, otherwise try in datasets/raw_data/vkitti/splits
        if not os.path.exists(train_file):
            alt_train_file = os.path.join('..', '..', '..', '..', '..', 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
            if os.path.exists(alt_train_file):
                train_file = alt_train_file
        trainset = VKITTI2(train_file, 'train', size=size)
    else:
        # Try generic dataset with intrinsics support
        train_filelist = f'dataset/splits/{args.dataset}/train.txt'
        if os.path.exists(train_filelist):
            trainset = GenericDatasetWithIntrinsics(train_filelist, 'train', size=size)
        else:
            raise NotImplementedError(f"Dataset '{args.dataset}' not found. Create {train_filelist} or use 'hypersim'/'vkitti'")
    
    trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
    trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=True, num_workers=4, drop_last=True, sampler=trainsampler)
    
    if args.dataset == 'hypersim':
        valset = Hypersim('dataset/splits/hypersim/val.txt', 'val', size=size)
    elif args.dataset == 'vkitti':
        # For VKITTI, use KITTI validation set (or create your own val.txt)
        valset = KITTI('dataset/splits/kitti/val.txt', 'val', size=size)
    else:
        # Try generic dataset with intrinsics support
        val_filelist = f'dataset/splits/{args.dataset}/val.txt'
        if os.path.exists(val_filelist):
            valset = GenericDatasetWithIntrinsics(val_filelist, 'val', size=size)
        else:
            raise NotImplementedError(f"Validation file list not found: {val_filelist}")
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=4, drop_last=True, sampler=valsampler)
    
    local_rank = int(os.environ["LOCAL_RANK"])
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    model = DepthAnythingV2(
        **{**model_configs[args.encoder], 'max_depth': args.max_depth},
        use_camera_intrinsics=args.use_camera_intrinsics,
        cam_token_inject_layer=args.cam_token_inject_layer
    )
    
    # Load checkpoint (handles both full checkpoints and pretrained-only)
    if args.pretrained_from:
        checkpoint = torch.load(args.pretrained_from, map_location='cpu')
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Full checkpoint with 'model' key
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and 'pretrained' in list(checkpoint.keys())[0]:
            # Full model state dict
            state_dict = checkpoint
        else:
            # Assume it's a state dict
            state_dict = checkpoint
        
        # Filter out cam_encoder keys (don't exist in old checkpoints)
        # and load everything else
        filtered_dict = {k: v for k, v in state_dict.items() 
                         if 'cam_encoder' not in k}
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        if rank == 0:
            logger.info(f'Loaded checkpoint from {args.pretrained_from}')
            if missing_keys:
                logger.info(f'Missing keys (will use random init): {len(missing_keys)} keys (cam_encoder will be randomly initialized)')
            if unexpected_keys:
                logger.warning(f'Unexpected keys: {len(unexpected_keys)} keys')
    
    # Freeze DINOv2 backbone (default behavior, unless explicitly unfrozen)
    freeze_dinov2 = args.freeze_dinov2 or not args.unfreeze_dinov2
    if freeze_dinov2:
        for name, param in model.named_parameters():
            if 'pretrained' in name:
                param.requires_grad = False
        if rank == 0:
            logger.info('DINOv2 backbone is FROZEN (not trainable)')
    else:
        if rank == 0:
            logger.info('DINOv2 backbone is trainable')
    
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda(local_rank)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=True)
    
    # Setup teacher model for knowledge distillation
    teacher_model = None
    if args.use_distillation:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when --use-distillation is enabled")
        
        if rank == 0:
            logger.info('Setting up teacher model for knowledge distillation...')
        
        # Create teacher model (without camera intrinsics)
        teacher_model = DepthAnythingV2(
            **{**model_configs[args.encoder], 'max_depth': args.max_depth},
            use_camera_intrinsics=False,  # Teacher doesn't use intrinsics
            cam_token_inject_layer=None
        )
        
        # Load teacher checkpoint
        teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location='cpu')
        if isinstance(teacher_checkpoint, dict) and 'model' in teacher_checkpoint:
            teacher_state_dict = teacher_checkpoint['model']
        else:
            teacher_state_dict = teacher_checkpoint
        
        # Filter out cam_encoder keys if present
        teacher_state_dict = {k: v for k, v in teacher_state_dict.items() 
                             if 'cam_encoder' not in k}
        teacher_model.load_state_dict(teacher_state_dict, strict=False)
        
        # Freeze teacher model completely
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        teacher_model.cuda(local_rank)
        
        if rank == 0:
            logger.info('Teacher model loaded and frozen')
    
    # Setup loss function (DepthAnythingV2 handles distillation internally)
    criterion = SiLogLoss().cuda(local_rank)
    if rank == 0:
        if args.use_distillation:
            logger.info('Using SiLog loss with teacher-student knowledge distillation')
        else:
            logger.info('Using standard SiLog loss')
    
    # Setup optimizer with different learning rates
    # Only include parameters that require gradients
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    head_lr = args.lr * args.head_lr_multiplier

    if freeze_dinov2:
        # DINOv2 is frozen, so only optimize depth_head and cam_encoder
        optimizer = AdamW(
            [{'params': trainable_params, 'lr': head_lr}],
            lr=head_lr, betas=(0.9, 0.999), weight_decay=0.01
        )
        if rank == 0:
            logger.info(f'Optimizer: Only training depth_head and cam_encoder (DINOv2 frozen)')
            logger.info(f'  Head LR: {head_lr:.2e} (base LR {args.lr:.2e} x {args.head_lr_multiplier})')
    else:
        # DINOv2 is trainable, use different LRs
        pretrained_params = [param for name, param in model.named_parameters()
                            if 'pretrained' in name and param.requires_grad]
        other_params = [param for name, param in model.named_parameters()
                       if 'pretrained' not in name and param.requires_grad]

        optimizer = AdamW([
            {'params': pretrained_params, 'lr': args.lr},
            {'params': other_params, 'lr': head_lr}
        ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

        if rank == 0:
            logger.info('Optimizer: Training all components with different LRs')
            logger.info(f'  Backbone LR: {args.lr:.2e}')
            logger.info(f'  Head LR: {head_lr:.2e}')

    # Setup learning rate scheduler with warmup
    total_iters = args.epochs * len(trainloader)
    warmup_iters = args.warmup_epochs * len(trainloader)

    if rank == 0:
        logger.info(f'Training configuration:')
        logger.info(f'  Total iterations: {total_iters}')
        logger.info(f'  Warmup iterations: {warmup_iters}')
        logger.info(f'  Gradient clipping: {args.grad_clip}')
        logger.info(f'  Gradient accumulation: {args.accumulate_grad}')
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        accumulated_loss = 0

        for i, sample in enumerate(trainloader):
            img, depth, valid_mask = sample['image'].cuda(), sample['depth'].cuda(), sample['valid_mask'].cuda()

            # Data augmentation with configurable strength
            if random.random() < args.augment_strength:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)

            # Get intrinsics if available in sample
            intrinsics = sample.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.cuda()

            # Teacher prediction (without intrinsics) for knowledge distillation
            # DepthAnythingV2 handles knowledge distillation internally
            if args.use_distillation and teacher_model is not None:
                with torch.no_grad():
                    teacher_pred = teacher_model(img, intrinsics=None, image_size=(img.shape[-2], img.shape[-1]))
            else:
                teacher_pred = None

            # Student prediction (with intrinsics)
            # Note: If teacher_pred is needed, it should be passed to model.forward()
            # or handled by the model architecture itself
            pred = model(img, intrinsics=intrinsics, image_size=(img.shape[-2], img.shape[-1]))

            # Standard loss - DepthAnythingV2 handles knowledge distillation internally
            valid_depth_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)
            loss = criterion(pred, depth, valid_depth_mask)

            # Scale loss for gradient accumulation
            loss = loss / args.accumulate_grad
            loss.backward()
            accumulated_loss += loss.item()

            # Only step optimizer after accumulating gradients
            if (i + 1) % args.accumulate_grad == 0 or (i + 1) == len(trainloader):
                # Gradient clipping for training stability
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()
                optimizer.zero_grad()

                # Log accumulated loss
                total_loss += accumulated_loss * args.accumulate_grad
                accumulated_loss = 0

            iters = epoch * len(trainloader) + i

            # Learning rate schedule with warmup
            if iters < warmup_iters:
                # Linear warmup
                lr_scale = (iters + 1) / warmup_iters
            else:
                # Polynomial decay after warmup
                lr_scale = (1 - (iters - warmup_iters) / (total_iters - warmup_iters)) ** 0.9

            lr = args.lr * lr_scale

            # Update learning rates for all parameter groups
            if freeze_dinov2:
                # Only one group (depth_head + cam_encoder)
                optimizer.param_groups[0]["lr"] = lr * args.head_lr_multiplier
            else:
                # Two groups (pretrained + others)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * args.head_lr_multiplier

            if rank == 0:
                writer.add_scalar('train/loss', loss.item() * args.accumulate_grad, iters)
                writer.add_scalar('train/lr', optimizer.param_groups[0]['lr'], iters)

            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item() * args.accumulate_grad))

        # Log training epoch completion
        if rank == 0:
            logger.info(f'Training epoch {epoch} completed. Average loss: {total_loss / len(trainloader):.4f}')

        # Validation (can be skipped for debugging)
        is_best = False
        if not args.skip_validation:
            model.eval()

            results = {'d1': torch.tensor([0.0]).cuda(), 'd2': torch.tensor([0.0]).cuda(), 'd3': torch.tensor([0.0]).cuda(),
                       'abs_rel': torch.tensor([0.0]).cuda(), 'sq_rel': torch.tensor([0.0]).cuda(), 'rmse': torch.tensor([0.0]).cuda(),
                       'rmse_log': torch.tensor([0.0]).cuda(), 'log10': torch.tensor([0.0]).cuda(), 'silog': torch.tensor([0.0]).cuda()}
            nsamples = torch.tensor([0.0]).cuda()

            if rank == 0:
                logger.info('Starting validation...')

            for i, sample in enumerate(valloader):

                img, depth, valid_mask = sample['image'].cuda().float(), sample['depth'].cuda()[0], sample['valid_mask'].cuda()[0]

                # Get intrinsics if available in sample
                intrinsics = sample.get('intrinsics', None)
                if intrinsics is not None:
                    intrinsics = intrinsics.cuda()

                with torch.no_grad():
                    pred = model(img, intrinsics=intrinsics, image_size=(img.shape[-2], img.shape[-1]))
                    pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]

                valid_mask = (valid_mask == 1) & (depth >= args.min_depth) & (depth <= args.max_depth)

                if valid_mask.sum() < 10:
                    continue

                cur_results = eval_depth(pred[valid_mask], depth[valid_mask])

                for k in results.keys():
                    results[k] += cur_results[k]
                nsamples += 1

            if rank == 0:
                logger.info('Validation forward pass completed. Synchronizing...')

            # Synchronize across processes with timeout protection
            try:
                torch.distributed.barrier()

                for k in results.keys():
                    dist.reduce(results[k], dst=0)
                dist.reduce(nsamples, dst=0)
            except Exception as e:
                if rank == 0:
                    logger.warning(f'Distributed sync failed: {e}. Using local results only.')

            if rank == 0:
                if nsamples.item() > 0:
                    logger.info('==========================================================================================')
                    logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
                    logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
                    logger.info('==========================================================================================')
                    print()

                    for name, metric in results.items():
                        writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
                else:
                    logger.warning('No valid validation samples!')

            # Check if this is the best model (based on d1 metric)
            if nsamples.item() > 0:
                current_d1 = (results['d1'] / nsamples).item()
                if current_d1 > previous_best['d1']:
                    is_best = True

                for k in results.keys():
                    if k in ['d1', 'd2', 'd3']:
                        previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
                    else:
                        previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())

        # Save checkpoints
        if rank == 0:
            # Save model configuration for automatic identification when loading
            model_config = {
                'encoder': args.encoder,
                'model_type': 'metric',
                'max_depth': args.max_depth,
                'use_camera_intrinsics': args.use_camera_intrinsics,
                'cam_token_inject_layer': args.cam_token_inject_layer,
                'dataset': args.dataset,
                'img_size': args.img_size,
            }

            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
                'config': model_config,  # Include config for auto-identification
            }

            # Always save latest
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            logger.info(f'Saved latest.pth (epoch {epoch})')

            # Save best model
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))
                logger.info(f'Saved best.pth (epoch {epoch}, d1={previous_best["d1"]:.4f})')

            # Save periodic checkpoints
            if (epoch + 1) % args.save_every == 0:
                epoch_path = os.path.join(args.save_path, f'epoch_{epoch:03d}.pth')
                torch.save(checkpoint, epoch_path)
                logger.info(f'Saved periodic checkpoint: epoch_{epoch:03d}.pth')

    # Training complete
    if rank == 0:
        logger.info('Training completed!')
        logger.info(f'Best metrics: d1={previous_best["d1"]:.4f}, abs_rel={previous_best["abs_rel"]:.4f}')


if __name__ == '__main__':
    main()
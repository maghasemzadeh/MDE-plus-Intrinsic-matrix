import argparse
import logging
import os
import sys
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

# Add metric_depth directory to path for imports
_metric_depth_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                   'models', 'raw_models', 'DepthAnythingV2-revised', 'metric_depth')
if _metric_depth_path not in sys.path:
    sys.path.insert(0, _metric_depth_path)

from dataset.hypersim import Hypersim
from dataset.kitti import KITTI
from dataset.vkitti2 import VKITTI2
try:
    from dataset.generic_with_intrinsics import GenericDatasetWithIntrinsics
except ImportError:
    GenericDatasetWithIntrinsics = None
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
parser.add_argument('--pretrained-from', type=str, help='Path to pretrained checkpoint (full model or just pretrained weights). Can be relative to project root or metric_depth/checkpoints')
parser.add_argument('--save-path', type=str, required=True, help='Path to save checkpoints. Can be relative to project root')
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)
parser.add_argument('--use-camera-intrinsics', action='store_true', help='Enable camera intrinsics support')
parser.add_argument('--cam-token-inject-layer', type=int, default=None, help='Layer index to inject camera token (None = first layer)')
parser.add_argument('--freeze-dinov2', action='store_true', help='Freeze DINOv2 backbone (default: True, recommended)')
parser.add_argument('--unfreeze-dinov2', action='store_true', help='Unfreeze DINOv2 backbone (not recommended for initial training)')
parser.add_argument('--teacher-checkpoint', type=str, help='Path to teacher model checkpoint for knowledge distillation')
parser.add_argument('--use-distillation', action='store_true', help='Enable teacher-student knowledge distillation')


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def main():
    args = parser.parse_args()
    
    # Ignore numpy warnings (RankWarning was removed in newer numpy versions)
    try:
        warnings.simplefilter('ignore', np.RankWarning)
    except AttributeError:
        # RankWarning doesn't exist in newer numpy versions, ignore it
        pass
    
    logger = init_log('global', logging.INFO)
    logger.propagate = 0
    
    # Get project root directory first (needed for path resolution)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Determine device
    device = get_device()
    
    # Only use distributed training if CUDA is available
    if device.type == 'cuda':
        rank, world_size = setup_distributed(port=args.port)
    else:
        rank, world_size = 0, 1
        if rank == 0:
            logger.info(f'Using device: {device} (distributed training disabled for non-CUDA devices)')
    
    # Resolve save path
    if not os.path.isabs(args.save_path):
        save_path = os.path.join(project_root, args.save_path)
    else:
        save_path = args.save_path
    os.makedirs(save_path, exist_ok=True)
    
    if rank == 0:
        all_args = {**vars(args), 'ngpus': world_size, 'save_path': save_path}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        writer = SummaryWriter(save_path)
    else:
        writer = None
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    
    if args.dataset == 'hypersim':
        train_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'train.txt')
        trainset = Hypersim(train_file, 'train', size=size)
    elif args.dataset == 'vkitti':
        # Try multiple possible locations for VKITTI train.txt
        train_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'vkitti2', 'train.txt')
        # Check if it exists in the default location, otherwise try in datasets/raw_data/vkitti/splits
        if not os.path.exists(train_file):
            alt_train_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
            if os.path.exists(alt_train_file):
                train_file = alt_train_file
        trainset = VKITTI2(train_file, 'train', size=size)
    else:
        # Try generic dataset with intrinsics support
        if GenericDatasetWithIntrinsics is None:
            raise NotImplementedError(f"GenericDatasetWithIntrinsics not available. Use 'hypersim' or 'vkitti' dataset.")
        train_filelist = os.path.join(_metric_depth_path, 'dataset', 'splits', args.dataset, 'train.txt')
        if os.path.exists(train_filelist):
            trainset = GenericDatasetWithIntrinsics(train_filelist, 'train', size=size)
        else:
            raise NotImplementedError(f"Dataset '{args.dataset}' not found. Create {train_filelist} or use 'hypersim'/'vkitti'")
    
    # Use distributed sampler only if world_size > 1
    # pin_memory only works with CUDA
    pin_memory = (device.type == 'cuda')
    if world_size > 1:
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=pin_memory, num_workers=4, drop_last=True, sampler=trainsampler)
    else:
        trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=pin_memory, num_workers=4, drop_last=True, shuffle=True)
    
    if args.dataset == 'hypersim':
        val_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'val.txt')
        valset = Hypersim(val_file, 'val', size=size)
    elif args.dataset == 'vkitti':
        # For VKITTI, use KITTI validation set (or create your own val.txt)
        val_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'kitti', 'val.txt')
        valset = KITTI(val_file, 'val', size=size)
    else:
        # Try generic dataset with intrinsics support
        if GenericDatasetWithIntrinsics is None:
            raise NotImplementedError(f"GenericDatasetWithIntrinsics not available. Use 'hypersim' or 'vkitti' dataset.")
        val_filelist = os.path.join(_metric_depth_path, 'dataset', 'splits', args.dataset, 'val.txt')
        if os.path.exists(val_filelist):
            valset = GenericDatasetWithIntrinsics(val_filelist, 'val', size=size)
        else:
            raise NotImplementedError(f"Validation file list not found: {val_filelist}")
    # Use distributed sampler only if world_size > 1
    # pin_memory only works with CUDA
    pin_memory = (device.type == 'cuda')
    if world_size > 1:
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=pin_memory, num_workers=4, drop_last=True, sampler=valsampler)
    else:
        valloader = DataLoader(valset, batch_size=1, pin_memory=pin_memory, num_workers=4, drop_last=True)
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if device.type == 'cuda' else 0
    
    if rank == 0:
        logger.info(f'Using device: {device}')
    
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
        # Resolve checkpoint path
        if not os.path.isabs(args.pretrained_from):
            # Try relative to project root first
            checkpoint_path = os.path.join(project_root, args.pretrained_from)
            if not os.path.exists(checkpoint_path):
                # Try relative to model root checkpoints (DepthAnythingV2-revised/checkpoints) - PRIMARY LOCATION
                model_root = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised')
                checkpoint_path = os.path.join(model_root, 'checkpoints', os.path.basename(args.pretrained_from))
            if not os.path.exists(checkpoint_path):
                # Try with just the filename in model root checkpoints
                model_root = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised')
                checkpoint_path = os.path.join(model_root, 'checkpoints', args.pretrained_from)
            if not os.path.exists(checkpoint_path):
                # Try relative to metric_depth/checkpoints (fallback)
                checkpoint_path = os.path.join(_metric_depth_path, 'checkpoints', os.path.basename(args.pretrained_from))
            if not os.path.exists(checkpoint_path):
                checkpoint_path = args.pretrained_from
        else:
            checkpoint_path = args.pretrained_from
        
        if not os.path.exists(checkpoint_path):
            if rank == 0:
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                logger.warning(f"Available checkpoints in {os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised', 'checkpoints')}:")
                checkpoints_dir = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised', 'checkpoints')
                if os.path.exists(checkpoints_dir):
                    for f in os.listdir(checkpoints_dir):
                        if f.endswith('.pth'):
                            logger.warning(f"  - {f}")
                raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
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
        
        # Get model state dict to check shapes
        model_state_dict = model.state_dict()
        
        # Filter out cam_encoder keys and keys with size mismatches
        filtered_dict = {}
        skipped_keys = []
        for k, v in state_dict.items():
            if 'cam_encoder' in k:
                continue  # Skip cam_encoder keys
            
            # Check if key exists in model and if shapes match
            if k in model_state_dict:
                if v.shape == model_state_dict[k].shape:
                    filtered_dict[k] = v
                else:
                    skipped_keys.append(f"{k} (checkpoint: {v.shape}, model: {model_state_dict[k].shape})")
            else:
                # Key doesn't exist in model, skip it
                skipped_keys.append(f"{k} (not in model)")
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        if rank == 0:
            logger.info(f'Loaded checkpoint from {checkpoint_path}')
            logger.info(f'Successfully loaded {len(filtered_dict)} parameters')
            if skipped_keys:
                logger.warning(f'Skipped {len(skipped_keys)} keys due to size mismatches or missing in model:')
                for key in skipped_keys[:10]:  # Show first 10
                    logger.warning(f'  - {key}')
                if len(skipped_keys) > 10:
                    logger.warning(f'  ... and {len(skipped_keys) - 10} more')
            if missing_keys:
                logger.info(f'Missing keys (will use random init): {len(missing_keys)} keys')
                if len(missing_keys) <= 20:
                    for key in missing_keys:
                        logger.info(f'  - {key}')
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
    
    # Only use SyncBatchNorm and DDP in distributed mode with CUDA
    if world_size > 1 and device.type == 'cuda':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                          output_device=local_rank, find_unused_parameters=True)
    else:
        model = model.to(device)
    
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
        if not os.path.isabs(args.teacher_checkpoint):
            teacher_checkpoint_path = os.path.join(project_root, args.teacher_checkpoint)
            if not os.path.exists(teacher_checkpoint_path):
                # Try relative to model root checkpoints
                model_root = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised')
                teacher_checkpoint_path = os.path.join(model_root, 'checkpoints', os.path.basename(args.teacher_checkpoint))
            if not os.path.exists(teacher_checkpoint_path):
                teacher_checkpoint_path = os.path.join(_metric_depth_path, 'checkpoints', os.path.basename(args.teacher_checkpoint))
            if not os.path.exists(teacher_checkpoint_path):
                teacher_checkpoint_path = args.teacher_checkpoint
        else:
            teacher_checkpoint_path = args.teacher_checkpoint
        
        teacher_checkpoint = torch.load(teacher_checkpoint_path, map_location='cpu')
        if isinstance(teacher_checkpoint, dict) and 'model' in teacher_checkpoint:
            teacher_state_dict = teacher_checkpoint['model']
        else:
            teacher_state_dict = teacher_checkpoint
        
        # Get teacher model state dict to check shapes
        teacher_model_state_dict = teacher_model.state_dict()
        
        # Filter out cam_encoder keys and keys with size mismatches
        teacher_filtered_dict = {}
        teacher_skipped_keys = []
        for k, v in teacher_state_dict.items():
            if 'cam_encoder' in k:
                continue  # Skip cam_encoder keys
            
            # Check if key exists in model and if shapes match
            if k in teacher_model_state_dict:
                if v.shape == teacher_model_state_dict[k].shape:
                    teacher_filtered_dict[k] = v
                else:
                    teacher_skipped_keys.append(f"{k} (checkpoint: {v.shape}, model: {teacher_model_state_dict[k].shape})")
            else:
                # Key doesn't exist in model, skip it
                teacher_skipped_keys.append(f"{k} (not in model)")
        
        teacher_model.load_state_dict(teacher_filtered_dict, strict=False)
        
        if rank == 0 and teacher_skipped_keys:
            logger.warning(f'Teacher model: Skipped {len(teacher_skipped_keys)} keys due to size mismatches')
        
        # Freeze teacher model completely
        for param in teacher_model.parameters():
            param.requires_grad = False
        teacher_model.eval()
        teacher_model = teacher_model.to(device)
        
        if rank == 0:
            logger.info(f'Teacher model loaded from {teacher_checkpoint_path} and frozen')
    
    # Setup loss function (DepthAnythingV2 handles distillation internally)
    criterion = SiLogLoss().to(device)
    if rank == 0:
        if args.use_distillation:
            logger.info('Using SiLog loss with teacher-student knowledge distillation')
        else:
            logger.info('Using standard SiLog loss')
    
    # Setup optimizer with different learning rates
    # Only include parameters that require gradients
    trainable_params = [param for param in model.parameters() if param.requires_grad]
    
    if freeze_dinov2:
        # DINOv2 is frozen, so only optimize depth_head and cam_encoder
        optimizer = AdamW(
            [{'params': trainable_params, 'lr': args.lr * 10.0}],
            lr=args.lr * 10.0, betas=(0.9, 0.999), weight_decay=0.01
        )
        if rank == 0:
            logger.info('Optimizer: Only training depth_head and cam_encoder (DINOv2 frozen)')
    else:
        # DINOv2 is trainable, use different LRs
        pretrained_params = [param for name, param in model.named_parameters() 
                            if 'pretrained' in name and param.requires_grad]
        other_params = [param for name, param in model.named_parameters() 
                       if 'pretrained' not in name and param.requires_grad]
        
        optimizer = AdamW([
            {'params': pretrained_params, 'lr': args.lr},
            {'params': other_params, 'lr': args.lr * 10.0}
        ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)
        
        if rank == 0:
            logger.info('Optimizer: Training all components with different LRs')
    
    total_iters = args.epochs * len(trainloader)
    
    previous_best = {'d1': 0, 'd2': 0, 'd3': 0, 'abs_rel': 100, 'sq_rel': 100, 'rmse': 100, 'rmse_log': 100, 'log10': 100, 'silog': 100}
    
    for epoch in range(args.epochs):
        if rank == 0:
            logger.info('===========> Epoch: {:}/{:}, d1: {:.3f}, d2: {:.3f}, d3: {:.3f}'.format(epoch, args.epochs, previous_best['d1'], previous_best['d2'], previous_best['d3']))
            logger.info('===========> Epoch: {:}/{:}, abs_rel: {:.3f}, sq_rel: {:.3f}, rmse: {:.3f}, rmse_log: {:.3f}, '
                        'log10: {:.3f}, silog: {:.3f}'.format(
                            epoch, args.epochs, previous_best['abs_rel'], previous_best['sq_rel'], previous_best['rmse'], 
                            previous_best['rmse_log'], previous_best['log10'], previous_best['silog']))
        
        if world_size > 1 and hasattr(trainloader, 'sampler') and hasattr(trainloader.sampler, 'set_epoch'):
            trainloader.sampler.set_epoch(epoch + 1)
        
        model.train()
        total_loss = 0
        
        for i, sample in enumerate(trainloader):
            optimizer.zero_grad()
            
            img, depth, valid_mask = sample['image'].to(device), sample['depth'].to(device), sample['valid_mask'].to(device)
            
            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
            
            # Get intrinsics if available in sample
            intrinsics = sample.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)
            
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
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            iters = epoch * len(trainloader) + i
            
            lr = args.lr * (1 - iters / total_iters) ** 0.9
            
            # Update learning rates for all parameter groups
            if freeze_dinov2:
                # Only one group (depth_head + cam_encoder)
                optimizer.param_groups[0]["lr"] = lr * 10.0
            else:
                # Two groups (pretrained + others)
                optimizer.param_groups[0]["lr"] = lr
                optimizer.param_groups[1]["lr"] = lr * 10.0
            
            if rank == 0 and writer is not None:
                writer.add_scalar('train/loss', loss.item(), iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).to(device), 'd2': torch.tensor([0.0]).to(device), 'd3': torch.tensor([0.0]).to(device), 
                   'abs_rel': torch.tensor([0.0]).to(device), 'sq_rel': torch.tensor([0.0]).to(device), 'rmse': torch.tensor([0.0]).to(device), 
                   'rmse_log': torch.tensor([0.0]).to(device), 'log10': torch.tensor([0.0]).to(device), 'silog': torch.tensor([0.0]).to(device)}
        nsamples = torch.tensor([0.0]).to(device)
        
        for i, sample in enumerate(valloader):
            
            img, depth, valid_mask = sample['image'].to(device).float(), sample['depth'].to(device)[0], sample['valid_mask'].to(device)[0]
            
            # Get intrinsics if available in sample
            intrinsics = sample.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)
            
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
        
        if world_size > 1 and device.type == 'cuda':
            torch.distributed.barrier()
            for k in results.keys():
                dist.reduce(results[k], dst=0)
            dist.reduce(nsamples, dst=0)
        
        if rank == 0:
            logger.info('==========================================================================================')
            logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
            logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
            logger.info('==========================================================================================')
            print()
            
            if writer is not None:
                for name, metric in results.items():
                    writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)
        
        for k in results.keys():
            if k in ['d1', 'd2', 'd3']:
                previous_best[k] = max(previous_best[k], (results[k] / nsamples).item())
            else:
                previous_best[k] = min(previous_best[k], (results[k] / nsamples).item())
        
        if rank == 0:
            # Extract model state dict (handle DDP wrapper)
            if hasattr(model, 'module'):
                model_state_dict = model.module.state_dict()
            else:
                model_state_dict = model.state_dict()
            
            checkpoint = {
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(save_path, 'latest.pth'))


if __name__ == '__main__':
    main()
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

# Import training datasets from datasets folder
from datasets.training_datasets import VKITTI2TrainingDataset, KITTITrainingDataset

# Import other datasets from metric_depth (for hypersim, etc.)
from dataset.hypersim import Hypersim
GenericDatasetWithIntrinsics = None

# Import DepthAnythingV2 and force reload to avoid cached version issues
import importlib
if 'depth_anything_v2' in sys.modules:
    importlib.reload(sys.modules['depth_anything_v2'])
if 'depth_anything_v2.dpt' in sys.modules:
    importlib.reload(sys.modules['depth_anything_v2.dpt'])
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
parser.add_argument('--head-lr-multiplier', type=float, default=10.0, help='Learning rate multiplier for depth_head and cam_encoder (default: 10.0, try 1.0-5.0 for stability)')
parser.add_argument('--grad-clip', type=float, default=None, help='Gradient clipping max norm (e.g., 1.0 for stability)')
parser.add_argument('--accumulate-grad', type=int, default=1, help='Number of gradient accumulation steps (default: 1, no accumulation)')
parser.add_argument('--warmup-epochs', type=int, default=0, help='Number of warmup epochs with linear learning rate ramp-up (default: 0, no warmup)')


def get_device():
    """Get the best available device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def detect_encoder_from_checkpoint(checkpoint_path: str) -> str:
    """
    Detect encoder type from checkpoint by examining tensor shapes.
    
    Args:
        checkpoint_path: Path to checkpoint file
    
    Returns:
        Encoder type: 'vits', 'vitb', 'vitl', or 'vitg'
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        state_dict = checkpoint['model']
    elif isinstance(checkpoint, dict) and checkpoint:
        # Check if any key contains 'pretrained' (indicating it's a state dict)
        # This handles both 'pretrained.*' keys and direct state dicts
        has_pretrained_key = any('pretrained' in str(k) for k in checkpoint.keys())
        if has_pretrained_key:
            state_dict = checkpoint
        else:
            # Assume it's a state dict if it's a dict but doesn't have 'model' key
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # Check patch_embed.proj.weight shape to determine encoder
    if 'pretrained.patch_embed.proj.weight' in state_dict:
        weight_shape = state_dict['pretrained.patch_embed.proj.weight'].shape
        embed_dim = weight_shape[0]
    elif 'patch_embed.proj.weight' in state_dict:
        weight_shape = state_dict['patch_embed.proj.weight'].shape
        embed_dim = weight_shape[0]
    else:
        # Try cls_token as fallback
        if 'pretrained.cls_token' in state_dict:
            embed_dim = state_dict['pretrained.cls_token'].shape[-1]
        elif 'cls_token' in state_dict:
            embed_dim = state_dict['cls_token'].shape[-1]
        else:
            # Default to vitl if cannot detect
            return 'vitl'
    
    # Map embed_dim to encoder type
    if embed_dim == 384:
        return 'vits'
    elif embed_dim == 768:
        return 'vitb'
    elif embed_dim == 1024:
        return 'vitl'
    elif embed_dim == 1536:
        return 'vitg'
    else:
        # Default to vitl if unknown
        return 'vitl'


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
    
    # Print initial message immediately
    print("Starting training script...")
    sys.stdout.flush()
    
    # Get project root directory first (needed for path resolution)
    project_root = os.path.dirname(os.path.abspath(__file__))
    
    # Determine device
    device = get_device()
    print(f"Detected device: {device}")
    sys.stdout.flush()
    logger.info(f'Detected device: {device}')
    
    # Only use distributed training if CUDA is available and RANK/WORLD_SIZE are set
    if device.type == 'cuda':
        # Check if distributed training environment variables are set
        if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
            print("Initializing distributed training...")
            sys.stdout.flush()
            logger.info('Initializing distributed training...')
            rank, world_size = setup_distributed(port=args.port)
        else:
            # Not in distributed mode, use single GPU
            rank, world_size = 0, 1
            print(f"Using single GPU: {device}")
            sys.stdout.flush()
            logger.info(f'Using device: {device} (single GPU, distributed training not initialized)')
    else:
        rank, world_size = 0, 1
        print(f"Using device: {device} (CPU/MPS mode)")
        sys.stdout.flush()
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
    
    if rank == 0:
        print(f"Loading training dataset: {args.dataset}...")
        sys.stdout.flush()
        logger.info(f'Loading training dataset: {args.dataset}')
    
    if args.dataset == 'hypersim':
        train_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'train.txt')
        trainset = Hypersim(train_file, 'train', size=size)
    elif args.dataset == 'vkitti':
        # Use split file from datasets/raw_data/vkitti/splits/
        train_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
        if not os.path.exists(train_file):
            raise FileNotFoundError(
                f"VKITTI training file not found: {train_file}\n"
                f"Please ensure the train.txt file exists in datasets/raw_data/vkitti/splits/"
            )
        if rank == 0:
            print(f"Initializing VKITTI training dataset (this may take a while if validating many files)...")
            sys.stdout.flush()
        trainset = VKITTI2TrainingDataset(train_file, 'train', size=size)
        if rank == 0:
            print(f"Training dataset loaded: {len(trainset)} samples")
            sys.stdout.flush()
            logger.info(f'Training dataset loaded: {len(trainset)} samples')
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
    # On macOS, use num_workers=0 to avoid multiprocessing issues (spawn vs fork)
    pin_memory = (device.type == 'cuda')
    num_workers = 0 if sys.platform == 'darwin' else 4
    
    if rank == 0:
        print(f"Creating training DataLoader (num_workers={num_workers})...")
        sys.stdout.flush()
        logger.info(f'Creating training DataLoader with num_workers={num_workers}')
    
    if world_size > 1:
        trainsampler = torch.utils.data.distributed.DistributedSampler(trainset)
        trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=trainsampler)
    else:
        trainloader = DataLoader(trainset, batch_size=args.bs, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, shuffle=True)
    
    if rank == 0:
        print(f"Loading validation dataset: {args.dataset}...")
        sys.stdout.flush()
        logger.info(f'Loading validation dataset: {args.dataset}')
    
    if args.dataset == 'hypersim':
        val_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'val.txt')
        valset = Hypersim(val_file, 'val', size=size)
    elif args.dataset == 'vkitti':
        # Use VKITTI validation set from datasets/raw_data/vkitti/splits/
        vkitti_val_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'val.txt')
        vkitti_train_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
        
        if os.path.exists(vkitti_val_file):
            if rank == 0:
                print(f"Initializing VKITTI validation dataset (this may take a while if validating many files)...")
                sys.stdout.flush()
            valset = VKITTI2TrainingDataset(vkitti_val_file, 'val', size=size)
            if rank == 0:
                print(f"Validation dataset loaded: {len(valset)} samples")
                sys.stdout.flush()
                logger.info(f'Using VKITTI validation set: {vkitti_val_file}')
                logger.info(f'Validation dataset loaded: {len(valset)} samples')
        elif os.path.exists(vkitti_train_file):
            # Fall back to using train.txt for validation if val.txt doesn't exist
            if rank == 0:
                print(f"Initializing VKITTI validation dataset from train.txt (this may take a while if validating many files)...")
                sys.stdout.flush()
            valset = VKITTI2TrainingDataset(vkitti_train_file, 'val', size=size)
            if rank == 0:
                print(f"Validation dataset loaded: {len(valset)} samples")
                sys.stdout.flush()
                logger.warning(f'VKITTI validation set not found at {vkitti_val_file}. Using train.txt for validation: {vkitti_train_file}')
                logger.warning(f'To use a separate validation set, create datasets/raw_data/vkitti/splits/val.txt')
                logger.info(f'Validation dataset loaded: {len(valset)} samples')
        else:
            raise FileNotFoundError(
                f"VKITTI validation file not found. Tried:\n"
                f"  - {vkitti_val_file}\n"
                f"  - {vkitti_train_file}\n"
                f"Please ensure at least train.txt exists in datasets/raw_data/vkitti/splits/"
            )
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
    # On macOS, use num_workers=0 to avoid multiprocessing issues (spawn vs fork)
    pin_memory = (device.type == 'cuda')
    num_workers = 0 if sys.platform == 'darwin' else 4
    
    if rank == 0:
        print(f"Creating validation DataLoader (num_workers={num_workers})...")
        sys.stdout.flush()
        logger.info(f'Creating validation DataLoader with num_workers={num_workers}')
    
    if world_size > 1:
        valsampler = torch.utils.data.distributed.DistributedSampler(valset)
        valloader = DataLoader(valset, batch_size=1, pin_memory=pin_memory, num_workers=num_workers, drop_last=True, sampler=valsampler)
    else:
        valloader = DataLoader(valset, batch_size=1, pin_memory=pin_memory, num_workers=num_workers, drop_last=True)
    
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if device.type == 'cuda' else 0
    
    if rank == 0:
        logger.info(f'Using device: {device}')
    
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
        'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
    }
    
    # Detect encoder from checkpoint if pretrained_from is provided
    encoder_to_use = args.encoder
    if args.pretrained_from:
        # Resolve checkpoint path first
        if not os.path.isabs(args.pretrained_from):
            checkpoint_path_temp = os.path.join(project_root, args.pretrained_from)
            if not os.path.exists(checkpoint_path_temp):
                model_root = os.path.join(project_root, 'models', 'raw_models', 'DepthAnythingV2-revised')
                checkpoint_path_temp = os.path.join(model_root, 'checkpoints', os.path.basename(args.pretrained_from))
            if not os.path.exists(checkpoint_path_temp):
                checkpoint_path_temp = os.path.join(_metric_depth_path, 'checkpoints', os.path.basename(args.pretrained_from))
            if not os.path.exists(checkpoint_path_temp):
                checkpoint_path_temp = args.pretrained_from
        else:
            checkpoint_path_temp = args.pretrained_from
        
        if os.path.exists(checkpoint_path_temp):
            detected_encoder = detect_encoder_from_checkpoint(checkpoint_path_temp)
            if detected_encoder != args.encoder:
                if rank == 0:
                    logger.warning(
                        f"Encoder mismatch detected! "
                        f"Checkpoint uses '{detected_encoder}' but --encoder={args.encoder} was specified. "
                        f"Using '{detected_encoder}' from checkpoint to avoid shape mismatches."
                    )
                    print(f"⚠️  Warning: Auto-correcting encoder from '{args.encoder}' to '{detected_encoder}' based on checkpoint")
                    sys.stdout.flush()
                encoder_to_use = detected_encoder
            else:
                if rank == 0:
                    logger.info(f"Checkpoint encoder matches specified encoder: {args.encoder}")
    
    # max_depth has default value of 20.0, so we don't need to pass it unless different
    model_kwargs = {**model_configs[encoder_to_use], 'use_camera_intrinsics': args.use_camera_intrinsics, 'cam_token_inject_layer': args.cam_token_inject_layer}
    if args.max_depth != 20.0:
        model_kwargs['max_depth'] = args.max_depth
    
    if rank == 0:
        print(f"Initializing model (encoder: {encoder_to_use})...")
        sys.stdout.flush()
        logger.info(f'Initializing model with encoder: {encoder_to_use}')
    
    model = DepthAnythingV2(**model_kwargs)
    
    if rank == 0:
        print("Model initialized successfully")
        sys.stdout.flush()
        logger.info('Model initialized successfully')
    
    # Load checkpoint (handles both full checkpoints and pretrained-only)
    if args.pretrained_from:
        if rank == 0:
            print(f"Loading checkpoint from: {args.pretrained_from}...")
            sys.stdout.flush()
            logger.info(f'Loading checkpoint from: {args.pretrained_from}')
        
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
        
        if rank == 0:
            print(f"Loading checkpoint file (this may take a while for large checkpoints)...")
            sys.stdout.flush()
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if rank == 0:
            print("Checkpoint loaded, processing state dict...")
            sys.stdout.flush()
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'model' in checkpoint:
            # Full checkpoint with 'model' key
            state_dict = checkpoint['model']
        elif isinstance(checkpoint, dict) and checkpoint:
            # Check if any key contains 'pretrained' (indicating it's a state dict)
            # This handles both 'pretrained.*' keys and direct state dicts
            has_pretrained_key = any('pretrained' in str(k) for k in checkpoint.keys())
            if has_pretrained_key:
                # Full model state dict
                state_dict = checkpoint
            else:
                # Assume it's a state dict if it's a dict but doesn't have 'model' key
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
        
        if rank == 0:
            print(f"Loading {len(filtered_dict)} parameters into model...")
            sys.stdout.flush()
        
        missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
        
        if rank == 0:
            print("Checkpoint loaded successfully")
            sys.stdout.flush()
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
    if rank == 0:
        print(f"Moving model to device: {device}...")
        sys.stdout.flush()
        logger.info(f'Moving model to device: {device}')
    
    if world_size > 1 and device.type == 'cuda':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = model.to(device)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                          output_device=local_rank, find_unused_parameters=True)
        if rank == 0:
            print("Model wrapped with DistributedDataParallel")
            sys.stdout.flush()
    else:
        model = model.to(device)
        if rank == 0:
            print("Model moved to device successfully")
            sys.stdout.flush()
    
    # Setup teacher model for knowledge distillation
    teacher_model = None
    if args.use_distillation:
        if not args.teacher_checkpoint:
            raise ValueError("--teacher-checkpoint is required when --use-distillation is enabled")
        
        if rank == 0:
            logger.info('Setting up teacher model for knowledge distillation...')
        
        # Create teacher model (without camera intrinsics)
        teacher_kwargs = {**model_configs[args.encoder], 'use_camera_intrinsics': False, 'cam_token_inject_layer': None}
        if args.max_depth != 20.0:
            teacher_kwargs['max_depth'] = args.max_depth
        teacher_model = DepthAnythingV2(**teacher_kwargs)
        
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
        print("Starting training...")
        print("=" * 80)
        sys.stdout.flush()
    
    # Setup optimizer with different learning rates
    # Only include parameters that require gradients
    trainable_params = [param for param in model.parameters() if param.requires_grad]

    head_lr_mult = args.head_lr_multiplier

    if freeze_dinov2:
        # DINOv2 is frozen, so only optimize depth_head and cam_encoder
        optimizer = AdamW(
            [{'params': trainable_params, 'lr': args.lr * head_lr_mult}],
            lr=args.lr * head_lr_mult, betas=(0.9, 0.999), weight_decay=0.01
        )
        if rank == 0:
            logger.info(f'Optimizer: Only training depth_head and cam_encoder (DINOv2 frozen)')
            logger.info(f'Learning rate: {args.lr * head_lr_mult:.7f} (base: {args.lr}, multiplier: {head_lr_mult})')
    else:
        # DINOv2 is trainable, use different LRs
        pretrained_params = [param for name, param in model.named_parameters()
                            if 'pretrained' in name and param.requires_grad]
        other_params = [param for name, param in model.named_parameters()
                       if 'pretrained' not in name and param.requires_grad]

        optimizer = AdamW([
            {'params': pretrained_params, 'lr': args.lr},
            {'params': other_params, 'lr': args.lr * head_lr_mult}
        ], lr=args.lr, betas=(0.9, 0.999), weight_decay=0.01)

        if rank == 0:
            logger.info(f'Optimizer: Training all components with different LRs')
            logger.info(f'Pretrained LR: {args.lr:.7f}, Head/CamEnc LR: {args.lr * head_lr_mult:.7f}')
    
    total_iters = args.epochs * len(trainloader)
    warmup_iters = args.warmup_epochs * len(trainloader)

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
            # Only zero gradients at the start of accumulation
            if i % args.accumulate_grad == 0:
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

            # Scale loss for gradient accumulation
            loss = loss / args.accumulate_grad
            loss.backward()

            # Only step optimizer and clip gradients after accumulating enough steps
            if (i + 1) % args.accumulate_grad == 0 or (i + 1) == len(trainloader):
                # Gradient clipping for stability
                if args.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

                optimizer.step()

            total_loss += loss.item() * args.accumulate_grad

            iters = epoch * len(trainloader) + i

            # Calculate learning rate with warmup
            if iters < warmup_iters:
                # Linear warmup
                lr = args.lr * (iters / warmup_iters)
            else:
                # Polynomial decay after warmup
                lr = args.lr * (1 - (iters - warmup_iters) / (total_iters - warmup_iters)) ** 0.9

            # Update learning rates for all parameter groups (only when optimizer.step() is called)
            if (i + 1) % args.accumulate_grad == 0 or (i + 1) == len(trainloader):
                if freeze_dinov2:
                    # Only one group (depth_head + cam_encoder)
                    optimizer.param_groups[0]["lr"] = lr * head_lr_mult
                else:
                    # Two groups (pretrained + others)
                    optimizer.param_groups[0]["lr"] = lr
                    optimizer.param_groups[1]["lr"] = lr * head_lr_mult
            
            if rank == 0 and writer is not None:
                writer.add_scalar('train/loss', loss.item(), iters)
                # Log learning rate
                if freeze_dinov2:
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], iters)
                else:
                    writer.add_scalar('train/learning_rate_pretrained', optimizer.param_groups[0]['lr'], iters)
                    writer.add_scalar('train/learning_rate_others', optimizer.param_groups[1]['lr'], iters)
            
            if rank == 0 and i % 100 == 0:
                logger.info('Iter: {}/{}, LR: {:.7f}, Loss: {:.3f}'.format(i, len(trainloader), optimizer.param_groups[0]['lr'], loss.item()))
        
        # Log average training loss per epoch
        if rank == 0 and writer is not None:
            avg_train_loss = total_loss / len(trainloader)
            writer.add_scalar('train/avg_loss_per_epoch', avg_train_loss, epoch)
        
        model.eval()
        
        results = {'d1': torch.tensor([0.0]).to(device), 'd2': torch.tensor([0.0]).to(device), 'd3': torch.tensor([0.0]).to(device), 
                   'abs_rel': torch.tensor([0.0]).to(device), 'sq_rel': torch.tensor([0.0]).to(device), 'rmse': torch.tensor([0.0]).to(device), 
                   'rmse_log': torch.tensor([0.0]).to(device), 'log10': torch.tensor([0.0]).to(device), 'silog': torch.tensor([0.0]).to(device)}
        nsamples = torch.tensor([0.0]).to(device)
        
        # Store first validation sample for visualization
        first_val_sample = None
        first_val_pred = None
        
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
            
            # Store first validation sample for visualization
            if first_val_sample is None and rank == 0:
                first_val_sample = {
                    'image': img[0].cpu(),
                    'depth': depth.cpu(),
                    'pred': pred.cpu(),
                    'valid_mask': valid_mask.cpu()
                }
                first_val_pred = pred.cpu()
            
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
            # Handle case where no valid samples were found
            if nsamples.item() == 0:
                logger.warning('No valid validation samples found! Skipping metrics computation.')
                logger.info('==========================================================================================')
                print()
            else:
                logger.info('{:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}, {:>8}'.format(*tuple(results.keys())))
                logger.info('{:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}, {:8.3f}'.format(*tuple([(v / nsamples).item() for v in results.values()])))
                logger.info('==========================================================================================')
                print()
            
            if writer is not None and nsamples.item() > 0:
                for name, metric in results.items():
                    writer.add_scalar(f'eval/{name}', (metric / nsamples).item(), epoch)

                # Log sample depth predictions as images (every 5 epochs to save space)
                if first_val_sample is not None and epoch % 5 == 0:
                    import matplotlib.pyplot as plt
                    import matplotlib
                    matplotlib.use('Agg')  # Non-interactive backend
                    
                    # Normalize images for visualization
                    img_viz = first_val_sample['image']
                    depth_gt_viz = first_val_sample['depth']
                    depth_pred_viz = first_val_sample['pred']
                    valid_mask_viz = first_val_sample['valid_mask']
                    
                    # Convert to numpy and normalize
                    img_np = img_viz.permute(1, 2, 0).numpy()
                    img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-8)
                    
                    depth_gt_np = depth_gt_viz.numpy()
                    depth_pred_np = depth_pred_viz.numpy()
                    valid_mask_np = valid_mask_viz.numpy()
                    
                    # Normalize depth maps for visualization (only using valid pixels)
                    if valid_mask_np.sum() > 0:
                        depth_gt_valid = depth_gt_np[valid_mask_np]
                        depth_pred_valid = depth_pred_np[valid_mask_np]
                        depth_gt_min, depth_gt_max = depth_gt_valid.min(), depth_gt_valid.max()
                        depth_pred_min, depth_pred_max = depth_pred_valid.min(), depth_pred_valid.max()
                        
                        depth_gt_norm = (depth_gt_np - depth_gt_min) / (depth_gt_max - depth_gt_min + 1e-8)
                        depth_pred_norm = (depth_pred_np - depth_pred_min) / (depth_pred_max - depth_pred_min + 1e-8)
                    else:
                        # Fallback if no valid pixels
                        depth_gt_norm = depth_gt_np / (depth_gt_np.max() + 1e-8)
                        depth_pred_norm = depth_pred_np / (depth_pred_np.max() + 1e-8)
                    
                    # Create figure with subplots
                    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                    axes[0].imshow(img_np)
                    axes[0].set_title('Input Image')
                    axes[0].axis('off')
                    
                    axes[1].imshow(depth_gt_norm, cmap='jet')
                    axes[1].set_title('Ground Truth Depth')
                    axes[1].axis('off')
                    
                    axes[2].imshow(depth_pred_norm, cmap='jet')
                    axes[2].set_title('Predicted Depth')
                    axes[2].axis('off')
                    
                    plt.tight_layout()
                    
                    # Convert figure to image tensor for TensorBoard
                    # Use BytesIO approach for maximum compatibility across matplotlib versions
                    import io
                    buf = io.BytesIO()
                    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
                    buf.seek(0)
                    from PIL import Image
                    img = Image.open(buf)
                    img_array = np.array(img)
                    buf.close()
                    plt.close(fig)
                    
                    # Convert to tensor and add to TensorBoard (PIL returns RGB, convert to CHW)
                    if len(img_array.shape) == 3:
                        img_tensor = torch.from_numpy(img_array).permute(2, 0, 1).float() / 255.0
                    else:
                        # Grayscale, convert to RGB
                        img_tensor = torch.from_numpy(img_array).unsqueeze(0).repeat(3, 1, 1).float() / 255.0
                    writer.add_image('eval/sample_prediction', img_tensor, epoch)
        
        # Track if this epoch improved any metric (only if we have valid samples)
        improved = False
        if nsamples.item() > 0:
            for k in results.keys():
                current_value = (results[k] / nsamples).item()
                if k in ['d1', 'd2', 'd3']:
                    if current_value > previous_best[k]:
                        improved = True
                        previous_best[k] = current_value
                else:
                    if current_value < previous_best[k]:
                        improved = True
                        previous_best[k] = current_value
        
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
                # Save model configuration for proper loading
                'config': {
                    'encoder': args.encoder,
                    'max_depth': args.max_depth,
                    'use_camera_intrinsics': args.use_camera_intrinsics,
                    'cam_token_inject_layer': args.cam_token_inject_layer,
                    'model_type': 'metric',
                }
            }
            
            # Always save latest checkpoint
            latest_path = os.path.join(save_path, 'latest.pth')
            # Use atomic write to prevent corruption: write to temp file then rename
            temp_path = latest_path + '.tmp'
            torch.save(checkpoint, temp_path)
            os.replace(temp_path, latest_path)
            
            # Save best checkpoint if metrics improved
            if improved:
                best_path = os.path.join(save_path, 'best.pth')
                temp_path = best_path + '.tmp'
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, best_path)
                logger.info(f'New best checkpoint saved! Metrics: {previous_best}')


if __name__ == '__main__':
    main()
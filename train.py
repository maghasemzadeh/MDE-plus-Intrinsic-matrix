import argparse
import logging
import os
import sys
import pprint
import random
from datetime import datetime

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
from datasets.training_datasets import VKITTI2TrainingDataset, KITTITrainingDataset, NYUTrainingDataset, DIODETrainingDataset, RobustDataset

# Import other datasets from metric_depth (for hypersim, etc.)
from dataset.hypersim import Hypersim
GenericDatasetWithIntrinsics = None

# Import metric model from metric_depth path
from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Metric
from util.dist_helper import setup_distributed
from util.loss import SiLogLoss, ScaleShiftInvariantLoss
from util.metric import eval_depth, eval_depth_scale_aligned
from util.utils import init_log

# Path to basic (relative depth) model in the revised DepthAnythingV2
_basic_model_parent_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                         'models', 'raw_models', 'DepthAnythingV2-revised')


def get_basic_depth_anything_v2():
    """Import and return the basic DepthAnythingV2 model class (relative depth, with intrinsics support)."""
    # Add the parent path to sys.path so we can import depth_anything_v2 as a package
    if _basic_model_parent_path not in sys.path:
        sys.path.insert(0, _basic_model_parent_path)

    # Import the module as a proper package to support relative imports
    from depth_anything_v2.dpt import DepthAnythingV2 as DepthAnythingV2Basic
    return DepthAnythingV2Basic


parser = argparse.ArgumentParser(description='Depth Anything V2 for Metric/Basic Depth Estimation')

parser.add_argument('--encoder', default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
parser.add_argument('--datasets', default='hypersim', 
                    help='Comma-separated list of datasets to train on (e.g., "vkitti,diode"). '
                         'Supported: hypersim, vkitti, nyu, diode')
parser.add_argument('--model-type', default='basic', choices=['metric', 'basic'],
                    help='Model type: metric (absolute depth in meters) or basic (relative depth with scale ambiguity)')
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
parser.add_argument('--model-variant', default='revised', choices=['base', 'revised'],
                    help='Model variant: base (original DA2, no intrinsics support) or revised (DA2 with camera intrinsics support). '
                         'Use base for a fair baseline comparison without intrinsics.')
parser.add_argument('--use-camera-intrinsics', action='store_true', help='Enable camera intrinsics support (only valid with --model-variant revised)')
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


def generate_run_name(args) -> str:
    """
    Generate a unique, descriptive run name for TensorBoard based on training parameters.
    This name will appear in TensorBoard as the run identifier.
    
    Args:
        args: Parsed command-line arguments
    
    Returns:
        A string with format: modeltype_encoder_dataset[_intrinsics][_distill]_lr{lr}_bs{bs}_timestamp
    """
    components = []

    # Model variant (base vs revised)
    components.append(args.model_variant)

    # Model type
    components.append(args.model_type)
    
    # Encoder
    components.append(args.encoder)
    
    # Dataset(s) - join with '+' if multiple
    dataset_names = [d.strip().lower() for d in args.datasets.split(',')]
    components.append('+'.join(dataset_names))
    
    # Camera intrinsics flag
    if args.use_camera_intrinsics:
        components.append('intrinsics')
        if args.cam_token_inject_layer is not None:
            components.append(f'layer{args.cam_token_inject_layer}')
    
    # Knowledge distillation flag
    if args.use_distillation:
        components.append('distill')
    
    # Learning rate (important hyperparameter)
    lr_str = f"{args.lr:.0e}".replace('e-0', 'e-').replace('e+0', 'e').replace('.0', '')
    components.append(f'lr{lr_str}')
    
    # Batch size (important hyperparameter)
    components.append(f'bs{args.bs}')
    
    # Head LR multiplier if different from default
    if args.head_lr_multiplier != 10.0:
        components.append(f'headlr{args.head_lr_multiplier}')
    
    # Add timestamp for uniqueness and to know when run was executed
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    components.append(timestamp)
    
    return '_'.join(components)


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

    # Validate model-variant / intrinsics combination
    if args.model_variant == 'base' and args.use_camera_intrinsics:
        raise ValueError(
            "--use-camera-intrinsics is not compatible with --model-variant base. "
            "The base model does not have a camera encoder. "
            "Use --model-variant revised to enable camera intrinsics support."
        )
    # Ensure base variant never receives intrinsics (safety guard)
    if args.model_variant == 'base':
        args.use_camera_intrinsics = False
        args.cam_token_inject_layer = None

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
    
    # Resolve base save path (run name will be generated after encoder detection)
    if not os.path.isabs(args.save_path):
        base_save_path = os.path.join(project_root, args.save_path)
    else:
        base_save_path = args.save_path
    os.makedirs(base_save_path, exist_ok=True)
    
    # TensorBoard writer will be created after encoder detection and run name generation
    writer = None
    
    cudnn.enabled = True
    cudnn.benchmark = True
    
    size = (args.img_size, args.img_size)
    
    # Parse datasets (comma-separated list)
    dataset_names = [d.strip().lower() for d in args.datasets.split(',')]
    
    if rank == 0:
        print(f"Loading training datasets: {dataset_names}...")
        sys.stdout.flush()
        logger.info(f'Loading training datasets: {dataset_names}')
    
    def create_train_dataset(dataset_name):
        """Create a training dataset by name."""
        if dataset_name == 'hypersim':
            train_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'train.txt')
            return Hypersim(train_file, 'train', size=size)
        elif dataset_name == 'vkitti':
            train_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
            if not os.path.exists(train_file):
                raise FileNotFoundError(
                    f"VKITTI training file not found: {train_file}\n"
                    f"Please ensure the train.txt file exists in datasets/raw_data/vkitti/splits/"
                )
            if rank == 0:
                print(f"Initializing VKITTI training dataset...")
                sys.stdout.flush()
            dataset = VKITTI2TrainingDataset(train_file, 'train', size=size)
            if rank == 0:
                print(f"VKITTI training dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        elif dataset_name == 'nyu':
            nyu_mat_file = os.path.join(project_root, 'datasets', 'raw_data', 'nyu', 'nyu_depth_v2_labeled.mat')
            if not os.path.exists(nyu_mat_file):
                raise FileNotFoundError(
                    f"NYU Depth V2 .mat file not found: {nyu_mat_file}\n"
                    f"Please download the labeled subset from:\n"
                    f"https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html\n"
                    f"and place nyu_depth_v2_labeled.mat in: {os.path.dirname(nyu_mat_file)}"
                )
            if rank == 0:
                print(f"Initializing NYU Depth V2 training dataset...")
                sys.stdout.flush()
            dataset = NYUTrainingDataset(nyu_mat_file, 'train', size=size)
            if rank == 0:
                print(f"NYU training dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        elif dataset_name == 'diode':
            diode_path = os.path.join(project_root, 'datasets', 'raw_data', 'diode')
            if not os.path.exists(diode_path):
                raise FileNotFoundError(
                    f"DIODE dataset not found: {diode_path}\n"
                    f"Please ensure the DIODE dataset is placed in: {diode_path}"
                )
            if rank == 0:
                print(f"Initializing DIODE training dataset...")
                sys.stdout.flush()
            dataset = DIODETrainingDataset(diode_path, 'train', size=size)
            if rank == 0:
                print(f"DIODE training dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        elif dataset_name == 'kitti':
            kitti_path = os.path.join(project_root, 'datasets', 'raw_data', 'kitti')
            if not os.path.exists(kitti_path):
                raise FileNotFoundError(
                    f"KITTI dataset not found: {kitti_path}\n"
                    f"Please ensure the KITTI dataset is placed in: {kitti_path}"
                )
            if rank == 0:
                print(f"Initializing KITTI training dataset...")
                sys.stdout.flush()
            dataset = KITTITrainingDataset(kitti_path, 'train', size=size)
            if rank == 0:
                print(f"KITTI training dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        else:
            # Try generic dataset with intrinsics support
            if GenericDatasetWithIntrinsics is None:
                raise NotImplementedError(
                    f"Unknown dataset: '{dataset_name}'. "
                    f"Supported datasets: hypersim, vkitti, nyu, diode, kitti"
                )
            train_filelist = os.path.join(_metric_depth_path, 'dataset', 'splits', dataset_name, 'train.txt')
            if os.path.exists(train_filelist):
                return GenericDatasetWithIntrinsics(train_filelist, 'train', size=size)
            else:
                raise NotImplementedError(
                    f"Dataset '{dataset_name}' not found. "
                    f"Supported datasets: hypersim, vkitti, nyu, diode, kitti"
                )
    
    # Create training datasets
    train_datasets = []
    for dataset_name in dataset_names:
        train_datasets.append(create_train_dataset(dataset_name))
    
    # Combine datasets if multiple
    if len(train_datasets) == 1:
        trainset = train_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        trainset = ConcatDataset(train_datasets)
        if rank == 0:
            print(f"Combined training dataset: {len(trainset)} samples from {len(train_datasets)} datasets")
            sys.stdout.flush()
            logger.info(f'Combined training dataset: {len(trainset)} samples from {len(train_datasets)} datasets')
    
    # Wrap with RobustDataset to skip corrupted samples (e.g., libpng CRC errors)
    trainset = RobustDataset(trainset)

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
        print(f"Loading validation datasets: {dataset_names}...")
        sys.stdout.flush()
        logger.info(f'Loading validation datasets: {dataset_names}')
    
    def create_val_dataset(dataset_name):
        """Create a validation dataset by name."""
        if dataset_name == 'hypersim':
            val_file = os.path.join(_metric_depth_path, 'dataset', 'splits', 'hypersim', 'val.txt')
            return Hypersim(val_file, 'val', size=size)
        elif dataset_name == 'vkitti':
            vkitti_val_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'val.txt')
            vkitti_train_file = os.path.join(project_root, 'datasets', 'raw_data', 'vkitti', 'splits', 'train.txt')
            
            if os.path.exists(vkitti_val_file):
                if rank == 0:
                    print(f"Initializing VKITTI validation dataset...")
                    sys.stdout.flush()
                dataset = VKITTI2TrainingDataset(vkitti_val_file, 'val', size=size)
                if rank == 0:
                    print(f"VKITTI validation dataset loaded: {len(dataset)} samples")
                    sys.stdout.flush()
                return dataset
            elif os.path.exists(vkitti_train_file):
                if rank == 0:
                    print(f"Initializing VKITTI validation dataset from train.txt...")
                    sys.stdout.flush()
                    logger.warning(f'VKITTI val.txt not found, using train.txt for validation')
                dataset = VKITTI2TrainingDataset(vkitti_train_file, 'val', size=size)
                if rank == 0:
                    print(f"VKITTI validation dataset loaded: {len(dataset)} samples")
                    sys.stdout.flush()
                return dataset
            else:
                raise FileNotFoundError(
                    f"VKITTI validation file not found. Tried:\n"
                    f"  - {vkitti_val_file}\n"
                    f"  - {vkitti_train_file}"
                )
        elif dataset_name == 'nyu':
            nyu_mat_file = os.path.join(project_root, 'datasets', 'raw_data', 'nyu', 'nyu_depth_v2_labeled.mat')
            if rank == 0:
                print(f"Initializing NYU Depth V2 validation dataset...")
                sys.stdout.flush()
            dataset = NYUTrainingDataset(nyu_mat_file, 'val', size=size)
            if rank == 0:
                print(f"NYU validation dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        elif dataset_name == 'diode':
            diode_path = os.path.join(project_root, 'datasets', 'raw_data', 'diode')
            if rank == 0:
                print(f"Initializing DIODE validation dataset...")
                sys.stdout.flush()
            dataset = DIODETrainingDataset(diode_path, 'val', size=size)
            if rank == 0:
                print(f"DIODE validation dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        elif dataset_name == 'kitti':
            kitti_path = os.path.join(project_root, 'datasets', 'raw_data', 'kitti')
            if rank == 0:
                print(f"Initializing KITTI validation dataset...")
                sys.stdout.flush()
            dataset = KITTITrainingDataset(kitti_path, 'val', size=size)
            if rank == 0:
                print(f"KITTI validation dataset loaded: {len(dataset)} samples")
                sys.stdout.flush()
            return dataset
        else:
            # Try generic dataset with intrinsics support
            if GenericDatasetWithIntrinsics is None:
                raise NotImplementedError(
                    f"Unknown dataset: '{dataset_name}'. "
                    f"Supported datasets: hypersim, vkitti, nyu, diode, kitti"
                )
            val_filelist = os.path.join(_metric_depth_path, 'dataset', 'splits', dataset_name, 'val.txt')
            if os.path.exists(val_filelist):
                return GenericDatasetWithIntrinsics(val_filelist, 'val', size=size)
            else:
                raise NotImplementedError(
                    f"Dataset '{dataset_name}' not found. "
                    f"Supported datasets: hypersim, vkitti, nyu, diode, kitti"
                )
    
    # Create validation datasets
    val_datasets = []
    for dataset_name in dataset_names:
        val_datasets.append(create_val_dataset(dataset_name))
    
    # Combine datasets if multiple
    if len(val_datasets) == 1:
        valset = val_datasets[0]
    else:
        from torch.utils.data import ConcatDataset
        valset = ConcatDataset(val_datasets)
        if rank == 0:
            print(f"Combined validation dataset: {len(valset)} samples from {len(val_datasets)} datasets")
            sys.stdout.flush()
            logger.info(f'Combined validation dataset: {len(valset)} samples from {len(val_datasets)} datasets')
    # Wrap with RobustDataset to skip corrupted samples
    valset = RobustDataset(valset)
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
    
    # Generate unique run name with the correct encoder (after potential auto-correction)
    # Temporarily update args.encoder for run name generation
    original_encoder = args.encoder
    args.encoder = encoder_to_use
    run_name = generate_run_name(args)
    args.encoder = original_encoder  # Restore original for consistency
    
    # Create run-specific directory
    save_path = os.path.join(base_save_path, run_name)
    os.makedirs(save_path, exist_ok=True)
    
    # Prepare hyperparameters dictionary (accessible throughout the function for final logging)
    hparams = {
        'model_variant': args.model_variant,
        'model_type': args.model_type,
        'encoder': encoder_to_use,
        'datasets': args.datasets,
        'use_camera_intrinsics': args.use_camera_intrinsics,
        'cam_token_inject_layer': args.cam_token_inject_layer if args.use_camera_intrinsics else None,
        'use_distillation': args.use_distillation,
        'lr': args.lr,
        'batch_size': args.bs,
        'epochs': args.epochs,
        'img_size': args.img_size,
        'min_depth': args.min_depth,
        'max_depth': args.max_depth,
        'head_lr_multiplier': args.head_lr_multiplier,
        'grad_clip': args.grad_clip,
        'accumulate_grad': args.accumulate_grad,
        'warmup_epochs': args.warmup_epochs,
        'freeze_dinov2': args.freeze_dinov2 or not args.unfreeze_dinov2,
    }
    # Remove None values
    hparams = {k: v for k, v in hparams.items() if v is not None}
    
    if rank == 0:
        print(f"Training run name: {run_name}")
        print(f"Checkpoints and logs will be saved to: {save_path}")
        print(f"To view all runs in TensorBoard, run: tensorboard --logdir {base_save_path}")
        sys.stdout.flush()
        all_args = {**vars(args), 'ngpus': world_size, 'save_path': save_path, 'run_name': run_name, 'encoder_used': encoder_to_use}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        logger.info(f'Training run name: {run_name}')
        logger.info(f'Using encoder: {encoder_to_use}')
        logger.info(f'To view all runs in TensorBoard, run: tensorboard --logdir {base_save_path}')
        
        # Initialize TensorBoard writer with run-specific directory
        # TensorBoard will use the directory name as the run name
        writer = SummaryWriter(save_path)
        
        # Log hyperparameters early (will be updated with final metrics at end of training)
        # TensorBoard uses this to show hyperparameters for each run
        writer.add_hparams(hparams, {})
        
        # Add a text summary with run description for easy identification
        run_description = f"""
Training Run Summary:
- Run Name: {run_name}
- Model Variant: {args.model_variant}
- Model Type: {args.model_type}
- Encoder: {encoder_to_use}
- Datasets: {args.datasets}
- Camera Intrinsics: {args.use_camera_intrinsics}
- Learning Rate: {args.lr}
- Batch Size: {args.bs}
- Epochs: {args.epochs}
- Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        writer.add_text('run_info/description', run_description, 0)
    else:
        writer = None
    
    # Determine if we're training metric or basic model
    is_metric = args.model_type == 'metric'

    if rank == 0:
        print(f"Model type: {args.model_type} ({'absolute depth in meters' if is_metric else 'relative depth with scale ambiguity'})")
        sys.stdout.flush()
        logger.info(f'Model type: {args.model_type}')

    # Configure model kwargs based on model type
    # Both metric and basic models in the revised version support camera intrinsics
    model_kwargs = {
        **model_configs[encoder_to_use],
        'use_camera_intrinsics': args.use_camera_intrinsics,
        'cam_token_inject_layer': args.cam_token_inject_layer
    }

    if is_metric:
        # Metric model: add max_depth for scaling output to meters
        if args.max_depth != 20.0:
            model_kwargs['max_depth'] = args.max_depth

    if rank == 0:
        print(f"Initializing model (encoder: {encoder_to_use}, type: {args.model_type})...")
        sys.stdout.flush()
        logger.info(f'Initializing model with encoder: {encoder_to_use}, type: {args.model_type}')

    # Create model based on type
    if is_metric:
        model = DepthAnythingV2Metric(**model_kwargs)
    else:
        DepthAnythingV2Basic = get_basic_depth_anything_v2()
        model = DepthAnythingV2Basic(**model_kwargs)
    
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
        
        # Create teacher model (without camera intrinsics, same type as student)
        teacher_kwargs = {**model_configs[args.encoder], 'use_camera_intrinsics': False, 'cam_token_inject_layer': None}
        if is_metric:
            if args.max_depth != 20.0:
                teacher_kwargs['max_depth'] = args.max_depth
            teacher_model = DepthAnythingV2Metric(**teacher_kwargs)
        else:
            TeacherBasicClass = get_basic_depth_anything_v2()
            teacher_model = TeacherBasicClass(**teacher_kwargs)
        
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
    
    # Setup loss function based on model type
    if is_metric:
        # Metric model: use SiLog loss (requires absolute depth values)
        criterion = SiLogLoss().to(device)
        if rank == 0:
            if args.use_distillation:
                logger.info('Using SiLog loss with teacher-student knowledge distillation')
            else:
                logger.info('Using standard SiLog loss (metric model)')
    else:
        # Basic model: use scale-shift invariant loss (handles relative depth)
        criterion = ScaleShiftInvariantLoss(alpha=0.5).to(device)
        if rank == 0:
            logger.info('Using Scale-Shift Invariant loss (basic model)')

    if rank == 0:
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

            # Bug 1+2 fix: load intrinsics and original size BEFORE the flip so we can
            # update cx correctly, and pass the original (pre-resize) image dimensions to
            # the camera encoder so its normalisation (fx/W, cx/W …) matches inference.
            intrinsics = sample.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)

            original_size = sample.get('original_size', None)
            if original_size is not None:
                original_size = original_size.to(device)  # (B, 2)  [H, W]

            if random.random() < 0.5:
                img = img.flip(-1)
                depth = depth.flip(-1)
                valid_mask = valid_mask.flip(-1)
                # Bug 2 fix: horizontal flip mirrors the principal point.
                # cx_new = original_width - cx  (intrinsics are still at original resolution)
                if intrinsics is not None and original_size is not None:
                    intrinsics = intrinsics.clone()
                    original_w = original_size[:, 1].float()  # (B,)
                    intrinsics[:, 0, 2] = original_w - intrinsics[:, 0, 2]

            # Bug 1 fix: use the original image dimensions (matching the intrinsics'
            # resolution) so the camera encoder's normalisation is consistent with
            # inference.  All images in a batch from the same dataset share the same
            # original resolution, so using the first item's size is correct.
            if original_size is not None:
                image_size_for_cam = (original_size[0, 0].item(), original_size[0, 1].item())
            else:
                image_size_for_cam = (img.shape[-2], img.shape[-1])

            # Teacher prediction (without intrinsics) for knowledge distillation
            # DepthAnythingV2 handles knowledge distillation internally
            if args.use_distillation and teacher_model is not None:
                with torch.no_grad():
                    teacher_pred = teacher_model(img, intrinsics=None, image_size=image_size_for_cam)
            else:
                teacher_pred = None

            # Student prediction (with intrinsics)
            # Note: If teacher_pred is needed, it should be passed to model.forward()
            # or handled by the model architecture itself
            pred = model(img, intrinsics=intrinsics, image_size=image_size_for_cam)

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
            
            # Bug 1 fix: use original image size for the camera encoder's normalisation
            intrinsics = sample.get('intrinsics', None)
            if intrinsics is not None:
                intrinsics = intrinsics.to(device)
            
            original_size = sample.get('original_size', None)
            if original_size is not None:
                original_size = original_size.to(device)
                image_size_for_cam = (original_size[0, 0].item(), original_size[0, 1].item())
            else:
                image_size_for_cam = (img.shape[-2], img.shape[-1])
            
            with torch.no_grad():
                pred = model(img, intrinsics=intrinsics, image_size=image_size_for_cam)
                # Handle both 3D (B, H, W) and 4D (B, 1, H, W) model outputs
                if pred.dim() == 4:
                    # Already has channel dimension, just interpolate and take first sample
                    pred = F.interpolate(pred, depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
                else:
                    # 3D output (B, H, W), add channel dim for interpolation
                    pred = F.interpolate(pred[:, None], depth.shape[-2:], mode='bilinear', align_corners=True)[0, 0]
            
            # Ensure valid_mask is boolean and matches depth constraints
            valid_mask = valid_mask.bool() & (depth >= args.min_depth) & (depth <= args.max_depth)
            
            # Verify shapes match before indexing
            if pred.shape != valid_mask.shape:
                if rank == 0 and i == 0:
                    logger.warning(f'Shape mismatch: pred={pred.shape}, valid_mask={valid_mask.shape}, depth={depth.shape}')
                continue
            
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
            
            # Use appropriate evaluation function based on model type
            if is_metric:
                cur_results = eval_depth(pred[valid_mask], depth[valid_mask])
            else:
                # Basic model: use scale-aligned evaluation
                cur_results = eval_depth_scale_aligned(pred[valid_mask], depth[valid_mask])

            for k in results.keys():
                if k in cur_results:
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
        else:
            # No valid validation samples: treat first epoch as improved so we always save at least one checkpoint
            improved = (epoch == 0)
        
        if rank == 0:
            # Ensure save directory exists (in case it was removed)
            os.makedirs(save_path, exist_ok=True)
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
                    'model_variant': args.model_variant,
                    'encoder': args.encoder,
                    'max_depth': args.max_depth,
                    'use_camera_intrinsics': args.use_camera_intrinsics,
                    'cam_token_inject_layer': args.cam_token_inject_layer,
                    'model_type': args.model_type,
                }
            }
            
            # Always save latest checkpoint
            latest_path = os.path.join(save_path, 'latest.pth')
            try:
                temp_path = latest_path + '.tmp'
                torch.save(checkpoint, temp_path)
                os.replace(temp_path, latest_path)
                logger.info(f'Checkpoint saved: {latest_path}')
            except Exception as e:
                logger.error(f'Failed to save latest.pth to {latest_path}: {e}')
                raise
            
            # Save best checkpoint if metrics improved
            if improved:
                best_path = os.path.join(save_path, 'best.pth')
                try:
                    temp_path = best_path + '.tmp'
                    torch.save(checkpoint, temp_path)
                    os.replace(temp_path, best_path)
                    logger.info(f'Best checkpoint saved: {best_path}')
                    logger.info(f'New best checkpoint saved! Metrics: {previous_best}')
                except Exception as e:
                    logger.error(f'Failed to save best.pth to {best_path}: {e}')
    
    # After training completes, log final hyperparameters with best metrics
    if rank == 0 and writer is not None:
        # Prepare final metrics dictionary with best values achieved during training
        final_metrics = {}
        for metric_name, metric_value in previous_best.items():
            # For accuracy metrics (higher is better), use as is
            # For error metrics (lower is better), negate for TensorBoard visualization
            if metric_name in ['d1', 'd2', 'd3']:
                final_metrics[f'hparam/{metric_name}'] = metric_value
            else:
                # Negate error metrics so they appear as "higher is better" in hparam view
                final_metrics[f'hparam/{metric_name}'] = -metric_value
        
        # Log hyperparameters with final metrics for proper TensorBoard visualization
        # This allows filtering and comparing runs by hyperparameters and their results
        try:
            writer.add_hparams(hparams, final_metrics)
            logger.info('Final hyperparameters and metrics logged to TensorBoard')
        except Exception as e:
            logger.warning(f'Could not log final hyperparameters: {e}')
        
        # Close the writer to ensure all data is flushed
        writer.close()
        logger.info('TensorBoard writer closed. Training completed.')


if __name__ == '__main__':
    main()
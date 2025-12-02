import os
import subprocess

import torch
import torch.distributed as dist


def setup_distributed(backend="nccl", port=None):
    """AdaHessian Optimizer
    Lifted from https://github.com/BIGBALLON/distribuuuu/blob/master/distribuuuu/utils.py
    Originally licensed MIT, Copyright (c) 2020 Wei Li
    """
    num_gpus = torch.cuda.device_count()

    # Check if running in distributed mode
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        # Distributed training mode
        if "SLURM_JOB_ID" in os.environ:
            rank = int(os.environ["SLURM_PROCID"])
            world_size = int(os.environ["SLURM_NTASKS"])
            node_list = os.environ["SLURM_NODELIST"]
            addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
            # specify master port
            if port is not None:
                os.environ["MASTER_PORT"] = str(port)
            elif "MASTER_PORT" not in os.environ:
                os.environ["MASTER_PORT"] = "10685"
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = addr
            os.environ["WORLD_SIZE"] = str(world_size)
            os.environ["LOCAL_RANK"] = str(rank % num_gpus)
            os.environ["RANK"] = str(rank)
        else:
            rank = int(os.environ["RANK"])
            world_size = int(os.environ["WORLD_SIZE"])

        torch.cuda.set_device(rank % num_gpus)

        dist.init_process_group(
            backend=backend,
            world_size=world_size,
            rank=rank,
        )
    else:
        # Single GPU mode - no distributed setup needed
        rank = 0
        world_size = 1
        if "LOCAL_RANK" not in os.environ:
            os.environ["LOCAL_RANK"] = "0"
        if num_gpus > 0:
            torch.cuda.set_device(0)
    
    return rank, world_size

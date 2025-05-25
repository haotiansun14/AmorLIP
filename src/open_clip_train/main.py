import glob
import logging
import os
import time
import re
import subprocess
import sys
import random
from datetime import datetime
from functools import partial

import json

import numpy as np
import torch
from torch import optim

import warnings
warnings.filterwarnings("ignore", message="Length of IterableDataset")

try:
    import wandb
except ImportError:
    wandb = None

try:
    import torch.utils.tensorboard as tensorboard
except ImportError:
    tensorboard = None

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

import copy
import functools
import socket
from collections import defaultdict

from open_clip import create_model_and_transforms, trace_model, get_tokenizer
from open_clip_train.data import get_data
from open_clip_train.distributed import is_master, init_distributed_device, broadcast_object
from open_clip_train.logger import setup_logging
from open_clip_train.params import parse_args
from open_clip_train.train import evaluate
from open_clip_train.file_utils import pt_load, start_sync_process, remote_sync

from open_clip_train.train import AverageMeter
from open_clip.utils import EMA, get_ema_model, get_cosine_ema_beta
from open_clip.loss import ConditionalExpLoss, LambdaLoss, LogTargetCollector
from open_clip_train.params import parse_args
from open_clip_train.train import optimize_step, log_func, evaluate, RestartCosineScheduler
from open_clip.factory import create_model_and_transforms
from open_clip.model import LambdaMLPs

from datacomp.evaluate import evaluate as evaluate_datacomp
from datacomp.evaluate import convert_to_csv


# torch.autograd.set_detect_anomaly(True)
LATEST_CHECKPOINT_NAME = "epoch_latest.pt"


def random_seed(seed=42, rank=0):
    torch.manual_seed(seed + rank)
    np.random.seed(seed + rank)
    random.seed(seed + rank)



def natural_key(string_):
    """See http://www.codinghorror.com/blog/archives/001018.html"""
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', string_.lower())]


def get_latest_checkpoint(path: str, remote : bool):
    # as writen, this glob recurses, so can pick up checkpoints across multiple sub-folders
    if remote:
        result = subprocess.run(["aws", "s3", "ls", path + "/"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        print(result)
        if result.returncode == 1:
            return None
        checkpoints = [os.path.join(path, x.split(' ')[-1]) for x in result.stdout.decode().split('\n')[:-1]]
    else:
        checkpoints = glob.glob(path + '**/*.pt', recursive=True)
    if checkpoints:
        checkpoints = sorted(checkpoints, key=natural_key)
        return checkpoints[-1]
    return None


def reinitialize_lambda_mlps(lambda_dict, args, embed_dim, device):
    """
    Reinitialize the lambda_mlps and its optimizer during training.

    Args:
        lambda_dict (dict): Dictionary containing the lambda-related components.
        args (argparse.Namespace): Arguments specifying model configurations.
        embed_dim (int): Embedding dimension for the model.
        device (torch.device): Device to place the model on.

    Returns:
        Updated lambda_dict with reinitialized lambda_mlps and optimizer.
    """
    # ---------------------------------------------
    # Step 1: Reinitialize the model on the correct device
    # ---------------------------------------------
    new_lambda_mlps = LambdaMLPs(
        embed_dim, 
        h_dim_factor=args.h_dim_factor,
        device=device, 
        output_dict=True
    )
    new_lambda_mlps.to(device)

    # ---------------------------------------------
    # Step 2: Handle distributed training
    # ---------------------------------------------
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            new_lambda_mlps = torch.nn.SyncBatchNorm.convert_sync_batchnorm(new_lambda_mlps)

        # Prepare DDP arguments
        ddp_args = {}
        if args.ddp_static_graph:
            ddp_args['static_graph'] = True

        # If using GPUs, pass device_ids as a list of int
        if device.type == 'cuda':
            new_lambda_mlps = torch.nn.parallel.DistributedDataParallel(
                new_lambda_mlps,
                device_ids=[device.index],         # Must be an int, e.g., 0
                output_device=device.index,
                **ddp_args
            )
        else:
            # CPU or other device
            new_lambda_mlps = torch.nn.parallel.DistributedDataParallel(new_lambda_mlps, **ddp_args)

    # ---------------------------------------------
    # Step 3: Update the lambda dictionary with the new model
    # ---------------------------------------------
    lambda_dict["lambda_mlps"] = new_lambda_mlps

    # ---------------------------------------------
    # Step 4: Reinitialize the optimizer
    # ---------------------------------------------
    # Make sure to use new model parameters here
    lambda_dict["lambda_optimizer"] = torch.optim.Adam(
        new_lambda_mlps.parameters(),
        lr=args.lambda_lr * args.world_size
    )

    # If you have an EMA model, reinitialize that as well
    lambda_dict["lambda_ema_mlps"] = get_ema_model(lambda_dict["lambda_mlps"])

    return lambda_dict



def main(args):
    args = parse_args(args)

    if torch.cuda.is_available():
        # This enables tf32 on Ampere GPUs which is only 8% slower than
        # float16 and almost as accurate as float32
        # This was a default in pytorch until 1.12
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False

    # fully initialize distributed device environment
    device = init_distributed_device(args)

    # get the name of the experiments
    if args.name is None:
        # sanitize model name for filesystem / uri use, easier if we don't use / in name as a rule?
        model_name_safe = args.model.replace('/', '-')
        date_str = datetime.now().strftime("%m%d%H%M%S")
        if args.distributed:
            # sync date_str from master to all ranks
            date_str = broadcast_object(args, date_str)
        args.name = "".join([
        '-'.join([
            str(args.note),
            f"coef_{args.pos_coef}",
            f"scale{args.scale_loss}",
            f"fit_r_{args.lambda_loss_type}",
            ]),
        '-'.join([
            f"{model_name_safe}",
            f"{args.batch_size}",
            ]),
        f'_{date_str}',
        ])

    resume_latest = args.resume == 'latest'
    log_base_path = os.path.join(args.logs, args.name)
    args.log_path = None
    if is_master(args, local=args.log_local):
        os.makedirs(log_base_path, exist_ok=True)
        log_filename = f'out-{args.rank}' if args.log_local else 'out.log'
        args.log_path = os.path.join(log_base_path, log_filename)
        if os.path.exists(args.log_path) and not resume_latest:
            print(
                "Error. Experiment already exists. Use --name {} to specify a new experiment."
            )
            return -1

    # Setup text logger
    args.log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(args.log_path, args.log_level)

    # Setup wandb, tensorboard, checkpoint logging
    args.wandb = 'wandb' in args.report_to or 'all' in args.report_to
    args.tensorboard = 'tensorboard' in args.report_to or 'all' in args.report_to
    args.checkpoint_path = os.path.join(log_base_path, "checkpoints")
    if is_master(args):
        args.tensorboard_path = os.path.join(log_base_path, "tensorboard") if args.tensorboard else ''
        for dirname in [args.tensorboard_path, args.checkpoint_path]:
            if dirname:
                os.makedirs(dirname, exist_ok=True)
    else:
        args.tensorboard_path = ''

    if resume_latest:
        resume_from = None
        checkpoint_path = args.checkpoint_path
        # If using remote_sync, need to check the remote instead of the local checkpoints folder.
        if args.remote_sync is not None:
            checkpoint_path = os.path.join(args.remote_sync, args.name, "checkpoints")
            if args.save_most_recent:
                print('Error. Cannot use save-most-recent with remote_sync and resume latest.')
                return -1
            if args.remote_sync_protocol != 's3':
                print('Error. Sync protocol not supported when using resume latest.')
                return -1
        if is_master(args):
            # Checking for existing checkpoint via master rank only. It is possible for
            # different rank processes to see different files if a shared file-system is under
            # stress, however it's very difficult to fully work around such situations.
            if args.save_most_recent:
                # if --save-most-recent flag is set, look for latest at a fixed filename
                resume_from = os.path.join(checkpoint_path, LATEST_CHECKPOINT_NAME)
                if not os.path.exists(resume_from):
                    # If no latest checkpoint has been saved yet, don't try to resume
                    resume_from = None
            else:
                # otherwise, list checkpoint dir contents and pick the newest checkpoint
                resume_from = get_latest_checkpoint(checkpoint_path, remote=args.remote_sync is not None)
            if resume_from:
                logging.info(f'Found latest resume checkpoint at {resume_from}.')
            else:
                logging.info(f'No latest resume checkpoint found in {checkpoint_path}.')
        if args.distributed:
            # sync found checkpoint path to all ranks
            resume_from = broadcast_object(args, resume_from)
        args.resume = resume_from

    if args.copy_codebase:
        copy_codebase(args)

    # start the sync proces if remote-sync is not None
    remote_sync_process = None
    if is_master(args) and args.remote_sync is not None:
        # first make sure it works
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('remote sync successful.')
        else:
            logging.info('Error: remote sync failed. Exiting.')
            return -1
        # if all looks good, start a process to do this every args.remote_sync_frequency seconds
        remote_sync_process = start_sync_process(
            args.remote_sync_frequency,
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        remote_sync_process.start()

    if args.precision == 'fp16':
        logging.warning(
            'It is recommended to use AMP mixed-precision instead of FP16. '
            'FP16 support needs further verification and tuning, especially for train.')

    if args.horovod:
        logging.info(
            f'Running in horovod mode with multiple processes / nodes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    elif args.distributed:
        logging.info(
            f'Running in distributed mode with multiple processes. Device: {args.device}.'
            f'Process (global: {args.rank}, local {args.local_rank}), total {args.world_size}.')
    else:
        logging.info(f'Running with a single process. Device {args.device}.')

    dist_model = None
    args.distill = args.distill_model is not None and args.distill_pretrained is not None
    if args.distill:
        #FIXME: support distillation with grad accum.
        assert args.accum_freq == 1
        #FIXME: support distillation with coca.
        assert 'coca' not in args.model.lower()

    if isinstance(args.force_image_size, (tuple, list)) and len(args.force_image_size) == 1:
        # arg is nargs, single (square) image size list -> int
        args.force_image_size = args.force_image_size[0]
    random_seed(args.seed, 0)
    init_logit_scale = args.init_logit_scale
    if init_logit_scale is None:
        logging.info("Initializing logit scale based on loss type.")
        init_logit_scale = 1 / 0.07
            
    init_logit_scale = np.log(init_logit_scale)
    
    model_kwargs = {
        "init_logit_scale": init_logit_scale,
        "normalize_type": args.normalize_type,
        "norm_cap": args.norm_cap,
        "learn_logit_scale": args.learn_logit_scale,
        "learn_logit_bias": args.learn_logit_bias,
    }

    if args.init_logit_bias:
        model_kwargs["init_logit_bias"] = args.init_logit_bias
    model, preprocess_train, preprocess_val = create_model_and_transforms(
        args.model,
        args.pretrained,
        precision=args.precision,
        device=device,
        jit=args.torchscript,
        force_quick_gelu=args.force_quick_gelu,
        force_custom_text=args.force_custom_text,
        force_patch_dropout=args.force_patch_dropout,
        force_image_size=args.force_image_size,
        image_mean=args.image_mean,
        image_std=args.image_std,
        image_interpolation=args.image_interpolation,
        image_resize_mode=args.image_resize_mode,  # only effective for inference
        aug_cfg=args.aug_cfg,
        pretrained_image=args.pretrained_image,
        output_dict=True,
        cache_dir=args.cache_dir,
        **model_kwargs,
    )
    if args.distill:
        # FIXME: currently assumes the model you're distilling from has the same tokenizer & transforms.
        dist_model, _, _ = create_model_and_transforms(
            args.distill_model, 
            args.distill_pretrained,
            device=device,
            precision=args.precision,
            output_dict=True,
            cache_dir=args.cache_dir,
        )
    if args.use_bnb_linear is not None:
        print('=> using a layer from bitsandbytes.\n'
              '   this is an experimental feature which requires two extra pip installs\n'
              '   pip install bitsandbytes triton'
              '   please make sure to use triton 2.0.0')
        import bitsandbytes as bnb
        from open_clip.utils import replace_linear
        print(f'=> replacing linear layers with {args.use_bnb_linear}')
        linear_replacement_cls = getattr(bnb.nn.triton_based_modules, args.use_bnb_linear)
        replace_linear(model, linear_replacement_cls)
        model = model.to(device)

    random_seed(args.seed, args.rank)

    if args.trace:
        model = trace_model(model, batch_size=args.batch_size, device=device)

    if args.lock_image:
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        model.lock_image_tower(
            unlocked_groups=args.lock_image_unlocked_groups,
            freeze_bn_stats=args.lock_image_freeze_bn_stats)
    if args.lock_text:
        model.lock_text_tower(
            unlocked_layers=args.lock_text_unlocked_layers,
            freeze_layer_norm=args.lock_text_freeze_layer_norm)

    if args.grad_checkpointing:
        model.set_grad_checkpointing()

    if is_master(args):
        logging.info("Model:")
        logging.info(f"{str(model)}")
        logging.info("Params:")
        params_file = os.path.join(args.logs, args.name, "params.txt")
        with open(params_file, "w") as f:
            for name in sorted(vars(args)):
                val = getattr(args, name)
                logging.info(f"  {name}: {val}")
                f.write(f"{name}: {val}\n")

    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        # * for two step updates
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[device], **ddp_args)
    
        if args.distill:
            dist_model = torch.nn.parallel.DistributedDataParallel(dist_model, device_ids=[device], **ddp_args)

    # create optimizer and scaler
    model_optimizer = None
    scaler = None

    if args.train_data or args.dataset_type == "synthetic":
        assert not args.trace, 'Cannot train with traced model'        
        is_logit_scale = lambda n, p: "logit_scale" in n or "logit_bias" in n
        exclude = lambda n, p: (p.ndim < 2 and "logit_scale" not in n and "logit_bias" not in n) or "bn" in n or "ln" in n or "bias" in n
        include = lambda n, p: not exclude(n, p) and not is_logit_scale(n, p)

        lr_tau = args.lr_tau if args.lr_tau > 0 else args.lr

        logging.info(f"lr: {args.lr}, lr_tau: {args.lr_tau}")

        # Define filtering functions
        is_logit_scale = lambda n, p: "logit_scale" in n or "logit_bias" in n
        exclude = lambda n, p: (
            (p.ndim < 2 and not is_logit_scale(n, p)) or  # Exclude low-dim params unless they are logit_scale/logit_bias
            "bn" in n or "ln" in n or ("bias" in n and not is_logit_scale(n, p))  # Exclude biases except logit_bias
        )
        include = lambda n, p: not exclude(n, p) and not is_logit_scale(n, p)

        # Learning rate for logit parameters
        lr_tau = args.lr_tau if args.lr_tau > 0 else args.lr
        logging.info(f"lr: {args.lr}, lr_tau: {args.lr_tau}")

        # Group parameters
        named_parameters = list(model.named_parameters())
        gain_or_bias_params, gain_or_bias_params_names = [], []
        rest_params, rest_params_names = [], []
        logit_scale_params, logit_scale_params_names = [], []

        for n, p in named_parameters:
            if is_logit_scale(n, p) and p.requires_grad:
                logit_scale_params.append(p)
                logit_scale_params_names.append(n)
            elif exclude(n, p) and p.requires_grad:
                gain_or_bias_params.append(p)
                gain_or_bias_params_names.append(n)
            elif include(n, p) and p.requires_grad:
                rest_params.append(p)
                rest_params_names.append(n)


        model_optimizer = optim.AdamW(
                [
                    {"params": gain_or_bias_params, "weight_decay": 0.},
                    {"params": rest_params, "weight_decay": args.wd},
                    {"params": logit_scale_params, "weight_decay": 0., "lr": lr_tau},
                ],
                lr=args.lr,
                betas=(args.beta1, args.beta2),
                eps=args.eps,
            )

        if args.horovod:
            model_optimizer = hvd.DistributedOptimizer(model_optimizer, named_parameters=model.named_parameters())
            hvd.broadcast_parameters(model.state_dict(), root_rank=0)
            hvd.broadcast_optimizer_state(model_optimizer, root_rank=0)

        scaler = None
        if args.precision == "amp":
            try:
                scaler = torch.amp.GradScaler(device=device)
            except (AttributeError, TypeError) as e:
                scaler = torch.cuda.amp.GradScaler()

    # * create lambda_tensor(s) for lambda update
    json_file = f"open_clip/model_configs/{args.model}.json"
    with open(json_file, "r") as f:
        model_config = json.load(f)
    embed_dim = model_config["embed_dim"]

    # embed_dim = 1024 if args.model.lower() == "rn50" else 512 # model_config["embed_dim"]
    if args.rank == 0:
        print(f"Feature dimension: {embed_dim}, effective batch size: {args.batch_size * args.world_size}")
    
    lambda_dict = {}

    # * create lambda MLPs for both modalities
    lambda_mlps = LambdaMLPs(embed_dim, 
                                h_dim_factor=args.h_dim_factor,
                                device=device,
                                output_dict=True)
    if args.distributed and not args.horovod:
        if args.use_bn_sync:
            lambda_mlps = torch.nn.SyncBatchNorm.convert_sync_batchnorm(lambda_mlps)
        # * for two step updates
        ddp_args = {}
        if args.ddp_static_graph:
            # this doesn't exist in older PyTorch, arg only added if enabled
            ddp_args['static_graph'] = True
        lambda_mlps = torch.nn.parallel.DistributedDataParallel(lambda_mlps, device_ids=[device], **ddp_args)
    lambda_dict["lambda_mlps"] = lambda_mlps
    
    # * create lambda loss 
    lambda_dict["lambda_loss"] = LambdaLoss(
            rank=args.rank,
            world_size=args.world_size,
            lambda_loss_type=args.lambda_loss_type,
            pos_coef=args.pos_coef,
            lambda_fit_neg_only=args.lambda_fit_neg_only,
            div_reg_coef=args.div_reg_coef,
    )
    # * create lambda loss optimizer
    lambda_dict["lambda_optimizer"] = torch.optim.Adam(
        lambda_dict["lambda_mlps"].parameters(), lr=args.lambda_lr
    )
    #  * create lambda MLP ema updater and ema model
    lambda_dict["lambda_ema_updater"] =  EMA(
            beta_init=args.model_beta_init,
        )
    lambda_dict["lambda_ema_mlps"] = get_ema_model(lambda_dict["lambda_mlps"])
    lambda_dict["lambda_scaler"] = torch.amp.GradScaler(device=device) if args.precision == "amp" else None
    
    if args.fit_w_prev:
        lambda_dict["ema_model_beta_func"] = lambda _: args.model_ema_beta
        lambda_dict["ema_z_beta_func"] = partial(
            get_cosine_ema_beta, beta_decay_epochs=args.beta_decay_epochs, beta_max=args.z_beta_max,
        )
    else:
        lambda_dict["ema_model_beta_func"] = partial(
            get_cosine_ema_beta, beta_decay_epochs=args.beta_decay_epochs, beta_max=args.z_beta_max,
        )
        lambda_dict["ema_z_beta_func"] = lambda _: 1.0

    
    assert not all([p.requires_grad for p in lambda_dict["lambda_ema_mlps"].parameters()]), "Target should not require gradients"

    # optionally resume from a checkpoint
    start_epoch = 0
    if args.resume is not None:
        checkpoint = pt_load(args.resume, map_location='cpu')
        if 'epoch' in checkpoint:
            # resuming a train checkpoint w/ epoch and optimizer state
            start_epoch = checkpoint["epoch"]
            sd = checkpoint["state_dict"]
            if args.distributed and not next(iter(sd.keys())).startswith('module'):
                sd = {f'module.{k}': v for k, v in sd.items()}
            elif not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                sd = {k[len('module.'):]: v for k, v in sd.items()}
            model.load_state_dict(sd)
            if model_optimizer is not None:
                model_optimizer.load_state_dict(checkpoint["model_optimizer"])
            if scaler is not None and 'scaler' in checkpoint:
                scaler.load_state_dict(checkpoint['scaler'])
                
            #  Load `COND_EXP`-specific components
            if "lambda_mlps_state_dict" in checkpoint:
                sd = checkpoint["lambda_mlps_state_dict"]
                if args.distributed and not next(iter(sd.keys())).startswith('module'):
                    # If the current run is DDP but the checkpoint is not, prefix keys with 'module.'
                    sd = {f'module.{k}': v for k, v in sd.items()}
                elif not args.distributed and next(iter(sd.items()))[0].startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                lambda_dict["lambda_mlps"].load_state_dict(sd)
            if "lambda_scaler" in checkpoint:
                lambda_dict["lambda_scaler"].load_state_dict(checkpoint["lambda_scaler"])
            if "lambda_optimizer" in checkpoint:
                lambda_dict["lambda_optimizer"].load_state_dict(checkpoint["lambda_optimizer"])
            if "lambda_ema_mlps_state_dict" in checkpoint:
                sd = checkpoint["lambda_ema_mlps_state_dict"]
                if next(iter(sd.keys())).startswith('module'):
                    sd = {k[len('module.'):]: v for k, v in sd.items()}
                lambda_dict["lambda_ema_mlps"].load_state_dict(sd)
                # set lambda_ema_mlps not require gradients
                for p in lambda_dict["lambda_ema_mlps"].parameters():
                    p.requires_grad = False
            logging.info(f"=> resuming checkpoint '{args.resume}' (epoch {start_epoch})")
        else:
            # loading a bare (model only) checkpoint for fine-tune or evaluation
            model.load_state_dict(checkpoint)
            logging.info(f"=> loaded checkpoint '{args.resume}' (epoch {start_epoch})")

    # initialize datasets
    tokenizer = get_tokenizer(args.model, cache_dir=args.cache_dir)
    data = get_data(
        args,
        (preprocess_train, preprocess_val),
        epoch=start_epoch,
        tokenizer=tokenizer,
    )
    assert len(data), 'At least one train or eval dataset must be specified.'

    # create scheduler if train
    scheduler = None
    if 'train' in data and model_optimizer is not None:
        total_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs
        
        if args.lr_tau < 0:
            from open_clip_train.scheduler import cosine_lr, const_lr, const_lr_cooldown

            if args.lr_scheduler == "cosine":
                scheduler = cosine_lr(model_optimizer, args.lr, args.warmup, total_steps)
            elif args.lr_scheduler == "const":
                scheduler = const_lr(model_optimizer, args.lr, args.warmup, total_steps)
            elif args.lr_scheduler == "const-cooldown":
                assert args.epochs_cooldown is not None,\
                    "Please specify the number of cooldown epochs for this lr schedule."
                cooldown_steps = (data["train"].dataloader.num_batches // args.accum_freq) * args.epochs_cooldown
                scheduler = const_lr_cooldown(
                    model_optimizer, args.lr, args.warmup, total_steps,
                    cooldown_steps, args.lr_cooldown_power, args.lr_cooldown_end)
            elif args.lr_scheduler == "cyclic":
                scheduler = RestartCosineScheduler(
                    model_optimizer, args.lr, args.warmup, total_steps, restart_step=117_000)
            else:
                logging.error(
                    f'Unknown scheduler, {args.lr_scheduler}. Available options are: cosine, const, const-cooldown.')
                exit(1)
            schedulers = [scheduler]
        else:
            from open_clip_train.param_scheduler import cosine_lr, const_lr, const_lr_cooldown, step_lr_thresh
            param_groups = model_optimizer.param_groups
            logit_scale_group = param_groups[-1:]
            other_groups = param_groups[:-1]
            model_scheduler = cosine_lr(other_groups, args.lr, args.warmup, total_steps)
            scale_scheduler = step_lr_thresh(logit_scale_group, args.lr_tau, 0, [0.03], [1/3], model=model if not hasattr(model, "module") else model.module)
            schedulers = [model_scheduler, scale_scheduler]

    # determine if this worker should save logs and checkpoints. only do so if it is rank == 0
    args.save_logs = args.logs and args.logs.lower() != 'none' and is_master(args)
    writer = None
    if args.save_logs and args.tensorboard:
        assert tensorboard is not None, "Please install tensorboard."
        writer = tensorboard.SummaryWriter(args.tensorboard_path)

    if args.wandb and is_master(args):
        assert wandb is not None, 'Please install wandb.'
        logging.debug('Starting wandb.')
        args.train_sz = data["train"].dataloader.num_samples
        if args.val_data is not None:
            args.val_sz = data["val"].dataloader.num_samples
        # you will have to configure this for your project!
        wandb.init(
            project=args.wandb_project_name,
            name=args.name,
            id=args.name,
            notes=args.wandb_notes,
            tags=[],
            resume='auto' if args.resume == "latest" else None,
            config=vars(args),
        )
        if args.debug:
            wandb.watch(model, log='all')
        wandb.save(params_file)
        logging.debug('Finished loading wandb.')

    # Pytorch 2.0 adds '_orig_mod.' prefix to keys of state_dict() of compiled models.
    # For compatibility, we save state_dict() of the original model, which shares the
    # weights without the prefix.
    original_model = model
    if args.torchcompile:
        logging.info('Compiling model...')

        if args.grad_checkpointing and args.distributed:
            logging.info('Disabling DDP dynamo optimizer when grad checkpointing enabled.')
            # As of now (~PyTorch 2.4/2.5), compile + checkpointing but DDP optimizer must be disabled
            torch._dynamo.config.optimize_ddp = False

        model = torch.compile(original_model)

    if 'train' not in data:
        # If using int8, convert to inference mode.
        if args.use_bnb_linear is not None:
            from open_clip.utils import convert_int8_model_to_inference_mode
            convert_int8_model_to_inference_mode(model)
        evaluate(model, data, start_epoch, args, tb_writer=writer, tokenizer=tokenizer, transform=preprocess_val)
        return



    # * implemented new loss
    model_loss = ConditionalExpLoss(
        rank=args.rank,
        world_size=args.world_size,
        scale_loss=args.scale_loss,
        pos_coef=args.pos_coef,
        lambda_eps=args.lambda_eps,
        calculate_full=args.calculate_full,
        adaptive_tau=args.adaptive_tau,
        lambda_fit_neg_only=args.lambda_fit_neg_only,
    )

    # * Target collector
    log_target_collector = LogTargetCollector(
        rank=args.rank,
        world_size=args.world_size,
        calculate_full=args.calculate_full,
        pos_coef=args.pos_coef,
        lambda_fit_neg_only=args.lambda_fit_neg_only,
    )

    for epoch in range(start_epoch, args.epochs):
        if is_master(args):
            logging.info(f'Start epoch {epoch}')

        
        lambda_dict["lambda_ema_updater"].update_beta_by_value(
            lambda_dict["ema_model_beta_func"](epoch)
        )
        
        
        if args.fit_w_prev:
            ema_model = lambda_dict["lambda_ema_mlps"]
            if isinstance(ema_model, torch.nn.parallel.DistributedDataParallel):
                ema_model = ema_model.module
            lambda_dict["lambda_prev_mlps"] = copy.deepcopy(ema_model)
            device = next(ema_model.parameters()).device
            lambda_dict["lambda_prev_mlps"].to(device)
            for param in lambda_dict["lambda_prev_mlps"].parameters():
                param.requires_grad = False
                
        if args.reinit_lambda_every_n_epochs > 0 and epoch % args.reinit_lambda_every_n_epochs == 0:
            lambda_dict = reinitialize_lambda_mlps(lambda_dict, args, embed_dim, device)
            if is_master(args):
                logging.info(f'Reinitialized lambda_mlps.')
        
        model.train()
        lambda_dict["lambda_mlps"].train()
        
        data['train'].set_epoch(epoch)  # set epoch in process safe manner via sampler or shared_epoch
        dataloader = data['train'].dataloader
        num_batches_per_epoch = dataloader.num_batches 

        losses_m = {}
        batch_time_m = AverageMeter()
        data_time_m = AverageMeter()
        end_time = time.time()
        samples_per_epoch = dataloader.num_samples
        
        update_ema_mlps = True
        train_lambda_mlps = True

        for step, batch in enumerate(dataloader):
            data_time_m.update(time.time() - end_time)
            i_accum = step // args.accum_freq

            train_lambda_mlps = step % args.update_lambda_every_n_steps == 0
            update_ema_mlps = step % args.update_ema_mlps_every_n_steps == 0
            
            losses = optimize_step(
                    batch,
                    model, model_loss, model_optimizer, 
                    lambda_dict, 
                    epoch, step, num_batches_per_epoch,
                    scaler, schedulers, log_target_collector, args, update_ema_mlps, train_lambda_mlps
                )
            
            batch_time_m.update(time.time() - end_time)
            end_time = time.time()
            batch_count = i_accum + 1
            
            if is_master(args) and (i_accum % args.log_every_n_steps == 0 or batch_count == num_batches_per_epoch):
                log_func(
                    losses,
                    step, epoch,
                    args.batch_size, num_batches_per_epoch, samples_per_epoch, batch_count,
                    data_time_m, batch_time_m, losses_m, model_optimizer, lambda_dict,
                    args, tb_writer=writer
                )


        completed_epoch = epoch + 1
        

        if any(v in data for v in ('val', 'imagenet-val', 'imagenet-v2')):
            eval_model = model.module if hasattr(model, "module") else model
            evaluate(eval_model, data, completed_epoch, args, tb_writer=writer, tokenizer=tokenizer, transform=preprocess_val)

        # Saving checkpoints.
        if args.save_logs:
            checkpoint_dict = {
                "epoch": completed_epoch,
                "name": args.name,
                "state_dict": original_model.state_dict(),
                "model_optimizer": model_optimizer.state_dict(),
            }
            checkpoint_dict["lambda_mlps_state_dict"] = lambda_dict["lambda_mlps"].state_dict()
            checkpoint_dict["lambda_scaler"] = lambda_dict["lambda_scaler"].state_dict()
            checkpoint_dict["lambda_optimizer"] = lambda_dict["lambda_optimizer"].state_dict()
            checkpoint_dict["lambda_ema_mlps_state_dict"] = lambda_dict["lambda_ema_mlps"].state_dict()
            if scaler is not None:
                checkpoint_dict["scaler"] = scaler.state_dict()

            if completed_epoch == args.epochs or (
                args.save_frequency > 0 and (completed_epoch % args.save_frequency) == 0
            ):
                torch.save(
                    checkpoint_dict,
                    os.path.join(args.checkpoint_path, f"epoch_{completed_epoch}.pt"),
                )
            if args.delete_previous_checkpoint:
                previous_checkpoint = os.path.join(args.checkpoint_path, f"epoch_{completed_epoch - 1}.pt")
                if os.path.exists(previous_checkpoint):
                    os.remove(previous_checkpoint)

            if args.save_most_recent:
                # try not to corrupt the latest checkpoint if save fails
                tmp_save_path = os.path.join(args.checkpoint_path, "tmp.pt")
                latest_save_path = os.path.join(args.checkpoint_path, LATEST_CHECKPOINT_NAME)
                torch.save(checkpoint_dict, tmp_save_path)
                os.replace(tmp_save_path, latest_save_path)


    model_dict = {
            "model": eval_model,
            "device": device,
            "transform": preprocess_val,
            "tokenizer": tokenizer,
        }
    evaluate_datacomp(
            path_to_results=f"{log_base_path}/dc_final.jsonl",
            data_dir=args.datacomp_dir,
            train_info=None,
            model_dict=model_dict,
            eval_ret_only=False
        )
    
    
    if is_master(args):
        path_to_csv = f"{log_base_path}/eval_results.csv"
        convert_to_csv(f"{log_base_path}/dc_final.jsonl", path_to_csv)
    
    if args.wandb and is_master(args):
        wandb.finish()

    # run a final sync.
    if remote_sync_process is not None:
        logging.info('Final remote sync.')
        remote_sync_process.terminate()
        result = remote_sync(
            os.path.join(args.logs, args.name), 
            os.path.join(args.remote_sync, args.name), 
            args.remote_sync_protocol
        )
        if result:
            logging.info('Final remote sync successful.')
        else:
            logging.info('Final remote sync failed.')
    

def copy_codebase(args):
    from shutil import copytree, ignore_patterns
    new_code_path = os.path.join(args.logs, args.name, "code")
    if os.path.exists(new_code_path):
        print(
            f"Error. Experiment already exists at {new_code_path}. Use --name to specify a new experiment."
        )
        return -1
    print(f"Copying codebase to {new_code_path}")
    current_code_path = os.path.realpath(__file__)
    for _ in range(3):
        current_code_path = os.path.dirname(current_code_path)
    copytree(current_code_path, new_code_path, ignore=ignore_patterns('log', 'logs', 'wandb'))
    print("Done copying code.")
    return 1


if __name__ == "__main__":
    main(sys.argv[1:])

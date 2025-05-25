import json
import logging
import math
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.distributed as dist

try:
    import wandb
except ImportError:
    wandb = None

from open_clip import get_input_dtype
from open_clip_train.distributed import is_master
from open_clip_train.zero_shot import zero_shot_eval
from open_clip_train.precision import get_autocast
from datacomp.evaluate import evaluate as evaluate_datacomp


import warnings
warnings.filterwarnings("ignore", message="Length of IterableDataset")


def create_epoch_schedule(values, total_epochs):
    """
    Creates an epoch-wise schedule for a given list of values.

    Args:
        values (int, float, or list): A single value or a list of values.
        total_epochs (int): Total number of epochs.

    Returns:
        list: A list containing values assigned to each epoch.
    """
    if isinstance(values, (int, float)):  # If it's a single value, repeat for all epochs
        return [values] * total_epochs

    num_parts = len(values)  # Number of segments
    part_size = total_epochs // num_parts  # Size of each segment

    schedule = []
    for i in range(num_parts):
        if i == num_parts - 1:
            # Assign the last value for the remaining epochs
            schedule.extend([values[i]] * (total_epochs - part_size * i))
        else:
            # Assign values to their corresponding epoch range
            schedule.extend([values[i]] * part_size)

    return schedule

class RestartCosineScheduler:
    """
    Callable scheduler that:
      - Warms up from 0 to base_lr for `warmup_steps`.
      - Cosine decays from `warmup_steps` to `restart_step`.
      - *Restart* at `restart_step` back to base_lr.
      - Cosine decays again from `restart_step` to `total_steps`.
      - Clamps LR to 0 after `total_steps`.
    
    Usage:
        scheduler = RestartCosineScheduler(optimizer, base_lr=0.1,
                                           warmup_steps=5, total_steps=50, restart_step=30)
        for step in range(num_train_steps):
            # train ...
            scheduler(step)  # calls scheduler to set the LR
    """

    def __init__(self, optimizer, base_lr, warmup_steps, total_steps, restart_step):
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.restart_step = restart_step

    def __call__(self, current_step: int):
        """Sets the learning rate for the optimizer based on the current step."""
        # Compute the LR factor in [0, 1], then multiply by base_lr.
        lr_factor = self._get_lr_factor(current_step)
        
        new_lr = self.base_lr * lr_factor
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

    def _get_lr_factor(self, step: int) -> float:
        """Compute the LR as a fraction of base_lr."""
        # 1) Warmup phase
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))

        # 2) Before restart
        if step < self.restart_step:
            progress = float(step - self.warmup_steps) / float(self.restart_step - self.warmup_steps)
            # half-cosine from 1 down to ~0
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        # 3) After restart
        if step <= self.total_steps:
            progress2 = float(step - self.restart_step) / float(self.total_steps - self.restart_step)
            return 0.5 * (1.0 + math.cos(math.pi * progress2))

        # beyond total_steps => 0 LR
        return 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def postprocess_clip_output(model_out):
    return {
        "image_features": model_out[0],
        "text_features": model_out[1],
        "logit_scale": model_out[2]
    }


def unwrap_model(model):
    if hasattr(model, 'module'):
        return model.module
    else:
        return model


def backward(total_loss, scaler):
    if scaler is not None:
        scaler.scale(total_loss).backward()
    else:
        total_loss.backward()


def optimize_step(
        batch, 
        model, model_loss, model_optimizer, 
        lambda_dict, 
        epoch, step, num_batches_per_epoch,
        scaler, schedulers, log_target_collector, args, update_ema_mlps=False, train_lambda_mlps=True
    ):
    device = torch.device(args.device)
    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

        
    i_accum = step // args.accum_freq
    total_step = num_batches_per_epoch * epoch + i_accum

    if not args.skip_scheduler:
        for scheduler in schedulers:
            scheduler(total_step)

    images, texts = batch
    images = images.to(device=device, dtype=input_dtype, non_blocking=True)
    texts = texts.to(device=device, non_blocking=True)

    num_lambda_updates = 0

    # * Update lambda mlp for Conditional Amortization
    lambda_mlps = lambda_dict["lambda_mlps"]
    lambda_loss = lambda_dict["lambda_loss"]
    lambda_optimizer = lambda_dict["lambda_optimizer"]
    lambda_ema_updater = lambda_dict["lambda_ema_updater"]
    lambda_ema_mlps = lambda_dict["lambda_ema_mlps"]
    lambda_scaler = lambda_dict["lambda_scaler"]   
    ema_z_beta = lambda_dict["ema_z_beta_func"](epoch)
    lambda_prev_mlps = lambda_dict.get("lambda_prev_mlps", None)
    
    with autocast():
        model_out = model(images, texts)
    
    lambda_update_frequency = create_epoch_schedule(args.lambda_update_frequency, args.epochs)
    rho_schedule = create_epoch_schedule(args.rho_list, args.epochs)
    
    
    with torch.no_grad():
        image_features_detached, text_features_detached = (
            model_out[k].detach() for k in ["image_features", "text_features"])
        logit_scale = model_out["logit_scale"].detach()  # scale of the inner product
        logit_bias = model_out.get("logit_bias", 
                    torch.tensor(0.0, device=image_features_detached.device)).detach()  # bias of the inner product       

        log_target_image = log_target_collector(image_features_detached, text_features_detached, logit_scale, logit_bias)
        log_target_text = log_target_collector(text_features_detached, image_features_detached, logit_scale, logit_bias)
        
        if args.fit_w_prev and lambda_prev_mlps is not None:
            with torch.no_grad():
                prev_lambda_out = lambda_prev_mlps(image_features_detached, text_features_detached)
                prev_lambda_image, prev_lambda_text = (
                    prev_lambda_out[k].exp().detach() for k in ["log_lambda_image", "log_lambda_text"]
                )
                log_target_image = (ema_z_beta * prev_lambda_image + (1 - ema_z_beta) * log_target_image.exp()).log()
                log_target_text = (ema_z_beta * prev_lambda_text + (1 - ema_z_beta) * log_target_text.exp()).log()
            
    # * lambda loss
    lambda_losses = None
    if train_lambda_mlps:
        previous_loss = None
        convergence_count = 0  # To track how long we’ve seen no significant improvement
        for num_lambda_updates in range(lambda_update_frequency[epoch]):
            if args.denormalize_features:
                current_lambda_out = lambda_mlps(
                    image_features_detached, text_features_detached, logit_scale.detach())

            else:
                current_lambda_out = lambda_mlps(
                    image_features_detached, text_features_detached)
                
        
            detached_model_out = {k: v.detach() for k, v in model_out.items()}
            detached_model_out["logit_bias"] = logit_bias 
            
            lambda_losses = lambda_loss(**detached_model_out, 
                                        log_target_image=log_target_image, 
                                        log_target_text=log_target_text, 
                                        current_lambda_out=current_lambda_out, 
                                        output_dict=True,
                                        # force_l2_loss=step<num_batches_per_epoch*0.05, # !todo: abl this    
                                    )
            total_loss = lambda_losses.pop("total_loss")

            # Check for convergence
            if previous_loss is not None:
                loss_change = abs(previous_loss - total_loss.item())
                if loss_change < args.lambda_tolerance:
                    convergence_count += 1
                else:
                    convergence_count = 0  # Reset count if loss improves significantly
                
            if args.world_size > 1:
                # ----- rank 0 decides whether to stop -----
                if args.rank == 0:
                    local_stop = int(convergence_count >= 5)
                else:
                    local_stop = 0

                # Put the decision in a tensor and broadcast to all ranks
                stop_early_tensor = torch.tensor([local_stop], dtype=torch.int, device=image_features_detached.device)
                dist.broadcast(stop_early_tensor, src=0)

                if stop_early_tensor.item() == 1:
                    break
            elif convergence_count >= 5:
                    break
            
            previous_loss = total_loss.item()
            
            lambda_optimizer.zero_grad()
            backward(total_loss, scaler=lambda_scaler)
            if lambda_scaler is not None:
                if args.horovod:
                    lambda_optimizer.synchronize()
                    lambda_scaler.unscale_(lambda_optimizer)
                    if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(lambda_mlps.parameters(), args.grad_clip_norm, norm_type=2.0)
                    with lambda_optimizer.skip_synchronize():
                        lambda_scaler.step(lambda_optimizer)
                else:
                    if args.grad_clip_norm is not None:
                        lambda_scaler.unscale_(lambda_optimizer)
                        torch.nn.utils.clip_grad_norm_(lambda_mlps.parameters(), args.grad_clip_norm, norm_type=2.0)
                    lambda_scaler.step(lambda_optimizer)
                lambda_scaler.update()
            else:
                if args.grad_clip_norm is not None:
                        torch.nn.utils.clip_grad_norm_(lambda_mlps.parameters(), args.grad_clip_norm, norm_type=2.0)
                lambda_optimizer.step()
        
    # query     
    with torch.no_grad():
        if args.fit_w_prev:
            ema_lambda_out = {
                "log_lambda_image": log_target_image,
                "log_lambda_text": log_target_text
            }
        else:
            if args.denormalize_features:
                ema_lambda_out = lambda_ema_mlps(image_features_detached, text_features_detached, logit_scale.detach())
            else:
                ema_lambda_out = lambda_ema_mlps(image_features_detached, text_features_detached)
        

    if update_ema_mlps:
        lambda_ema_updater.update_moving_average(lambda_ema_mlps, lambda_mlps)

    # * model loss
    model_optimizer.zero_grad()
    
    if args.accum_freq == 1:
        with autocast():
            logit_scale = model_out["logit_scale"]
            logit_bias = model_out.get("logit_bias", torch.tensor(0.))
            model_out["logit_bias"] = logit_bias 
            
            losses = model_loss(**model_out, rho=rho_schedule[epoch], ema_lambda_out=ema_lambda_out, output_dict=True)
            total_loss = losses["total_loss"]
            
        backward(total_loss, scaler)
    else:
        raise ValueError("Unsupported accumulation frequency")
    
    grad_norm = 0.
    if scaler is not None:
        if args.horovod:
            model_optimizer.synchronize()
            scaler.unscale_(model_optimizer)
            if args.grad_clip_norm is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            with model_optimizer.skip_synchronize():
                scaler.step(model_optimizer)
        else:
            if args.grad_clip_norm is not None:
                scaler.unscale_(model_optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
            scaler.step(model_optimizer)
        scaler.update()
    else:
        if args.grad_clip_norm is not None:
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip_norm, norm_type=2.0)
        model_optimizer.step()

    # Note: we clamp to 4.6052 = ln(100), as in the original paper.
    with torch.no_grad():
        unwrap_model(model).logit_scale.clamp_(0, math.log(args.logit_scale_clamp))

    losses.update({"logit_scale": logit_scale, 
                   "logit_bias": logit_bias,
                    "grad_norm": grad_norm.sum().item(),
                  "num_lambda_updates": num_lambda_updates})
    if lambda_losses is not None:
        losses.update(lambda_losses)

    return losses

def log_func(
        losses,
        step, epoch,
        batch_size, num_batches_per_epoch, samples_per_epoch, batch_count,
        data_time_m, batch_time_m, losses_m, model_optimizer, lambda_dict,
        args, tb_writer=None
    ):
    num_samples = batch_count * batch_size * args.accum_freq * args.world_size
    percent_complete = 100.0 * batch_count / num_batches_per_epoch
    total_step = num_batches_per_epoch * epoch + step

    logit_scale_scalar = losses.pop("logit_scale").item() if "logit_scale" in losses.keys() and losses['logit_scale'] else 0.
    logit_bias_scalar = losses.pop("logit_bias").item() if "logit_bias" in losses.keys() and losses['logit_bias'] else 0.
    grad_norm = losses.pop("grad_norm") if "grad_norm" in losses.keys() else 0.
    num_lambda_updates = losses.pop("num_lambda_updates") if "num_lambda_updates" in losses.keys() else 0
    
    
    ema_z_beta = lambda_dict["ema_z_beta_func"](epoch)
    ema_model_beta = lambda_dict["ema_model_beta_func"](epoch)
    
    for key, val in losses.items():
        if key not in losses_m:
            losses_m[key] = AverageMeter()
        if val is not None:
            losses_m[key].update(val.item(), batch_size)
        
    model_loss_log = " ".join(
        [f"{loss_name.capitalize()}: {loss_m.val:#.5g} ({loss_m.avg:#.5g})" 
            for loss_name, loss_m in losses_m.items() if loss_m is not None]
    )
    
    samples_per_second = args.accum_freq * args.batch_size * args.world_size / batch_time_m.val
    samples_per_second_per_gpu = args.accum_freq * args.batch_size / batch_time_m.val
    sample_digits = math.ceil(math.log(samples_per_epoch + 1, 10))

    logging.info(
        f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
        f"Step: {total_step/1000:.1f}K | "
        f"Data (t): {data_time_m.avg:.3f} "
        f"Batch (t): {batch_time_m.avg:.3f}, {samples_per_second:#g}/s, {samples_per_second_per_gpu:#g}/s/gpu "
        f"Logit Scale: {logit_scale_scalar:.3f} | Logit Bias: {logit_bias_scalar:.3f}  | "
        f"| {model_loss_log} | L_updates: {num_lambda_updates} | z_beta: {ema_z_beta}, model_beta: {ema_model_beta} | "
    )

    # Save train loss / etc. Using non avg meter values as loggers have their own smoothing
    log_data = {
        "data_time": data_time_m.val,
        "batch_time": batch_time_m.val,
        "samples_per_second": samples_per_second,
        "samples_per_second_per_gpu": samples_per_second_per_gpu,
        "scale": logit_scale_scalar,
        "grad_norm": grad_norm,
        "logit_bias": logit_bias_scalar,
        "lr": model_optimizer.param_groups[0]["lr"],
        "num_lambda_updates": num_lambda_updates,
        "ema_z_beta": ema_z_beta,
        "ema_model_beta": ema_model_beta,
    }            
    log_data.update({name:val.val for name,val in losses_m.items()})
    log_data.update(losses)

    log_data = {f"train_cond_exp/" + name: val for name, val in log_data.items()}

    if tb_writer is not None:
        for name, val in log_data.items():
            tb_writer.add_scalar(name, val, total_step)
    
    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        log_data['step'] = total_step  # for backwards compatibility
        wandb.log(log_data, step=total_step)
    
    # resetting batch / data time meters per log window
    batch_time_m.reset()
    data_time_m.reset()



def evaluate(model, data, epoch, args, tb_writer=None, tokenizer=None, transform=None):
    metrics = {}
    if not is_master(args):
        return metrics
    device = torch.device(args.device)
    model.eval()

    if args.datacomp_dir is not None:
        log_base_path = f"{args.logs}/{args.name}/dc_eval"
        os.makedirs(log_base_path, exist_ok=True)

        model_dict = {
            "model": model,
            "device": device,
            "transform": transform,
            "tokenizer": tokenizer,
        }
        datacomp_metrics = evaluate_datacomp(
            path_to_results=f"{log_base_path}/epoch{epoch}.jsonl",
            data_dir=args.datacomp_dir,
            train_info=None,
            model_dict=model_dict,
            eval_ret_only=True
        )
        # extract results[task_name]['metrics']['main_metric'] for each task and update metrics
        for task_name, task_metrics in datacomp_metrics.items():
            metrics[task_name] = task_metrics["metrics"]["main_metric"]


    zero_shot_metrics = zero_shot_eval(model, data, epoch, args, tokenizer=tokenizer)
    metrics.update(zero_shot_metrics)

    autocast = get_autocast(args.precision, device_type=device.type)
    input_dtype = get_input_dtype(args.precision)

    if 'val' in data and (args.val_frequency and ((epoch % args.val_frequency) == 0 or epoch == args.epochs)):
        dataloader = data['val'].dataloader
        num_samples = 0
        samples_per_val = dataloader.num_samples

        # FIXME this does not scale past small eval datasets
        # all_image_features @ all_text_features will blow up memory and compute very quickly
        cumulative_loss = 0.0
        cumulative_gen_loss = 0.0
        all_image_features, all_text_features = [], []
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                images, texts = batch
                images = images.to(device=device, dtype=input_dtype, non_blocking=True)
                texts = texts.to(device=device, non_blocking=True)

                with autocast():
                    model_out = model(images, texts)
                    image_features = model_out["image_features"]
                    text_features = model_out["text_features"]
                    logit_scale = model_out["logit_scale"] 
                    # features are accumulated in CPU tensors, otherwise GPU memory exhausted quickly
                    # however, system RAM is easily exceeded and compute time becomes problematic
                    all_image_features.append(image_features.cpu())
                    all_text_features.append(text_features.cpu())
                    logit_scale = logit_scale.mean()
                    logits_per_image = logit_scale * image_features @ text_features.t()
                    logits_per_text = logits_per_image.t()

                    batch_size = images.shape[0]
                    labels = torch.arange(batch_size, device=device).long()
                    total_loss = (
                        F.cross_entropy(logits_per_image, labels) +
                        F.cross_entropy(logits_per_text, labels)
                    ) / 2

                    gen_loss = maybe_compute_generative_loss(model_out)

                cumulative_loss += total_loss * batch_size
                num_samples += batch_size
                if is_master(args) and (i % 100) == 0:
                    logging.info(
                        f"Eval Epoch: {epoch} [{num_samples} / {samples_per_val}]\t"
                        f"Clip Loss: {cumulative_loss / num_samples:.6f}\t")

                    if gen_loss is not None:
                        cumulative_gen_loss += gen_loss * batch_size
                        logging.info(
                            f"Generative Loss: {cumulative_gen_loss / num_samples:.6f}\t")

            val_metrics = get_clip_metrics(
                image_features=torch.cat(all_image_features),
                text_features=torch.cat(all_text_features),
                logit_scale=logit_scale.cpu(),
            )
            loss = cumulative_loss / num_samples
            metrics.update(
                {**val_metrics, "clip_val_loss": loss.item(), "epoch": epoch, "num_samples": num_samples}
            )
            if gen_loss is not None:
                gen_loss = cumulative_gen_loss / num_samples
                metrics.update({"val_generative_loss": gen_loss.item()})

    if not metrics:
        return metrics

    logging.info(
        f"Eval Epoch: {epoch} "
        + "\t".join([f"{k}: {round(v, 4):.4f}" for k, v in metrics.items()])
    )

    log_data = {"val/" + name: val for name, val in metrics.items()}

    if args.save_logs:
        if tb_writer is not None:
            for name, val in log_data.items():
                tb_writer.add_scalar(name, val, epoch)

        with open(os.path.join(args.checkpoint_path, "results.jsonl"), "a+") as f:
            f.write(json.dumps(metrics))
            f.write("\n")

    if args.wandb:
        assert wandb is not None, 'Please install wandb.'
        if 'train' in data:
            dataloader = data['train'].dataloader
            num_batches_per_epoch = dataloader.num_batches // args.accum_freq
            step = num_batches_per_epoch * epoch
        else:
            step = None
        log_data['epoch'] = epoch
        wandb.log(log_data, step=step)

    return metrics


def get_clip_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics


def maybe_compute_generative_loss(model_out):
    if "logits" in model_out and "labels" in model_out:
        token_logits = model_out["logits"]
        token_labels = model_out["labels"]
        return F.cross_entropy(token_logits.permute(0, 2, 1), token_labels)

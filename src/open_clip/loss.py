import torch
import torch.nn as nn
from torch.nn import functional as F

import numpy as np

import random

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

from open_clip.utils import exp_inner_prod, get_dist_matrix

def gather_features(
        image_features,
        text_features,
        local_loss=False,
        gather_with_grad=False,
        rank=0,
        world_size=1,
        use_horovod=False
):
    assert has_distributed, 'torch.distributed did not import correctly, please use a PyTorch version with support.'
    if use_horovod:
        assert hvd is not None, 'Please install horovod'
        if gather_with_grad:
            all_image_features = hvd.allgather(image_features)
            all_text_features = hvd.allgather(text_features)
        else:
            with torch.no_grad():
                all_image_features = hvd.allgather(image_features)
                all_text_features = hvd.allgather(text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features = list(all_image_features.chunk(world_size, dim=0))
                gathered_text_features = list(all_text_features.chunk(world_size, dim=0))
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
                all_image_features = torch.cat(gathered_image_features, dim=0)
                all_text_features = torch.cat(gathered_text_features, dim=0)
    else:
        # We gather tensors from all gpus
        if gather_with_grad:
            all_image_features = torch.cat(torch.distributed.nn.all_gather(image_features), dim=0)
            all_text_features = torch.cat(torch.distributed.nn.all_gather(text_features), dim=0)
        else:
            gathered_image_features = [torch.zeros_like(image_features) for _ in range(world_size)]
            gathered_text_features = [torch.zeros_like(text_features) for _ in range(world_size)]
            dist.all_gather(gathered_image_features, image_features)
            dist.all_gather(gathered_text_features, text_features)
            if not local_loss:
                # ensure grads for local rank when all_* features don't have a gradient
                gathered_image_features[rank] = image_features
                gathered_text_features[rank] = text_features
            all_image_features = torch.cat(gathered_image_features, dim=0)
            all_text_features = torch.cat(gathered_text_features, dim=0)

    return all_image_features, all_text_features


class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        labels = self.get_ground_truth(device, logits_per_image.shape[0])

        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        return {"contrastive_loss": total_loss} if output_dict else total_loss

class CoCaLoss(ClipLoss):
    def __init__(
            self,
            caption_loss_weight,
            clip_loss_weight,
            pad_id=0,  # pad_token for open_clip custom tokenizer
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__(
            local_loss=local_loss,
            gather_with_grad=gather_with_grad,
            cache_labels=cache_labels,
            rank=rank,
            world_size=world_size,
            use_horovod=use_horovod
        )

        self.clip_loss_weight = clip_loss_weight
        self.caption_loss_weight = caption_loss_weight
        self.caption_loss = nn.CrossEntropyLoss(ignore_index=pad_id)

    def forward(self, image_features, text_features, logits, labels, logit_scale, output_dict=False):
        
        clip_loss = torch.tensor(0)
        
        if self.clip_loss_weight:
            clip_loss = super().forward(image_features, text_features, logit_scale)
            clip_loss = self.clip_loss_weight * clip_loss

        caption_loss = self.caption_loss(
            logits.permute(0, 2, 1),
            labels,
        )
        caption_loss = caption_loss * self.caption_loss_weight

        if output_dict:
            return {"contrastive_loss": clip_loss, "caption_loss": caption_loss}

        return clip_loss, caption_loss


class DistillClipLoss(ClipLoss):

    def dist_loss(self, teacher_logits, student_logits):
        return -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=0)

    def forward(
            self,
            image_features,
            text_features,
            logit_scale,
            dist_image_features,
            dist_text_features,
            dist_logit_scale,
            output_dict=False,
    ):
        logits_per_image, logits_per_text = \
            self.get_logits(image_features, text_features, logit_scale)

        dist_logits_per_image, dist_logits_per_text = \
            self.get_logits(dist_image_features, dist_text_features, dist_logit_scale)

        labels = self.get_ground_truth(image_features.device, logits_per_image.shape[0])

        contrastive_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
        ) / 2

        distill_loss = (
            self.dist_loss(dist_logits_per_image, logits_per_image) +
            self.dist_loss(dist_logits_per_text, logits_per_text)
        ) / 2

        if output_dict:
            return {"contrastive_loss": contrastive_loss, "distill_loss": distill_loss}

        return contrastive_loss, distill_loss


def neighbour_exchange(from_rank, to_rank, tensor, group=None):
    tensor_recv = torch.zeros_like(tensor)
    send_op = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor,
        to_rank,
        group=group,
    )
    recv_op = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_recv,
        from_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op, recv_op])
    for req in reqs:
        req.wait()
    return tensor_recv


def neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    tensor_from_left = torch.zeros_like(tensor_to_right)
    tensor_from_right = torch.zeros_like(tensor_to_left)
    send_op_left = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_left,
        left_rank,
        group=group,
    )
    send_op_right = torch.distributed.P2POp(
        torch.distributed.isend,
        tensor_to_right,
        right_rank,
        group=group,
    )
    recv_op_left = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_left,
        left_rank,
        group=group,
    )
    recv_op_right = torch.distributed.P2POp(
        torch.distributed.irecv,
        tensor_from_right,
        right_rank,
        group=group,
    )
    reqs = torch.distributed.batch_isend_irecv([send_op_right, send_op_left, recv_op_right, recv_op_left])
    for req in reqs:
        req.wait()
    return tensor_from_right, tensor_from_left


class NeighbourExchange(torch.autograd.Function):
    @staticmethod
    def forward(ctx, from_rank, to_rank, group, tensor):
        ctx.group = group
        ctx.from_rank = from_rank
        ctx.to_rank = to_rank
        return neighbour_exchange(from_rank, to_rank, tensor, group=group)

    @staticmethod
    def backward(ctx, grad_output):
        return (None, None, None) + (NeighbourExchange.apply(ctx.to_rank, ctx.from_rank, ctx.group, grad_output),)


def neighbour_exchange_with_grad(from_rank, to_rank, tensor, group=None):
    return NeighbourExchange.apply(from_rank, to_rank, group, tensor)


class NeighbourExchangeBidir(torch.autograd.Function):
    @staticmethod
    def forward(ctx, left_rank, right_rank, group, tensor_to_left, tensor_to_right):
        ctx.group = group
        ctx.left_rank = left_rank
        ctx.right_rank = right_rank
        return neighbour_exchange_bidir(left_rank, right_rank, tensor_to_left, tensor_to_right, group=group)

    @staticmethod
    def backward(ctx, *grad_outputs):
        return (None, None, None) + \
            NeighbourExchangeBidir.apply(ctx.right_rank, ctx.left_rank, ctx.group, *grad_outputs)


def neighbour_exchange_bidir_with_grad(left_rank, right_rank, tensor_to_left, tensor_to_right, group=None):
    return NeighbourExchangeBidir.apply(left_rank, right_rank, group, tensor_to_left, tensor_to_right)


class SigLipLoss(nn.Module):
    """ Sigmoid Loss for Language Image Pre-Training (SigLIP) - https://arxiv.org/abs/2303.15343

    @article{zhai2023sigmoid,
      title={Sigmoid loss for language image pre-training},
      author={Zhai, Xiaohua and Mustafa, Basil and Kolesnikov, Alexander and Beyer, Lucas},
      journal={arXiv preprint arXiv:2303.15343},
      year={2023}
    }
    """
    def __init__(
            self,
            cache_labels=False,
            rank=0,
            world_size=1,
            bidir=True,
            use_horovod=False,
    ):
        super().__init__()
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        assert not use_horovod  # FIXME need to look at hvd ops for ring transfers
        self.use_horovod = use_horovod
        self.bidir = bidir

        # cache state FIXME cache not currently used, worthwhile?
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, dtype, num_logits, negative_only=False) -> torch.Tensor:
        labels = -torch.ones((num_logits, num_logits), device=device, dtype=dtype)
        if not negative_only:
            labels = 2 * torch.eye(num_logits, device=device, dtype=dtype) + labels
        return labels

    def get_logits(self, image_features, text_features, logit_scale, logit_bias=None):
        logits = logit_scale * image_features @ text_features.T
        if logit_bias is not None:
            logits += logit_bias
        return logits

    def _loss(self, image_features, text_features, logit_scale, logit_bias=None, negative_only=False):
        logits = self.get_logits(image_features, text_features, logit_scale, logit_bias)
        labels = self.get_ground_truth(
            image_features.device,
            image_features.dtype,
            image_features.shape[0],
            negative_only=negative_only,
        )
        loss = -F.logsigmoid(labels * logits).sum() / image_features.shape[0]
        return loss

    def forward(self, image_features, text_features, logit_scale, logit_bias, output_dict=False):
        loss = self._loss(image_features, text_features, logit_scale, logit_bias)

        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_features_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )

                    for f in text_features_recv:
                        loss += self._loss(
                            image_features,
                            f,
                            logit_scale,
                            logit_bias,
                            negative_only=True,
                        )
                    text_features_to_left, text_features_to_right = text_features_recv

                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_recv,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
            else:
                text_features_to_right = text_features
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)

                    loss += self._loss(
                        image_features,
                        text_features_from_left,
                        logit_scale,
                        logit_bias,
                        negative_only=True,
                    )
                    text_features_to_right = text_features_from_left

        return {"contrastive_loss": loss} if output_dict else loss


# * Added func
def get_cross_batch(image_features, text_features, logit_scale, logit_bias, is_same_batch, pos_coef, mask_diagonal=False):
    dist_matrix = get_dist_matrix(
        image_features, text_features, 
        scale=logit_scale, logit_bias=logit_bias)
    
    diag_mask = torch.eye(dist_matrix.size(0), dtype=torch.bool, device=dist_matrix.device)
    if pos_coef > 0 and is_same_batch:
        dist_matrix[diag_mask] = dist_matrix[diag_mask] + np.log(pos_coef)

    if mask_diagonal and is_same_batch:
        mask = ~diag_mask
    else:
        # keep positive
        mask = torch.ones_like(dist_matrix, dtype=torch.bool, device=dist_matrix.device) 
    dist_matrix = dist_matrix.masked_fill(~mask, float('-inf'))
    
    # get numerators
    cross_image_batch = torch.logsumexp(dist_matrix, dim=-1).exp() # E_y, (B,)
    cross_text_batch = torch.logsumexp(dist_matrix, dim=0).exp() # E_x, (B,)

    
    return {
        "cross_image_batch": cross_image_batch,
        "cross_text_batch": cross_text_batch,
        "dist_matrix": dist_matrix,
    }


# * Added losses
class ConditionalExpLoss(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
            bidir=True,
            scale_loss=False,
            pos_coef=1.,
            lambda_eps=1e-6,
            calculate_full=False,
            adaptive_tau=False,
            lambda_fit_neg_only=False,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.bidir = bidir
        self.scale_loss = scale_loss
        self.pos_coef = pos_coef
        self.lambda_eps = lambda_eps
        self.calculate_full = calculate_full
        self.adaptive_tau = adaptive_tau
        self.lambda_fit_neg_only = lambda_fit_neg_only

    def _negative_loss(
        self,
        image_features, 
        text_features, 
        logit_scale, 
        logit_bias,
        lambda_pos_image,
        lambda_pos_text,
        is_same_batch,
    ): 
        
        assert lambda_pos_image.size(0) == 2 and lambda_pos_text.size(0) == 2, f"Lambda should be 2D tensor: {lambda_pos_image.size()}, {lambda_pos_text.size()}"
        lambda_image, pos_exp_image = lambda_pos_image[0], lambda_pos_image[1]       
        lambda_text, pos_exp_text = lambda_pos_text[0], lambda_pos_text[1]
        
        
        assert (not is_same_batch) or all(pos_exp_image == pos_exp_text), "Lambda should be same for image and text for same batch"
        assert is_same_batch or any(pos_exp_image != pos_exp_text) , "Lambda should be different for image and text for different batch"
        
        outputs = get_cross_batch(
            image_features, text_features, logit_scale, logit_bias, 
            is_same_batch=is_same_batch, pos_coef=self.pos_coef, mask_diagonal=False,
        )
        cross_image_batch, cross_text_batch, dist_matrix = (
            outputs[k] for k in ["cross_image_batch", "cross_text_batch", "dist_matrix"]
        )
        
        assert not (lambda_image.requires_grad or lambda_text.requires_grad), "lambda should be detached"
        assert cross_image_batch.requires_grad and cross_image_batch.requires_grad, "cross should have gradient"
        
        if self.pos_coef > 0 and self.lambda_fit_neg_only:
            lambda_image = lambda_image + pos_exp_image * self.pos_coef
            lambda_text = lambda_text + pos_exp_text * self.pos_coef
        
        cross_image = (cross_image_batch / lambda_image).mean() # E_x[ E_y[ D(x, y) / l(x) ] ]
        cross_text = (cross_text_batch / lambda_text).mean() # E_y[ E_x[ D(x, y) / l(y) ] ]

        return cross_image + cross_text


    def forward(self, 
                image_features, text_features, logit_scale, logit_bias,
                ema_lambda_out, rho, 
                output_dict=False,
            ): 
        log_lambda_image, log_lambda_text = (
            ema_lambda_out[k].detach() + self.lambda_eps 
                for k in ["log_lambda_image", "log_lambda_text"]
        )

        lambda_image, lambda_text = log_lambda_image.exp(), log_lambda_text.exp()
        
        # * matched image and text features
        with torch.no_grad():
            pos_exp = (
                exp_inner_prod(image_features, text_features, mean_exp=False) * logit_scale + logit_bias
                ).exp().squeeze().detach()
            lambda_pos_image = torch.stack([lambda_image, pos_exp], dim=0).detach()
            lambda_pos_text = torch.stack([lambda_text, pos_exp], dim=0).detach()
            
        # positive loss
        exp_joint = (
                exp_inner_prod(image_features, text_features, mean_exp=False) * 2.
            ).mean() * logit_scale 
            
        # negative loss
        exp_cross = self._negative_loss(
            image_features, text_features, logit_scale, logit_bias, 
            lambda_pos_image, lambda_pos_text, is_same_batch=True
        )
        if self.world_size > 1:
            # exchange text features w/ neighbour world_size - 1 times
            right_rank = (self.rank + 1) % self.world_size
            left_rank = (self.rank - 1 + self.world_size) % self.world_size
            if self.bidir:
                text_features_to_right = text_features_to_left = text_features
                lambda_pos_text_to_right = lambda_pos_text_to_left = lambda_pos_text.detach()
                num_bidir, remainder = divmod(self.world_size - 1, 2)
                for i in range(num_bidir):
                    text_feats_recv  = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        text_features_to_left,
                        text_features_to_right,
                    )
                    
                    lambda_pos_text_recv = neighbour_exchange_bidir_with_grad(
                        left_rank,
                        right_rank,
                        lambda_pos_text_to_left,
                        lambda_pos_text_to_right,
                    )
                    
                    for idx, (remote_text_features, remote_lambda_pos_text) in enumerate(
                        zip(text_feats_recv, lambda_pos_text_recv)
                    ):
                        exp_cross += self._negative_loss(
                            image_features, remote_text_features, logit_scale, logit_bias,
                            lambda_pos_image, remote_lambda_pos_text, is_same_batch=False,
                        )
                     
                    text_features_to_left, text_features_to_right = text_feats_recv
                    lambda_pos_text_to_left, lambda_pos_text_to_right = lambda_pos_text_recv

                    
                if remainder:
                    text_features_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    lambda_pos_text_recv = neighbour_exchange_with_grad(
                        left_rank, right_rank, lambda_pos_text_to_right)
                    exp_cross += self._negative_loss(
                        image_features, text_features_recv, logit_scale, logit_bias,
                        lambda_pos_image, lambda_pos_text_recv, is_same_batch=False,
                    )

            else:
                text_features_to_right = text_features
                lambda_pos_text_to_right = lambda_pos_text.detach()
                for i in range(self.world_size - 1):
                    text_features_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, text_features_to_right)
                    lambda_pos_text_from_left = neighbour_exchange_with_grad(
                        left_rank, right_rank, lambda_pos_text_to_right)
                    exp_cross += self._negative_loss(
                        image_features, text_features_from_left, logit_scale, logit_bias,
                        lambda_pos_image, lambda_pos_text_from_left, is_same_batch=False,
                    )
                    text_features_to_right = text_features_from_left
                    lambda_pos_text_to_right = lambda_pos_text_from_left
        
        if not self.calculate_full:
            exp_cross = exp_cross / self.world_size  
        total_loss = - exp_joint + exp_cross
        
        if self.scale_loss:
            total_loss = total_loss / logit_scale.detach()
            total_loss = total_loss + rho / logit_scale
            if self.adaptive_tau:
                B = log_lambda_image.size(0)
                total_loss = total_loss + (
                    - exp_joint.detach() + (log_lambda_image + log_lambda_text - 2 * np.log(B)).mean().detach()
                ) / logit_scale
            
        return {
                "total_loss": total_loss,
                "joint_loss": exp_joint,
                "cross_loss": exp_cross,
                } if output_dict else total_loss
        

class LambdaLoss(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
            lambda_loss_type=False,
            pos_coef=1.,
            lambda_fit_neg_only=False,
            div_reg_coef=0.1,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.lambda_loss_type = lambda_loss_type
        self.pos_coef = pos_coef
        self.lambda_fit_neg_only = lambda_fit_neg_only
        self.div_reg_coef = div_reg_coef

    def forward(self, 
                image_features, 
                text_features, 
                logit_scale, 
                logit_bias,
                log_target_image, log_target_text, current_lambda_out, 
                output_dict=False,
                force_l2_loss=False,
            ):

        # lambda_image: l(x), lambda_text: l(y)
        log_lambda_image, log_lambda_text = (current_lambda_out[k] for k in ["log_lambda_image", "log_lambda_text"])

        lambda_loss_image = F.mse_loss(log_lambda_image.squeeze(), log_target_image.squeeze().detach()) # sum_x ||l(x) - sg(E_y[D(x, y)])||^2
        lambda_loss_text = F.mse_loss(log_lambda_text.squeeze(), log_target_text.squeeze().detach()) # sum_y ||l(y) - sg(E_x[D(x, y)])||^2
        
        # Current version supports KL for single GPU and log-L2 for multi-GPU training
        if "div" in self.lambda_loss_type and not force_l2_loss:
            with torch.no_grad():
                outputs = get_cross_batch(
                    image_features, text_features, logit_scale, logit_bias, 
                    is_same_batch=True,
                    pos_coef=self.pos_coef,
                    mask_diagonal=False,
                )
            
            lambda_image, lambda_text = log_lambda_image.exp(), log_lambda_text.exp() 
            target_image, target_text = log_target_image.exp(), log_target_text.exp()       
            B = image_features.size(0)
            
            if self.pos_coef > 0 and self.lambda_fit_neg_only:
                pos_exp = outputs["dist_matrix"].diag().exp().squeeze().detach()
                
                assert pos_exp.requires_grad == False, "pos_exp should be detached"
                
                lambda_image = lambda_image + pos_exp
                lambda_text = lambda_text + pos_exp
                target_image = target_image + pos_exp
                target_text = target_text + pos_exp
                log_lambda_image, log_lambda_text = lambda_image.log(), lambda_text.log()
                log_target_image, log_target_text = target_image.log(), target_text.log()
            
            # Asymmetric KL loss
            if "kl" in self.lambda_loss_type:
                
                cross_image_batch, cross_text_batch = (
                    outputs[k] for k in ["cross_image_batch", "cross_text_batch"]
                )
                
                z_term_image = (log_target_image - log_lambda_image) / lambda_image
                z_term_text = (log_target_text - log_lambda_text) / lambda_text

                lambda_loss_image_div = exp_inner_prod(cross_image_batch, z_term_image) / B ** 2
                lambda_loss_text_div = exp_inner_prod(cross_text_batch, z_term_text) / B ** 2
                
            # Symmetric KL loss
            if "_sym" in self.lambda_loss_type:
                z_term_image = - (log_target_image - log_lambda_image) / target_image
                z_term_text = - (log_target_text - log_lambda_text) / target_text

                lambda_loss_image_div += exp_inner_prod(target_image, z_term_image) / B ** 2
                lambda_loss_text_div += exp_inner_prod(target_text, z_term_text) / B ** 2
                
            # JS divergence loss
            if "js" in self.lambda_loss_type:
                cross_exp = outputs["dist_matrix"].exp()
                cur_prob_image = cross_exp / lambda_image.unsqueeze(-1) #+ 1e-7
                cur_prob_text = cross_exp / lambda_text.unsqueeze(0) #+ 1e-7

                tar_prob_image = cross_exp / target_image.unsqueeze(-1) #+ 1e-7
                tar_prob_text = cross_exp / target_text.unsqueeze(0) #+ 1e-7

                mixed_prob_image = (cur_prob_image + tar_prob_image) / 2
                mixed_prob_text = (cur_prob_text + tar_prob_text) / 2
                
                lambda_loss_image_div = (F.kl_div(mixed_prob_image.log(), cur_prob_image) + F.kl_div(mixed_prob_image.log(), tar_prob_image)).mean() / 2
                lambda_loss_text_div = (F.kl_div(mixed_prob_text.log(), cur_prob_text) + F.kl_div(mixed_prob_text.log(), tar_prob_text)).mean() / 2
                
            # add regularization loss
            lambda_loss_image = self.div_reg_coef * lambda_loss_image + lambda_loss_image_div       
            lambda_loss_text = self.div_reg_coef * lambda_loss_text + lambda_loss_text_div
            
        total_loss = lambda_loss_image + lambda_loss_text
            
        return {
                "total_loss": total_loss,
                "image_lambda_loss": lambda_loss_image,
                "text_lambda_loss": lambda_loss_text,
                } if output_dict else total_loss
        
        
class LogTargetCollector(nn.Module):
    def __init__(
            self,
            rank=0,
            world_size=1,
            bidir=True,
            calculate_full=False,
            pos_coef=1.,
            lambda_fit_neg_only=False,
    ):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.bidir = bidir
        self.calculate_full = calculate_full
        self.pos_coef = pos_coef
        self.lambda_fit_neg_only = lambda_fit_neg_only
        
    @torch.no_grad()
    def _negative_exp(
            self,
            image_features, 
            text_features, 
            logit_scale, 
            logit_bias,
            is_same_batch,
        ): 
        with torch.no_grad():
            dist_matrix = get_dist_matrix(
                image_features, text_features, 
                scale=logit_scale, logit_bias=logit_bias)
  
            if is_same_batch:
                mask = ~torch.eye(dist_matrix.size(0), dtype=torch.bool, device=dist_matrix.device)
            else:
                mask = torch.ones_like(dist_matrix, dtype=torch.bool, device=dist_matrix.device) 
                
            dist_matrix = dist_matrix.masked_fill(~mask, float('-inf'))
            neg_exp = torch.logsumexp(dist_matrix, dim=-1).exp() # E_y, (B,)
        
        assert not neg_exp.requires_grad, "cross_batch should be detached"
        return neg_exp

    @torch.no_grad()
    def forward(self, 
                m0_features, m1_features, logit_scale, logit_bias
            ): 

        # * matched image and text features
        neg_exp = self._negative_exp(
            m0_features, m1_features, logit_scale, logit_bias, is_same_batch=True)

        if self.world_size > 1:
            if self.calculate_full:
                # exchange text features w/ neighbour world_size - 1 times
                right_rank = (self.rank + 1) % self.world_size
                left_rank = (self.rank - 1 + self.world_size) % self.world_size
                if self.bidir:
                    m1_feats_to_right = m1_feats_to_left = m1_features
                    num_bidir, remainder = divmod(self.world_size - 1, 2)

                    for i in range(num_bidir):
                        m1_feats_recv = neighbour_exchange_bidir(
                            left_rank,
                            right_rank,
                            m1_feats_to_left,
                            m1_feats_to_right,
                        )
                        for remote_m1_features in m1_feats_recv:
                            neg_exp += self._negative_exp(
                                m0_features, remote_m1_features, logit_scale, logit_bias,
                                is_same_batch=False)

                        m1_feats_to_left, m1_feats_to_right = m1_feats_recv

                    if remainder:
                        m1_feats_recv = neighbour_exchange(
                            left_rank, right_rank, m1_feats_to_right)
                        neg_exp += self._negative_exp(
                            m0_features, m1_feats_recv, logit_scale, logit_bias,
                            is_same_batch=False)

                else:
                    m1_feats_to_right = m1_features
                    for i in range(self.world_size - 1):
                        m1_feats_from_left = neighbour_exchange(
                            left_rank, right_rank, m1_feats_to_right)
                        neg_exp += self._negative_exp(
                            m0_features, m1_feats_from_left, logit_scale, logit_bias,
                            is_same_batch=False)
                        m1_feats_to_right = m1_feats_from_left
            else:
                neg_exp *= self.world_size
        
        if self.pos_coef > 0 and not self.lambda_fit_neg_only: 
            with torch.no_grad():
                pos_exp= (
                    exp_inner_prod(m0_features, m1_features, mean_exp=False) * logit_scale + logit_bias
                    ).exp().squeeze().detach()
                neg_exp += self.pos_coef * pos_exp
                    
        return neg_exp.log()

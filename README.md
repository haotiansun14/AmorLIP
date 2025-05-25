## AmorLIP: Efficient Language-Image Pretraining via Amortization

Implementation of the paper "AmorLIP: Efficient Language-Image Pretraining via Amortization"

### Abstract
Contrastive Language-Image Pretraining (CLIP) has demonstrated strong zero-shot performance across diverse downstream text-image tasks. Existing CLIP methods typically optimize a contrastive objective using negative samples drawn from each minibatch. To achieve robust representation learning, these methods require extremely large batch sizes and escalate computational demands to hundreds or even thousands of GPUs. Prior approaches to mitigate this issue often compromise downstream performance, prolong training duration, or face scalability challenges with very large datasets. To overcome these limitations, we propose AmorLIP, an efficient CLIP pretraining framework that amortizes expensive computations involved in contrastive learning through lightweight neural networks, which substantially improves training efficiency and performance. Leveraging insights from a spectral factorization of energy-based models, we introduce novel amortization objectives along with practical techniques to improve training stability. Extensive experiments across 38 downstream tasks demonstrate the superior zero-shot classification and retrieval capabilities of AmorLIP, consistently outperforming standard CLIP baselines with substantial relative improvements of up to 12.24%.

### Requirements
Please install the required packages using `pip install -r requirements.txt`.

### Experiments
The training script for large-scale setting with l2-log objective is provided as follows:
```bash
TRAIN_NUM_SAMPLES=8059642
VAL_NUM_SAMPLES=8678
BATCH_SIZE=2048

TRAIN_DATA=<PATH_TO_TRAIN_DATA>
VAL_DATA=<PATH_TO_VAL_DATA>
IMAGENET_VAL=<PATH_TO_IMAGENET_VAL>
DATACOMP_DATA=<PATH_TO_DATACOMP_DATA>


torchrun --nproc_per_node 1 -m open_clip_train.main --report-to none \
   --save-frequency=1 --zeroshot-frequency=1 --val-frequency=1 \
   --train-data="$TRAIN_DATA" --train-num-samples="$TRAIN_NUM_SAMPLES" \
   --val-data="$VAL_DATA" --val-num-samples="$VAL_NUM_SAMPLES" \
   --imagenet-val="$IMAGENET_VAL" \
   --dataset-type webdataset --warmup 10000 --epochs 33 --workers 6 \
   --datacomp_dir="$DATACOMP_DATA" \
   --model ViT-B-32 --seed 42  --precision amp \
   --batch-size "$BATCH_SIZE" --lr=4e-4 --grad-clip-norm 1.0 \
   --lambda-lr 0.001 --lambda-update-frequency "[3]" \
   --learn-logit-scale \
   --lambda_tolerance=5e-3 --calculate_full \
   --h_dim_factor 1 --lr_tau 1e-4 --scale_loss \
   --model_ema_beta 0.92 --beta_decay_epochs 16 --z_beta_max 0.8 --update_ema_mlps_every_n_steps 2 \
   --pos_coef 1e-3 --fit_w_prev --lambda_eps 0. \
   --reinit_lambda_every_n_epochs 1 \
   --rho_list="[8.5 -8.5 -8.5]" --lambda_loss_type "log_l2"
```

We provide amortization objectives as ```--lambda_loss_type``. The options are:
- `log_l2`: The default log-l2 objective.
- `kl_div_asym`: KL divergence.
- `js_div`: Jensen-Shannon divergence.


import argparse
import ast
from functools import partial

from open_clip.utils import NormalizeType

def get_default_params(model_name):
    # Params from paper (https://arxiv.org/pdf/2103.00020.pdf)
    model_name = model_name.lower()
    if "vit" in model_name:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.98, "eps": 1.0e-6}
    else:
        return {"lr": 5.0e-4, "beta1": 0.9, "beta2": 0.999, "eps": 1.0e-8}

def parse_list(value, data_type=int):
    """
    Parses a single value or a list-like string (e.g., '[500 -100 50 -50 20 -20]').

    Args:
        value (str): Input value from argparse, which can be a single number or a list string.
        data_type (type): Data type to enforce (int or float).

    Returns:
        list: A list containing the parsed values.

    Raises:
        argparse.ArgumentTypeError: If the value cannot be parsed.
    """
    try:
        # If brackets are present, parse as a list
        if "[" in value and "]" in value:
            value = value.replace(" ", ",")  # Convert space-separated lists to comma-separated
            parsed_list = ast.literal_eval(value)  # Safely evaluate as a Python list

            if not isinstance(parsed_list, list):
                # If parsing results in a single number, wrap it in a list
                return [data_type(parsed_list)]

            return [data_type(x) for x in parsed_list]  # Convert all elements to the desired type

        # If a single value is given without brackets, return as a list
        return [data_type(value)]

    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(
            f"Invalid value: {value}. Must be an {data_type.__name__} or a list in the format '[x y z]'."
        )

class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        kw = {}
        for value in values:
            key, value = value.split('=')
            try:
                kw[key] = ast.literal_eval(value)
            except ValueError:
                kw[key] = str(value)  # fallback to string (avoid need to escape on command line)
        setattr(namespace, self.dest, kw)


def parse_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-data",
        type=str,
        default=None,
        help="Path to file(s) with training data. When using webdataset, multiple datasources can be combined using the `::` separator.",
    )
    parser.add_argument(
        "--train-data-upsampling-factors",
        type=str,
        default=None,
        help=(
            "When using multiple data sources with webdataset and sampling with replacement, this can be used to upsample specific data sources. "
            "Similar to --train-data, this should be a string with as many numbers as there are data sources, separated by `::` (e.g. 1::2::0.5) "
            "By default, datapoints are sampled uniformly regardless of the dataset sizes."
        )
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        help="Path to file(s) with validation data",
    )
    parser.add_argument(
        "--train-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Required for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--val-num-samples",
        type=int,
        default=None,
        help="Number of samples in dataset. Useful for webdataset if not available in info file.",
    )
    parser.add_argument(
        "--dataset-type",
        choices=["webdataset", "csv", "synthetic", "auto"],
        default="auto",
        help="Which type of dataset to process."
    )
    parser.add_argument(
        "--dataset-resampled",
        default=False,
        action="store_true",
        help="Whether to use sampling with replacement for webdataset shard selection."
    )
    parser.add_argument(
        "--csv-separator",
        type=str,
        default="\t",
        help="For csv-like datasets, which separator to use."
    )
    parser.add_argument(
        "--csv-img-key",
        type=str,
        default="filepath",
        help="For csv-like datasets, the name of the key for the image paths."
    )
    parser.add_argument(
        "--csv-caption-key",
        type=str,
        default="title",
        help="For csv-like datasets, the name of the key for the captions."
    )
    parser.add_argument(
        "--imagenet-val",
        type=str,
        default=None,
        help="Path to imagenet val set for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--imagenet-v2",
        type=str,
        default=None,
        help="Path to imagenet v2 for conducting zero shot evaluation.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override system default cache path for model & tokenizer file downloads.",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default="./logs",
        help="Where to store tensorboard logs. Use None to avoid storing logs.",
    )
    parser.add_argument(
        "--log-local",
        action="store_true",
        default=False,
        help="log files on local master, otherwise global master only.",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Optional identifier for the experiment when storing logs. Otherwise use current time.",
    )
    parser.add_argument(
        "--workers", type=int, default=4, help="Number of dataloader workers per GPU."
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size per GPU."
    )
    parser.add_argument(
        "--epochs", type=int, default=32, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--epochs-cooldown", type=int, default=None,
        help="When scheduler w/ cooldown used, perform cooldown from total_epochs - cooldown_epochs onwards."
    )
    parser.add_argument("--lr", type=float, default=None, help="Learning rate.")
    parser.add_argument("--beta1", type=float, default=None, help="Adam beta 1.")
    parser.add_argument("--beta2", type=float, default=None, help="Adam beta 2.")
    parser.add_argument("--eps", type=float, default=None, help="Adam epsilon.")
    parser.add_argument("--wd", type=float, default=0.2, help="Weight decay.")
    parser.add_argument(
        "--warmup", type=int, default=10000, help="Number of steps to warmup for."
    )
    parser.add_argument(
        "--use-bn-sync",
        default=False,
        action="store_true",
        help="Whether to use batch norm sync.")
    parser.add_argument(
        "--skip-scheduler",
        action="store_true",
        default=False,
        help="Use this flag to skip the learning rate decay.",
    )
    parser.add_argument(
        "--lr-scheduler",
        type=str,
        default='cosine',
        help="LR scheduler. One of: 'cosine', 'const' (constant), 'const-cooldown' (constant w/ cooldown). Default: cosine",
    )
    parser.add_argument(
        "--lr-cooldown-end", type=float, default=0.0,
        help="End learning rate for cooldown schedule. Default: 0"
    )
    parser.add_argument(
        "--lr-cooldown-power", type=float, default=1.0,
        help="Power for polynomial cooldown schedule. Default: 1.0 (linear decay)"
    )
    parser.add_argument(
        "--save-frequency", type=int, default=1, help="How often to save checkpoints."
    )
    parser.add_argument(
        "--save-most-recent",
        action="store_true",
        default=False,
        help="Always save the most recent model trained to epoch_latest.pt.",
    )
    parser.add_argument(
        "--zeroshot-frequency", type=int, default=2, help="How often to run zero shot."
    )
    parser.add_argument(
        "--val-frequency", type=int, default=1, help="How often to run evaluation with val data."
    )
    parser.add_argument(
        "--resume",
        default=None,
        type=str,
        help="path to latest checkpoint (default: none)",
    )
    parser.add_argument(
        "--precision",
        choices=["amp", "amp_bf16", "amp_bfloat16", "bf16", "fp16", "pure_bf16", "pure_fp16", "fp32"],
        default="amp",
        help="Floating point precision."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="RN50",
        help="Name of the vision backbone to use.",
    )
    parser.add_argument(
        "--pretrained",
        default='',
        type=str,
        help="Use a pretrained CLIP model weights with the specified tag or file path.",
    )
    parser.add_argument(
        "--pretrained-image",
        default=False,
        action='store_true',
        help="Load imagenet pretrained weights for image tower backbone if available.",
    )
    parser.add_argument(
        "--lock-image",
        default=False,
        action='store_true',
        help="Lock full image tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-image-unlocked-groups",
        type=int,
        default=0,
        help="Leave last n image tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-image-freeze-bn-stats",
        default=False,
        action='store_true',
        help="Freeze BatchNorm running stats in image tower for any locked layers.",
    )
    parser.add_argument(
        '--image-mean', type=float, nargs='+', default=None, metavar='MEAN',
        help='Override default image mean value of dataset')
    parser.add_argument(
        '--image-std', type=float, nargs='+', default=None, metavar='STD',
        help='Override default image std deviation of of dataset')
    parser.add_argument(
        '--image-interpolation',
        default=None, type=str, choices=['bicubic', 'bilinear', 'random'],
        help="Override default image resize interpolation"
    )
    parser.add_argument(
        '--image-resize-mode',
        default=None, type=str, choices=['shortest', 'longest', 'squash'],
        help="Override default image resize (& crop) mode during inference"
    )
    parser.add_argument('--aug-cfg', nargs='*', default={}, action=ParseKwargs)
    parser.add_argument(
        "--grad-checkpointing",
        default=False,
        action='store_true',
        help="Enable gradient checkpointing.",
    )
    parser.add_argument(
        "--local-loss",
        default=False,
        action="store_true",
        help="calculate loss w/ local features @ global (instead of realizing full global @ global matrix)"
    )
    parser.add_argument(
        "--gather-with-grad",
        default=False,
        action="store_true",
        help="enable full distributed gradient for feature gather"
    )
    parser.add_argument(
        '--force-image-size', type=int, nargs='+', default=None,
        help='Override default image size'
    )
    parser.add_argument(
        "--force-quick-gelu",
        default=False,
        action='store_true',
        help="Force use of QuickGELU activation for non-OpenAI transformer models.",
    )
    parser.add_argument(
        "--force-patch-dropout",
        default=None,
        type=float,
        help="Override the patch dropout during training, for fine tuning with no dropout near the end as in the paper",
    )
    parser.add_argument(
        "--force-custom-text",
        default=False,
        action='store_true',
        help="Force use of CustomTextCLIP model (separate text-tower).",
    )
    parser.add_argument(
        "--torchscript",
        default=False,
        action='store_true',
        help="torch.jit.script the model, also uses jit version of OpenAI models if pretrained=='openai'",
    )
    parser.add_argument(
        "--torchcompile",
        default=False,
        action='store_true',
        help="torch.compile() the model, requires pytorch 2.0 or later.",
    )
    parser.add_argument(
        "--trace",
        default=False,
        action='store_true',
        help="torch.jit.trace the model for inference / eval only",
    )
    parser.add_argument(
        "--accum-freq", type=int, default=1, help="Update the model every --acum-freq steps."
    )
    parser.add_argument(
        "--device", default="cuda", type=str, help="Accelerator to use."
    )
    # arguments for distributed training
    parser.add_argument(
        "--dist-url",
        default=None,
        type=str,
        help="url used to set up distributed training",
    )
    parser.add_argument(
        "--dist-backend",
        default=None,
        type=str,
        help="distributed backend. \"nccl\" for GPU, \"hccl\" for Ascend NPU"
    )
    parser.add_argument(
        "--report-to",
        default='',
        type=str,
        help="Options are ['wandb', 'tensorboard', 'wandb,tensorboard']"
    )
    parser.add_argument(
        "--wandb-notes",
        default='',
        type=str,
        help="Notes if logging with wandb"
    )
    parser.add_argument(
        "--wandb-project-name",
        type=str,
        default='open-clip',
        help="Name of the project if logging with wandb.",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="If true, more information is logged."
    )
    parser.add_argument(
        "--copy-codebase",
        default=False,
        action="store_true",
        help="If true, we copy the entire base on the log directory, and execute from there."
    )
    parser.add_argument(
        "--horovod",
        default=False,
        action="store_true",
        help="Use horovod for distributed training."
    )
    parser.add_argument(
        "--ddp-static-graph",
        default=False,
        action='store_true',
        help="Enable static graph optimization for DDP in PyTorch >= 1.11.",
    )
    parser.add_argument(
        "--no-set-device-rank",
        default=False,
        action="store_true",
        help="Don't set device index from local rank (when CUDA_VISIBLE_DEVICES restricted to one per proc)."
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="Default random seed."
    )
    parser.add_argument(
        "--grad-clip-norm", type=float, default=None, help="Gradient clip."
    )
    parser.add_argument(
        "--lock-text",
        default=False,
        action='store_true',
        help="Lock full text tower by disabling gradients.",
    )
    parser.add_argument(
        "--lock-text-unlocked-layers",
        type=int,
        default=0,
        help="Leave last n text tower layer groups unlocked.",
    )
    parser.add_argument(
        "--lock-text-freeze-layer-norm",
        default=False,
        action='store_true',
        help="Freeze LayerNorm running stats in text tower for any locked layers.",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=100,
        help="Log every n steps to tensorboard/console/wandb.",
    )
    parser.add_argument(
        "--coca-caption-loss-weight",
        type=float,
        default=2.0,
        help="Weight assigned to caption loss in CoCa."
    )
    parser.add_argument(
        "--coca-contrastive-loss-weight",
        type=float,
        default=1.0,
        help="Weight assigned to contrastive loss when training CoCa."
    )
    parser.add_argument(
        "--remote-sync",
        type=str,
        default=None,
        help="Optinoally sync with a remote path specified by this arg",
    )
    parser.add_argument(
        "--remote-sync-frequency",
        type=int,
        default=300,
        help="How frequently to sync to a remote directly if --remote-sync is not None.",
    )
    parser.add_argument(
        "--remote-sync-protocol",
        choices=["s3", "fsspec"],
        default="s3",
        help="How to do the remote sync backup if --remote-sync is not None.",
    )
    parser.add_argument(
        "--delete-previous-checkpoint",
        default=False,
        action="store_true",
        help="If true, delete previous checkpoint after storing a new one."
    )
    parser.add_argument(
        "--distill-model",
        default=None,
        help='Which model arch to distill from, if any.'
    )
    parser.add_argument(
        "--distill-pretrained",
        default=None,
        help='Which pre-trained weights to distill from, if any.'
    )
    parser.add_argument(
        "--use-bnb-linear",
        default=None,
        help='Replace the network linear layers from the bitsandbytes library. '
        'Allows int8 training/inference, etc.'
    )
    parser.add_argument(
        "--siglip",
        default=False,
        action="store_true",
        help='Use SigLip (sigmoid) loss.'
    )

    # * Added
    # Lambda params
    parser.add_argument(
        "--model_beta_init",
        default=0.,
        type=float,
        help="Weight for EMA update of lambda (outer loop)"
    )
    
    parser.add_argument(
        "--lambda-lr",
        default=0.001,
        type=float,
        help="Learning rate for lambda MLPs"
    )
    parser.add_argument(
        "--denormalize-features",
        default=False,
        action='store_true',
        help="Whether to denormalize features"
    )
    parser.add_argument(
        "--normalize-type",
        default=NormalizeType.L2,
        type=NormalizeType,
        choices=list(NormalizeType),
        help="Type of normalization for encoder features"
    )
    parser.add_argument(
        "--norm-cap",
        default=1.0,
        type=float,
        help="Normalization cap for encoder features"
    )

    parser.add_argument(
        "--init-logit-scale",
        default=None,
        type=float,
        help="Initial value for logit scale for encoder features"
    )
    parser.add_argument(
        "--init-logit-bias",
        default=None, 
        type=float,
        help="Bias value for logit scale for encoder features"
    )
    parser.add_argument(
        "--init-lambda",
        default=1.0,
        type=float,
        help="Initial value for lambda"
    )
    parser.add_argument(
        "--logit-scale-clamp",
        default=100, 
        type=float,
        help="Clamp value for logit scale for encoder features"
    )
    parser.add_argument(
        "--learn-logit-scale",
        default=False,
        action='store_true',
        help="Whether to use logit scale for encoder features"
    )
    parser.add_argument(
        "--learn-logit-bias",
        default=False,
        action='store_true',
        help="Whether to use logit bias for encoder features"
    )
    
    # * rec
    parser.add_argument(
        "--lambda_tolerance",
        default=1e-4,
        type=float,
        help="Tolerance for lambda update"
    )
    parser.add_argument(
        "--scale_loss",
        default=False,
        action='store_true',
        help="Whether to scale loss by the number of samples"
    )
    parser.add_argument(
        "--pos_coef",
        default=1.,
        type=float,
        help="Positive samples coefficient"
    )
    parser.add_argument(
        "--lambda_eps",
        default=0.,
        type=float,
        help="Epsilon for numerical stability"
    )
    parser.add_argument(
        "--calculate_full",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        "--update_lambda_every_n_steps",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--update_ema_mlps_every_n_steps",
        default=1,
        type=int,
        help="Update EMA MLPs every n steps"
    )
    parser.add_argument(
        "--model_ema_beta",
        default=0.9997,
        type=float,
        help="Max value for model_beta"
    )
    parser.add_argument(
        "--beta_decay_epochs",
        default=15,
        type=int,
        help="Number of epochs for decaying z_beta"
    )
    parser.add_argument(
        "--z_beta_max",
        default=0.8,
        type=float,
        help="Max value for z_beta"
    )

    parser.add_argument(
        "--h_dim_factor",
        default=1,
        type=float,
        help="Factor to multiply h_dim"
    )
    parser.add_argument(
        "--lr_tau", 
        type=float, 
        default=-1.0,
        help="Learning rate of the temperature parameter. If < 0, will be set to lr of the model."
    )
    
    parser.add_argument(
        "--fit_w_prev",
        default=False,
        action="store_true",
        help="Whether to use weighting of previous samples"
    )
    parser.add_argument(
        "--lambda-update-frequency",
        type=lambda x: parse_list(x, int),  # Parse as int or list of ints
        help="An integer or list of update frequencies, e.g., '[500 100 50 50]'."
    )

    parser.add_argument(
        "--rho_list",
        type=lambda x: parse_list(x, float),  # Parse as float or list of floats
        help="A float or list of rho values, e.g., '[0.9 0.8 0.7]'."
    )
    parser.add_argument(
        "--reinit_lambda_every_n_epochs",
        default=-1,
        type=int,
        help="Number of epochs for reinitializing lambda"
    )
    parser.add_argument(
        "--lambda_loss_type",
        default="log_l2",
        choices=["log_l2", "kl_div_asym", "kl_div_sym", "js_div"],
        help="Type of lambda loss"
    )
    parser.add_argument(
        "--adaptive_tau",
        default=False,
        action='store_true',
        help="Whether to use adaptive tau for the temperature parameter"
    )
    parser.add_argument(
        "--datacomp_dir",
        default=None,
        type=str,
        help="Directory path to the dataset"
    )
    parser.add_argument(
        "--lambda_fit_neg_only",
        default=False,
        action='store_true',
        help="Whether to fit lambda only on negative samples"
    )
    parser.add_argument(
        "--div_reg_coef",
        default=0.1,
        type=float,
        help="Coefficient for divergence regularization"
    )

    parser.add_argument(
        "--note",
        default="",
        type=str,
        help="Additional note for the experiment"
    )
    args = parser.parse_args(args)

    # If some params are not passed, we use the default values based on model name.
    default_params = get_default_params(args.model)
    for name, val in default_params.items():
        if getattr(args, name) is None:
            setattr(args, name, val)

    return args

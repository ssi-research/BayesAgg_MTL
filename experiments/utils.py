import argparse
import logging
import random
from pathlib import Path
import numpy as np
import torch


def str_to_list(string):
    return [float(s) for s in string.split(",")]


def str_or_float(value):
    try:
        return float(value)
    except:
        return value


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


# todo: refactor common parser
common_parser = argparse.ArgumentParser(add_help=False)
common_parser.add_argument("--data-path", type=Path, help="path to data")
common_parser.add_argument("--n-epochs", type=int, default=100)
common_parser.add_argument("--batch-size", type=int, default=32, help="batch size")
common_parser.add_argument(
    "--n-workers",
    type=int,
    default=4,
    help="number of workers",
)
common_parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
common_parser.add_argument("--wd", type=float, default=0.0, help="weight decay")
common_parser.add_argument(
    "--eval-every",
    type=int,
    default=1,
    help="number of epochs between evaluations"
)
common_parser.add_argument("--gpu", type=int, default=0, help="gpu device ID")
common_parser.add_argument("--seed", type=int, default=42, help="seed value")

common_parser.add_argument(
    "--gamma",
    type=float,
    default=1.,
    help="standard normal covariance scalar"
)
common_parser.add_argument(
    "--num-mc-samples",
    type=int,
    default=1024,
    help="number of MC samples to estimate mean and cov"
)
common_parser.add_argument(
        "--sqrt-power",
        type=float,
        default=0.5,
        help="inner optimizer lr",
)
common_parser.add_argument(
        "--sqrt-power-cls",
        type=float,
        default=0.5,
        help="inner optimizer lr",
)
common_parser.add_argument(
    "--ls-epochs",
    default=1,
    type=int,
    help="num pre-train epochs of linear scalarization"
)


def set_logger():
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.INFO,
    )


def set_seed(seed, ):
    """for reproducibility
    :param seed:
    :return:
    """
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    # torch.use_deterministic_algorithms(True)
    # os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"


def get_device(no_cuda=False, gpus="0"):
    return torch.device(
        f"cuda:{gpus}" if torch.cuda.is_available() and not no_cuda else "cpu"
    )
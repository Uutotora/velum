import os
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def set_env(
    deterministic=True, seed=0, allow_tf32_on_cudnn=True, allow_tf32_on_matmul=True
):
    if deterministic:
        torch.set_num_threads(1)
        random.seed(seed)
        np.random.seed(seed)
        cv2.setRNGSeed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)

        if torch.cuda.is_available():
            cudnn.benchmark = False
            cudnn.deterministic = True
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        # MPS does not support all deterministic ops, skip on Apple Silicon
        if not torch.backends.mps.is_available():
            torch.use_deterministic_algorithms(True)
    else:
        if torch.cuda.is_available():
            cudnn.benchmark = True
        torch.use_deterministic_algorithms(False)

    # https://pytorch.org/docs/stable/notes/cuda.html#tf32-on-ampere
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32_on_matmul
    torch.backends.cudnn.allow_tf32 = allow_tf32_on_cudnn

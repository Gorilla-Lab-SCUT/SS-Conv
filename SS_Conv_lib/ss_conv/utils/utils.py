# Modified From https://github.com/tscohen/se3cnn 
# Written by Jiehong Lin and Hongyang Li

import torch

def Rs2dim(Rs):
    dim = 0 
    for m,l in Rs:
        dim += (l*2 + 1)*m
    return dim

def calculate_fan_in(channel_in, kernel_size):
    # channel_in: int
    # chennel_out: int
    # kenerl_size: torch.tensor
    fan_in = channel_in * kernel_size.prod()

    return fan_in

class torch_default_dtype:

    def __init__(self, dtype):
        self.saved_dtype = None
        self.dtype = dtype

    def __enter__(self):
        self.saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(self.dtype)

    def __exit__(self, exc_type, exc_value, traceback):
        torch.set_default_dtype(self.saved_dtype)


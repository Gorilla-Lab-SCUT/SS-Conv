# ----------------------------------------------------------------
# Sparse Steerable Convolutions

# Dropout for sparse tensors
# Modified from https://github.com/tscohen/se3cnn 
# by Jiehong Lin and Hongyang Li
# ----------------------------------------------------------------


import torch
import torch.nn as nn


class Dropout(nn.Module):
    def __init__(self, Rs, p=0.5):
        super().__init__()
        self.Rs = [(m, l) for m, l in Rs if m > 0]
        self.p = p

    def __repr__(self):
        return "{} (p={})".format(
            self.__class__.__name__,
            self.p)

    def forward(self, input):
        if not self.training:
            return input

        x = input.features
        noises = []
        for m, l in self.Rs:
            d = 2*l+1
            noise = x.new_empty(x.size(0), m)

            if self.p == 1:
                noise.fill_(0)
            elif self.p == 0:
                noise.fill_(1)
            else:
                noise.bernoulli_(1 - self.p).div_(1 - self.p)

            noise = noise.unsqueeze(2).expand(-1, -1, d).contiguous().view(x.size(0), m*d)
            noises.append(noise)
        noise = torch.cat(noises, dim=1)
        input.features = x * noise
        return input
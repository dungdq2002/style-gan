from datetime import datetime
import functools
import os
import time
import shutil

import numpy as np
import torch as th
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as func
from torch.utils.serialization import load_lua


class PatchDiscriminator(nn.Module):
    def __init__(
        self,
        scn,
        ccn,
        input_nc=3,
        ndf=64,
        n_layers=3,
        norm_layer=nn.BatchNorm2d,
        use_sigmoid=False,
        use_proj=True,
    ):
        super(PatchDiscriminator, self).__init__()
        self.use_proj = use_proj
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True),
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=kw,
                    stride=2,
                    padding=padw,
                    bias=use_bias,
                ),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=kw,
                stride=1,
                padding=padw,
                bias=use_bias,
            ),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True),
        ]

        self.model = nn.Sequential(*sequence)

        sequence = []
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=3, stride=1, padding=padw)]
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        self.bclass = nn.Sequential(*sequence)
        if use_proj:
            self.sproj = nn.Embedding(scn, ndf * nf_mult)
            self.sproj.weight.data.fill_(0)  # init

        sequence = [nn.Linear(512 * 3 * 3, scn)]
        self.sclass = nn.Sequential(*sequence)

        self.ce = nn.CrossEntropyLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def get_adv_loss(self, indata, label=True):
        if use_real:
            target = th.ones_like(indata)
        else:
            target = th.zeros_like(indata)
        return self.bce(indata, target)

    def forward(self, indata, slabel=None, label=True):
        """
        Forward pass for the discriminator model.

        Parameters:
        indata (torch.Tensor): Input data for the discriminator.
        slabel (torch.Tensor, optional): Style label for the projection discriminator. Default is None.
        label (bool, optional): Whether the input is real or fake. Default is True.

        Returns:
        tuple: Returns a tuple containing:
            - bc_out (torch.Tensor): Output from the binary classifier, indicating whether the input is real or fake.
            - sc_out (torch.Tensor): Output from the style classifier, indicating the style of the input image.

        If use_proj is True and slabel is not None, the method performs projection discrimination.
        The feature map (ftr) is multiplied with the style projection (semb) and the mean of the result is added to bc_out.
        """
        ftr = self.model(indata)
        bc_out = self.bclass(ftr)  # True/False
        if self.use_proj:  # projection discriminator
            if slabel is not None:
                semb = self.sproj(slabel)
                sftr = ftr * semb.view(semb.size(0), semb.size(1), 1, 1)
                bc_out += th.mean(sftr, dim=1, keepdim=True)

        ftr = func.avg_pool2d(ftr, 9)  # average pooling
        ftr = ftr.view(ftr.size(0), -1)  # reshape
        sc_out = self.sclass(ftr)  # Style classificatin

        loss_cls = self.ce(sc_out, slabel)
        loss_adv = self.get_adv_loss(bc_out, label)

        return loss_cls, loss_adv

    def load_model(self, load_model):
        checkpoint = th.load(load_model)
        self.load_state_dict(checkpoint["disc"])
        print("discriminator loaded from:", load_model)

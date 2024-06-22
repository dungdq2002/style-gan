import torch
from torch import nn

from . import networks

# import blocks
# import time

# from vgg19 import vgg, VGG_loss
# from networks import EFDM_loss


class AesFA(nn.Module):
    def __init__(self, config):
        super(AesFA, self).__init__()

        self.config = config
        # self.device = self.config.device

        self.netE = networks.define_network(
            net_type="Encoder", config=config
        )  # Content Encoder
        self.netS = networks.define_network(
            net_type="Encoder", config=config
        )  # Style Encoder
        # self.netG = networks.define_network(net_type="Generator", config=config)

        # self.vgg_loss = VGG_loss(config, vgg)
        # self.efdm_loss = EFDM_loss()

        # self.optimizer_E = torch.optim.Adam(
        #     self.netE.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99)
        # )
        # self.optimizer_S = torch.optim.Adam(
        #     self.netS.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99)
        # )
        # self.optimizer_G = torch.optim.Adam(
        #     self.netG.parameters(), lr=self.config.lr, betas=(self.config.beta1, 0.99)
        # )

    def forward(self, content, style):
        self.real_A = content
        self.real_B = style

        self.content_A, _, _ = self.netE(self.real_A)
        _, self.style_B, self.content_B_feat = self.netS(self.real_B)
        self.style_B_feat = self.content_B_feat.copy()
        self.style_B_feat.append(self.style_B)

        return self.content_A, self.style_B

        # self.trs_AtoB, self.trs_AtoB_high, self.trs_AtoB_low = self.netG(
        #     self.content_A, self.style_B
        # )

        # self.trs_AtoB_content, _, self.content_trs_AtoB_feat = self.netE(self.trs_AtoB)
        # _, self.trs_AtoB_style, self.style_trs_AtoB_feat = self.netS(self.trs_AtoB)
        # self.style_trs_AtoB_feat.append(self.trs_AtoB_style)

        # return self.trs_AtoB, self.content_trs_AtoB_feat, self.style_trs_AtoB_feat

    def set_requires_grad(self, nets, requires_grad=False):
        if not isinstance(nets, list):
            nets = [nets]
        for net in nets:
            if net is not None:
                for param in net.parameters():
                    param.requires_grad = requires_grad

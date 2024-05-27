import numpy as np
import scipy.stats as stats
import torch
import torch.nn.functional as F
from torch import nn

from utils.misc import (  # accuracy,; get_world_size,; interpolate,; is_dist_avail_and_initialized,
    NestedTensor,
    nested_tensor_from_tensor_list,
)
from utils.support_functions import calc_mean_std, normal, normal_style


class StyTrans(nn.Module):
    """This is the style transform transformer module"""

    def __init__(self, encoder, decoder, patch_emb, transformer):
        super().__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.enc_5 = nn.Sequential(*enc_layers[31:44])  # relu4_1 -> relu5_1

        for name in ["enc_1", "enc_2", "enc_3", "enc_4", "enc_5"]:
            for param in getattr(self, name).parameters():
                param.requires_grad = False

        self.mse_loss = nn.MSELoss()
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.decode = decoder
        self.embedding = patch_emb

    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(5):
            func = getattr(self, "enc_{:d}".format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    def calc_content_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        return self.mse_loss(input, target)

    def calc_style_loss(self, input, target):
        assert input.size() == target.size()
        assert target.requires_grad is False
        input_mean, input_std = calc_mean_std(input)
        target_mean, target_std = calc_mean_std(target)
        return self.mse_loss(input_mean, target_mean) + self.mse_loss(
            input_std, target_std
        )

    def forward(self, samples_c: NestedTensor, samples_s: NestedTensor):
        """The forward expects a NestedTensor, which consists of:
        - samples.tensor: batched images, of shape [batch_size x 3 x H x W]
        - samples.mask: a binary mask of shape [batch_size x H x W], containing 1 on padded pixels

        """
        content_input = samples_c
        style_input = samples_s
        if isinstance(samples_c, (list, torch.Tensor)):
            samples_c = nested_tensor_from_tensor_list(
                samples_c
            )  # support different-sized images padding is used for mask [tensor, mask]
        if isinstance(samples_s, (list, torch.Tensor)):
            samples_s = nested_tensor_from_tensor_list(samples_s)

        # ### features used to calcate loss
        content_feats = self.encode_with_intermediate(samples_c.tensors)
        style_feats = self.encode_with_intermediate(samples_s.tensors)

        ### Linear projection
        style = self.embedding(samples_s.tensors)
        content = self.embedding(samples_c.tensors)

        # postional embedding is calculated in transformer.py
        pos_s = None
        pos_c = None

        mask = None
        hs = self.transformer(style, mask, content, pos_c, pos_s)
        Ics = self.decode(hs)

        Ics_feats = self.encode_with_intermediate(Ics)
        loss_c = self.calc_content_loss(
            normal(Ics_feats[-1]), normal(content_feats[-1])
        ) + self.calc_content_loss(normal(Ics_feats[-2]), normal(content_feats[-2]))
        # Style loss
        loss_s = self.calc_style_loss(Ics_feats[0], style_feats[0])
        for i in range(1, 5):
            loss_s += self.calc_style_loss(Ics_feats[i], style_feats[i])

        return Ics, loss_c, loss_s
        # Icc = self.decode(self.transformer(content, mask, content, pos_c, pos_c))
        # Iss = self.decode(self.transformer(style, mask, style, pos_s, pos_s))

        # # Identity losses lambda 1
        # loss_lambda1 = self.calc_content_loss(
        #     Icc, content_input
        # ) + self.calc_content_loss(Iss, style_input)

        # # Identity losses lambda 2
        # Icc_feats = self.encode_with_intermediate(Icc)
        # Iss_feats = self.encode_with_intermediate(Iss)
        # loss_lambda2 = self.calc_content_loss(
        #     Icc_feats[0], content_feats[0]
        # ) + self.calc_content_loss(Iss_feats[0], style_feats[0])
        # for i in range(1, 5):
        #     loss_lambda2 += self.calc_content_loss(
        #         Icc_feats[i], content_feats[i]
        #     ) + self.calc_content_loss(Iss_feats[i], style_feats[i])
        # # Please select and comment out one of the following two sentences
        # return Ics, loss_c, loss_s, loss_lambda1, loss_lambda2  # train
        # return Ics    #test

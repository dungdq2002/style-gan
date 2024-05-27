import torch

from torch import nn

from .discriminator.discriminator import PatchDiscriminator
from .generator.StyTR import StyTrans
from .generator.transformer import Transformer
from utils.support_functions import get_vgg, get_decoder, PatchEmbed


class Network(nn.Module):
    """Our model

    Args:
        nn (_type_): _description_
    """

    def __init__(self, args):
        super(Network, self).__init__()
        # vgg load
        self.vgg = get_vgg()
        # self.vgg.load_state_dict(torch.load(args.vgg))
        # self.vgg = nn.Sequential(*list(vgg.children())[:44])

        self.decoder = get_decoder()
        self.patch_embed = PatchEmbed()
        self.transformer = Transformer()

        self.generator = StyTrans(
            encoder=self.vgg,
            decoder=self.decoder,
            patch_emb=self.patch_embed,
            transformer=self.transformer,
        )

        self.discriminator = PatchDiscriminator(scn=args.scn)

    def forward(self, contents, styles, label=True):
        # unpatch style to img and slabel
        style_imgs, slabels = styles

        imgs, loss_c, loss_s = self.generator(contents, style_imgs)
        loss_cls, loss_adv = self.discriminator(imgs, slabels, label)

        return imgs, loss_cls, loss_adv, loss_c, loss_s

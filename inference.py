from utils.support_functions import get_vgg, get_decoder, PatchEmbed
from models.generator.transformer import Transformer
import models.generator.StyTR as StyTR
import torch

vgg = get_vgg()
vgg.load_state_dict(torch.load("experiment/models/vgg_normalised.pth"))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = torch.load("experiment/models/checkpoint_399000.pth")

generator_state_dict = {k: v for k, v in model.items() if "generator" in k}

decoder = get_decoder()
decoder_state_dict = {
    ".".join(k.split(".")[2:]): v
    for k, v in generator_state_dict.items()
    if "decode" in k.split(".")[1]
}
decoder.load_state_dict(decoder_state_dict)

transformer = Transformer()
transformer_state_dict = {
    ".".join(k.split(".")[2:]): v
    for k, v in generator_state_dict.items()
    if "transformer" in k.split(".")[1]
}
transformer.load_state_dict(transformer_state_dict)

emb = PatchEmbed()
emb_state_dict = {
    ".".join(k.split(".")[2:]): v
    for k, v in generator_state_dict.items()
    if "embedding" in k.split(".")[1]
}
emb.load_state_dict(emb_state_dict)

gen = StyTR.StyTrans(vgg, decoder, emb, transformer)

# load image to tensor
from PIL import Image
from torchvision.transforms import ToTensor
from torchvision.utils import save_image
from torchvision import transforms
import numpy as np


def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.CenterCrop(size))
    transform_list.append(transforms.ToTensor())
    transform = transforms.Compose(transform_list)
    return transform


content_size = 512
style_size = 512
crop = "store_true"

content_tf = test_transform(content_size, crop)
style_tf = test_transform(style_size, crop)

content_image = content_tf(Image.open("experiment/content/mountain.jpg").convert("RGB"))
style_image = style_tf(Image.open("experiment/style/starry_night.jpg").convert("RGB"))

content_image = content_image.to(device).unsqueeze(0)
style_image = style_image.to(device).unsqueeze(0)

# inference
gen.eval()
gen.to(device)
with torch.no_grad():
    output = gen(content_image, style_image)
    # save image
    save_image(output, "output.png")

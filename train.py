import os
import argparse
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torchvision.utils import save_image

from models.network import Network
from models.generator.sampler import InfiniteSamplerWrapper
from utils import folder
from utils import support_functions as sf
from utils.config import Config
from utils.img_transform import train_transform

from tensorboardX import SummaryWriter


def parser():
    parser = argparse.ArgumentParser(description="Training configuration")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="path to the configuration file (default: config.yaml)",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parser()

    config = Config(args.config)
    train_config = config.train

    if not os.path.exists(train_config.save_dir):
        os.makedirs(train_config.save_dir)

    if not os.path.exists(train_config.log_dir):
        os.makedirs(train_config.log_dir)

    if not os.path.exists(train_config.sample_output_dir):
        os.makedirs(train_config.sample_output_dir)

    # device config
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")
    # device = torch.device("cpu")
    train_tf = train_transform()
    # content img preparation
    content_dataset = folder.FolderDataset(
        root=train_config.content_dir, transform=train_tf
    )

    content_iter = iter(
        data.DataLoader(
            dataset=content_dataset,
            sampler=InfiniteSamplerWrapper(content_dataset),
            batch_size=train_config.batch_size,
            # shuffle=True,
            num_workers=train_config.num_workers if USE_CUDA else 0,
        )
    )

    # style img preparation
    style_dataset = folder.ImageFolder(root=train_config.style_dir, transform=train_tf)

    style_iter = iter(
        data.DataLoader(
            dataset=style_dataset,
            sampler=InfiniteSamplerWrapper(style_dataset),
            batch_size=train_config.batch_size,
            # shuffle=True,
            num_workers=train_config.num_workers if USE_CUDA else 0,
        )
    )

    number_of_styles = len(style_dataset.classes)

    # define network and stuff right here
    with torch.no_grad():
        network = Network(train_config, number_of_styles)
    network.train()
    network.to(device)

    print("== Device in use: ", device)

    # optimizer for generator
    optimizer = optim.Adam(
        [
            {"params": network.generator.transformer.parameters()},
            {"params": network.generator.decode.parameters()},
            {"params": network.generator.embedding.parameters()},
        ],
        lr=train_config.lr,
    )

    # optimier for discriminator
    doptimizer = optim.Adam(
        [{"params": network.discriminator.parameters()}],
        lr=train_config.d_lr,
    )

    writer = SummaryWriter(log_dir=train_config.log_dir)

    # training loop
    for it in range(train_config.max_iterations):
        if it < 1e4:
            sf.warmup_learning_rate(optimizer, train_config.lr, it)
        else:
            sf.adjust_learning_rate(optimizer, train_config.lr_decay, it)
        # implement adjust learning rate here if needed

        content_images = next(content_iter).to(device)
        style_images, slabels = next(style_iter)
        style_images = style_images.to(device)
        slabels = slabels.to(device)

        # ratio to flip label in GAN
        # ratio_thr = train_config.gan_ratio / max(it / train_config.gr_freq, 1.0)

        # if random.random() > ratio_thr:
        #     use_real = True
        # else:
        #     use_real = False

        # model output

        # ================ Train the generator (StyTr2) ================ #
        # network.generator.unfreeze()
        imgs, loss_c, loss_s, loss_id1, loss_id2 = network.generator(
            content_images, style_images
        )

        network.discriminator.requires_grad_(False)
        loss_cls, loss_adv = network.discriminator(imgs, slabels, True)

        loss_c *= train_config.content_weight
        loss_s *= train_config.style_weight

        optimizer.zero_grad()
        gen_loss = (
            loss_cls * train_config.cls_weight
            + loss_adv * train_config.bin_weight
            + loss_c
            + loss_s
            + loss_id1 * train_config.id_1_weight
            + loss_id2 * train_config.id_2_weight
        )  # compute with loss_cls, loss_adv, loss_c, loss_s
        gen_loss.sum().backward()
        optimizer.step()

        writer.add_scalar("loss/gen/total", gen_loss, it)
        writer.add_scalar("loss/gen/cls", loss_cls, it)
        writer.add_scalar("loss/gen/adv", loss_adv, it)
        writer.add_scalar("loss/gen/content", loss_c, it)
        writer.add_scalar("loss/gen/style", loss_s, it)
        writer.add_scalar("loss/gen/id1", loss_id1, it)
        writer.add_scalar("loss/gen/id2", loss_id2, it)

        # save checkpoint
        state_dict = network.state_dict()
        if it % 1000 == 0:
            torch.save(
                state_dict,
                f"{train_config.save_dir}/checkpoint_{it}.pth",
            )

        # generate sample image when training
        if it % train_config.num_iterations_per_sample_generation == 0:
            # save sample image
            output_name = f"{train_config.sample_output_dir}/sample_{it}.png"
            out = torch.cat((content_images, imgs), 0)
            out = torch.cat((style_images, out), 0)
            save_image(out, output_name)

        # ================== Train the discriminator ================== #
        # for real style images
        network.discriminator.requires_grad_(True)

        real_loss_cls, real_loss_adv = network.discriminator(
            style_images, slabels, True
        )
        # network.generator.freeze()
        # img, loss_cls, loss_adv, _, _ = network(
        #     content_images, style_images, slabels, not use_real
        # )
        # img, loss_cls, loss_adv, _, _, _, _ = network(
        #     content_images, style_images, slabels, False
        # )

        # imgs = deepcopy(imgs)

        loss_cls, loss_adv = network.discriminator(imgs.detach(), slabels, False)

        doptimizer.zero_grad()

        dis_loss = train_config.cls_weight * (
            real_loss_cls + loss_cls
        ) + train_config.bin_weight * (
            real_loss_adv + loss_adv
        )  # compute with real_loss_cls, loss_cls, real_loss_adv, loss_adv

        dis_loss.sum().backward()
        doptimizer.step()

        writer.add_scalar("loss/dis/total", dis_loss, it)
        writer.add_scalar("loss/dis/fake_cls", loss_cls, it)
        writer.add_scalar("loss/dis/fake_adv", loss_adv, it)
        writer.add_scalar("loss/dis/real_cls", real_loss_cls, it)
        writer.add_scalar("loss/dis/real_adv", real_loss_adv, it)

        print(
            f"Done iteration =[{it}]=, gen_loss: {gen_loss}, dis_loss: {dis_loss}, loss_c: {loss_c}, loss_s: {loss_s}, loss_adv: {loss_adv}, loss_cls: {loss_cls}"
        )

writer.close()

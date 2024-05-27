import argparse
import random

import torch
import torch.optim as optim
import torch.utils.data as data

from config import Config

if __name__ == "__main__":
    config = Config()

    # device config
    USE_CUDA = torch.cuda.is_available()
    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    # content img preparation
    content_dataset = None  # TODO: preprocess here
    content_iter = iter(
        data.DataLoader(
            dataset=content_dataset,
            sampler=InfiniteSamplerWrapper(content_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )

    # style img preparation
    style_dataset = None  # TODO: preprocess here
    style_iter = iter(
        data.DataLoader(
            dataset=style_dataset,
            sampler=InfiniteSamplerWrapper(style_dataset),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
        )
    )

    # define network and stuff right here
    network = None  # may include StyTr2 and Discriminator
    network.train()
    network.to(device)
    network = nn.DataParallel(network, device_ids=[0, 1])  # adjust devices

    # optimizer for generator
    optimizer = optim.Adam(
        [
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.generator.transformer.parameters()
            },
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.generator.decode.parameters()
            },
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.generator.embedding.parameters()
            },
        ],
        lr=args.lr,  # TODO: add more parameters if needed
    )

    # optimier for discriminator
    doptimizer = optim.Adam(
        [
            {  # TODO: make sure to get parameters of Discriminator
                "params": network.discriminator.parameters()
            }
        ],
        lr=args.d_lr,  # TODO: add more parameters if needed
    )

    # training loop
    for it in len(args.max_iterations):
        # implement adjust learning rate here if needed

        content_images = next(content_iter).to(device)
        styles = next(style_iter).to(device)

        # ratio to flip label in GAN
        ratio_thr = args.gan_ratio / max(it / args.gr_freq, 1.0)

        if random.random() > ratio_thr:
            use_real = True
        else:
            use_real = False

        # model output
        img, loss_cls, loss_adv, loss_c, loss_s = network(
            content_images, style, not use_real
        )

        # ================ Train the generator (StyTr2) ================ #
        network.discriminator.requires_grad_(False)

        optimizer.zero_grad()
        gen_loss = None  # compute with loss_cls, loss_adv, loss_c, loss_s
        gen_loss.sum().backward()
        optimizer.step()

        # ================== Train the discriminator ================== #
        # for real style images
        real_loss_cls, real_loss_adv = network.module.discriminator(
            style_images, slabels, use_real
        )

        network.discriminator.requires_grad_(True)

        doptimizer.zero_grad()
        dis_loss = None  # compute with real_loss_cls, real_loss_adv, loss_cls, loss_adv
        dis_loss.sum().backward()
        doptimizer.step()

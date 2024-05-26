import argparse
import random

import torch.optim as optim
import torch.utils.data as data

if __name__ == "__main__":
    # arg parser
    args = None

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

    # optimizer for generator
    optimizer = optim.Adam(
        [
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.module.transformer.parameters()
            },
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.module.decode.parameters()
            },
            {  # TODO: make sure to get parameters of StyTr2
                "params": network.module.embedding.parameters()
            },
        ],
        lr=args.lr,  # TODO: add more parameters if needed
    )

    # optimier for discriminator
    doptimizer = optim.Adam(
        [
            {  # TODO: make sure to get parameters of Discriminator
                "params": network.module.discriminator.parameters()
            }
        ],
        lr=args.d_lr,  # TODO: add more parameters if needed
    )

    # training loop
    for it in len(args.max_iterations):
        # implement adjust learning rate here if needed

        content_images = next(content_iter).to(device)
        style_images, style_labels = next(style_iter).to(device)

        # ratio to flip label in GAN
        ratio_thr = args.gan_ratio / max(it / args.gr_freq, 1.0)

        if random.random() > ratio_thr:
            use_real = True
        else:
            # use reconstruction instead of real image to make discriminator harder
            use_real = False

        # model output
        img, loss_cls, loss_adv, loss_c, loss_s = network(
            content_images, style_images, not use_real
        )

        # ================== Train the discriminator ================== #
        # for real style images
        real_loss_cls, real_loss_adv = network.module.discriminator(
            style_images, style_labels, use_real
        )

        dis_loss = None  # compute with real_loss_cls, real_loss_adv, loss_cls, loss_adv

        doptimizer.zero_grad()
        dis_loss.sum().backward()
        doptimizer.step()

        # ================ Train the generator (StyTr2) ================ #
        gen_loss = None  # compute with loss_cls, loss_adv, loss_c, loss_s

        optimizer.zero_grad()
        gen_loss.sum().backward()
        optimizer.step()

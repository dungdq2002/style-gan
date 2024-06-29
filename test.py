import os
from options.test_options import TestOptions
from data import create_dataset
from models import create_model
from util.visualizer import save_images
from util import html
import util.util as util
from PIL import Image
from data.base_dataset import get_transform

if __name__ == "__main__":
    opt = TestOptions().parse()  # get test options
    # hard-code some parameters for test
    opt.num_threads = 0  # test code only supports num_threads = 1
    opt.batch_size = 1  # test code only supports batch_size = 1
    opt.serial_batches = True  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = (
        True  # no flip; comment this line if results on flipped images are needed.
    )
    opt.display_id = (
        -1
    )  # no visdom display; the test code saves the results to a HTML file.
    dataset = create_dataset(
        opt
    )  # create a dataset given opt.dataset_mode and other options
    # train_dataset = create_dataset(util.copyconf(opt, phase="train"))
    model = create_model(opt)  # create a model given opt.model and other options
    # create a webpage for viewing the results
    web_dir = os.path.join(
        opt.results_dir, opt.name, "{}_{}".format(opt.phase, opt.epoch)
    )  # define the website directory
    print("creating web directory", web_dir)
    webpage = html.HTML(
        web_dir,
        "Experiment = %s, Phase = %s, Epoch = %s" % (opt.name, opt.phase, opt.epoch),
    )

    modified_opt = util.copyconf(
        opt,
        load_size=opt.load_size,
    )
    transform = get_transform(modified_opt)

    A_paths = os.listdir("./dataset/inference/valA")
    B_paths = os.listdir("./dataset/inference/valB/test")

    model.setup(opt)  # regular setup: load and print networks; schedulers
    model.parallelize()
    i = 1
    for A_path in A_paths:
        A_path = os.path.join("./dataset/inference/valA", A_path)
        A_img = Image.open(A_path).convert("RGB")
        A = transform(A_img)
        A = A.unsqueeze(0)

        for B_path in B_paths:
            # try:
            B_path = os.path.join("./dataset/inference/valB/test", B_path)

            B_img = Image.open(B_path).convert("RGB")
            B = transform(B_img)
            # add batch size dimension to A and B
            B = B.unsqueeze(0)
            # print("A shape", A.shape, "B shape", B.shape)

            model.set_input({"A": A, "B": B, "A_paths": A_path, "B_paths": B_path})
            model.test()  # run inference
            visuals = model.get_current_visuals()  # get image results
            img_path = model.get_image_paths()  # get image paths
            # if i % 5 == 0:  # save images to an HTML file
            print("processing (%04d)-th image... %s" % (i, img_path))
            save_images(webpage, visuals, img_path, width=opt.display_winsize, i=i)
            i += 1

    webpage.save()  # save the HTML

    # for i, data in enumerate(dataset):
    #     if i == 0:
    #         model.setup(opt)  # regular setup: load and print networks; schedulers
    #         model.parallelize()
    #         if opt.eval:
    #             model.eval()
    #     if i >= opt.num_test:  # only apply our model to opt.num_test images.
    #         break
    #     model.set_input(data)  # unpack data from data loader
    #     model.test()  # run inference
    #     visuals = model.get_current_visuals()  # get image results
    #     img_path = model.get_image_paths()  # get image paths
    #     # if i % 5 == 0:  # save images to an HTML file
    #     print("processing (%04d)-th image... %s" % (i, img_path))
    #     save_images(webpage, visuals, img_path, width=opt.display_winsize)
    # webpage.save()  # save the HTML

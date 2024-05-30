import yaml


class Train:
    def __init__(self, raw):
        self.content_dir = raw.get("content_dir", "dataset/content/train")
        self.style_dir = raw.get("style_dir", "dataset/style/train")
        self.save_dir = raw.get("save_dir", "save")
        self.log_dir = raw.get("log_dir", "log")
        self.batch_size = raw.get("batch_size", 8)
        # self.epochs = raw["epochs"]
        self.max_iterations = raw.get("max_iterations", 160000)
        self.lr = raw.get("lr", 1e-3)
        self.d_lr = raw.get("d_lr", 1e-3)
        self.content_weight = raw.get("content_weight", 1)
        self.style_weight = raw.get("style_weight", 1)
        self.cls_weight = raw.get("cls_weight", 1)
        self.bin_weight = raw.get("bin_weight", 1)
        self.id_1_weight = raw.get("id_1_weight", 50)
        self.id_2_weight = raw.get("id_2_weight", 1)
        self.gan_ratio = raw.get("gan_ratio", 1)  # ratio to flip label in GAN
        self.gr_freq = raw.get("gr_freq", 1)  # frequency to adjust gan ratio
        self.num_workers = raw.get("num_workers", 4)
        self.vgg_path = raw.get("vgg_path", None)
        self.sample_output_dir = raw.get("sample_output_dir", "out/sample")
        self.num_iterations_per_sample_generation = raw.get(
            "num_iterations_per_sample_generation", 1
        )
        self.lr_decay = raw.get("lr_decay", 5e-4)


class Config:
    def __init__(self, config_path):
        with open(config_path, "r") as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)
            self.train = Train(raw["train"])

import yaml


class Train:
    def __init__(self, raw):
        self.content_dir = raw["content_dir"]
        self.style_dir = raw["style_dir"]
        self.save_dir = raw["save_dir"]
        self.log_dir = raw["log_dir"]
        self.batch_size = raw["batch_size"]
        self.epochs = raw["epochs"]
        self.lr = raw["lr"]
        self.content_weight = raw["content_weight"]
        self.style_weight = raw["style_weight"]


class Config:
    def __init__(self):
        with open("config.yaml", "r") as f:
            raw = yaml.load(f, Loader=yaml.FullLoader)
            self.train = Train(raw["train"])

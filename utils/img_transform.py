from torchvision import transforms


# training data transform
def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(256),
        transforms.ToTensor(),
    ]

    return transforms.Compose(transform_list)


# test data transform
def test_transform(size, crop):
    transform_list = []

    if size != 0:
        transform_list.append(transforms.Resize(size))

    if crop:
        transform_list.append(transforms.CenterCrop(size))

    transform_list.append(transforms.ToTensor())

    return transforms.Compose(transform_list)

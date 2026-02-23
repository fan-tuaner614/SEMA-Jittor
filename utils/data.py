import numpy as np
from PIL import Image
from jittor.dataset import Dataset, ImageFolder
from jittor import transform
from utils.toolkit import split_images_labels


class RandomCropWithPadding:
    """Jittor 的 RandomCrop 不支持 padding 参数，此类模拟 PyTorch 的行为。"""
    def __init__(self, size, padding=0):
        self.size = size
        self.padding = padding

    def __call__(self, img):
        if self.padding > 0:
            # img is PIL Image
            from PIL import ImageOps
            img = ImageOps.expand(img, border=self.padding, fill=0)
        return transform.RandomCrop(self.size)(img)


class iData(object):
    train_trsf = []
    test_trsf = []
    common_trsf = []
    class_order = None


class iCIFAR10(iData):
    use_path = False
    train_trsf = [
        RandomCropWithPadding(32, padding=4),
        transform.RandomHorizontalFlip(p=0.5),
        transform.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = []
    common_trsf = [
        transform.ToTensor(),
        transform.ImageNormalize(
            mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
        ),
    ]
    class_order = np.arange(10).tolist()

    def download_data(self):
        from torchvision import datasets as tv_datasets
        train_dataset = tv_datasets.cifar.CIFAR10("./data", train=True, download=True)
        test_dataset = tv_datasets.cifar.CIFAR10("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iCIFAR100(iData):
    use_path = False
    train_trsf = [
        RandomCropWithPadding(32, padding=4),
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=63 / 255),
        transform.ToTensor()
    ]
    test_trsf = [transform.ToTensor()]
    common_trsf = [
        transform.ImageNormalize(
            mean=(0.5071, 0.4867, 0.4408), std=(0.2675, 0.2565, 0.2761)
        ),
    ]
    class_order = np.arange(100).tolist()

    def download_data(self):
        from torchvision import datasets as tv_datasets
        train_dataset = tv_datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = tv_datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


def build_transform(is_train, args=None):
    input_size = 224
    resize_im = input_size > 32
    if is_train:
        scale = (0.05, 1.0)
        ratio = (3. / 4., 4. / 3.)
        t = [
            transform.RandomResizedCrop(input_size, scale=scale, ratio=ratio),
            transform.RandomHorizontalFlip(p=0.5),
            transform.ToTensor(),
        ]
        return t

    t = []
    if resize_im:
        size = int((256 / 224) * input_size)
        t.append(transform.Resize(size))
        t.append(transform.CenterCrop(input_size))
    t.append(transform.ToTensor())
    return t


class iCIFAR224(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = False
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(100).tolist()

    def download_data(self):
        from torchvision import datasets as tv_datasets
        train_dataset = tv_datasets.cifar.CIFAR100("./data", train=True, download=True)
        test_dataset = tv_datasets.cifar.CIFAR100("./data", train=False, download=True)
        self.train_data, self.train_targets = train_dataset.data, np.array(train_dataset.targets)
        self.test_data, self.test_targets = test_dataset.data, np.array(test_dataset.targets)


class iImageNet1000(iData):
    use_path = True
    train_trsf = [
        transform.RandomResizedCrop(224),
        transform.RandomHorizontalFlip(),
        transform.ColorJitter(brightness=63 / 255),
    ]
    test_trsf = [
        transform.Resize(256),
        transform.CenterCrop(224),
    ]
    common_trsf = [
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"


class iImageNet100(iData):
    use_path = True
    train_trsf = [
        transform.RandomResizedCrop(224),
        transform.RandomHorizontalFlip(),
    ]
    test_trsf = [
        transform.Resize(256),
        transform.CenterCrop(224),
    ]
    common_trsf = [
        transform.ToTensor(),
        transform.ImageNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
    class_order = np.arange(1000).tolist()

    def download_data(self):
        assert 0, "You should specify the folder of your dataset"


class iImageNetR(iData):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.use_path = True
        self.train_trsf = build_transform(True, args)
        self.test_trsf = build_transform(False, args)
        self.common_trsf = []
        self.class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "data/imagenet-r/train/"
        test_dir = "data/imagenet-r/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class iImageNetA(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "data/imagenet-a/train/"
        test_dir = "data/imagenet-a/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class CUB(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "data/cub/train/"
        test_dir = "data/cub/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class objectnet(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(200).tolist()

    def download_data(self):
        train_dir = "data/objectnet/train/"
        test_dir = "data/objectnet/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class omnibenchmark(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(300).tolist()

    def download_data(self):
        train_dir = "data/omnibenchmark/train/"
        test_dir = "data/omnibenchmark/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)


class vtab(iData):
    use_path = True
    train_trsf = build_transform(True, None)
    test_trsf = build_transform(False, None)
    common_trsf = []
    class_order = np.arange(50).tolist()

    def download_data(self):
        train_dir = "data/vtab/train/"
        test_dir = "data/vtab/test/"
        train_dset = ImageFolder(train_dir)
        test_dset = ImageFolder(test_dir)
        self.train_data, self.train_targets = split_images_labels(train_dset.imgs)
        self.test_data, self.test_targets = split_images_labels(test_dset.imgs)

import logging
import numpy as np
from PIL import Image
from jittor.dataset import Dataset
from jittor import transform
from utils.data import (iCIFAR10, iCIFAR100, iImageNet100, iImageNet1000,
                         iCIFAR224, iImageNetR, iImageNetA, CUB,
                         objectnet, omnibenchmark, vtab)


class DataManager(object):
    def __init__(self, dataset_name, shuffle, seed, init_cls, increment, args):
        self.args = args
        self.dataset_name = dataset_name
        self._setup_data(dataset_name, shuffle, seed)
        assert init_cls <= len(self._class_order), "No enough classes."
        self._increments = [init_cls]
        while sum(self._increments) + increment < len(self._class_order):
            self._increments.append(increment)
        offset = len(self._class_order) - sum(self._increments)
        if offset > 0:
            self._increments.append(offset)

    @property
    def nb_tasks(self):
        return len(self._increments)

    def get_task_size(self, task):
        return self._increments[task]

    @property
    def nb_classes(self):
        return len(self._class_order)

    def get_dataset(self, indices, source, mode, appendent=None, ret_data=False, m_rate=None):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transform.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "flip":
            trsf = transform.Compose([
                *self._test_trsf,
                transform.RandomHorizontalFlip(p=1.0),
                *self._common_trsf,
            ])
        elif mode == "test":
            trsf = transform.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        data, targets = [], []
        for idx in indices:
            if m_rate is None:
                class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            else:
                class_data, class_targets = self._select_rmm(x, y, low_range=idx, high_range=idx + 1, m_rate=m_rate)
            data.append(class_data)
            targets.append(class_targets)

        if appendent is not None and len(appendent) != 0:
            appendent_data, appendent_targets = appendent
            data.append(appendent_data)
            targets.append(appendent_targets)

        data, targets = np.concatenate(data), np.concatenate(targets)

        if ret_data:
            return data, targets, DummyDataset(data, targets, trsf, self.use_path)
        else:
            return DummyDataset(data, targets, trsf, self.use_path)

    def get_dataset_with_split(self, indices, source, mode, appendent=None, val_samples_per_class=0):
        if source == "train":
            x, y = self._train_data, self._train_targets
        elif source == "test":
            x, y = self._test_data, self._test_targets
        else:
            raise ValueError("Unknown data source {}.".format(source))

        if mode == "train":
            trsf = transform.Compose([*self._train_trsf, *self._common_trsf])
        elif mode == "test":
            trsf = transform.Compose([*self._test_trsf, *self._common_trsf])
        else:
            raise ValueError("Unknown mode {}.".format(mode))

        train_data, train_targets = [], []
        val_data, val_targets = [], []
        for idx in indices:
            class_data, class_targets = self._select(x, y, low_range=idx, high_range=idx + 1)
            val_indx = np.random.choice(len(class_data), val_samples_per_class, replace=False)
            train_indx = list(set(np.arange(len(class_data))) - set(val_indx))
            val_data.append(class_data[val_indx])
            val_targets.append(class_targets[val_indx])
            train_data.append(class_data[train_indx])
            train_targets.append(class_targets[train_indx])

        if appendent is not None:
            appendent_data, appendent_targets = appendent
            for idx in range(0, int(np.max(appendent_targets)) + 1):
                append_data, append_targets = self._select(
                    appendent_data, appendent_targets, low_range=idx, high_range=idx + 1
                )
                val_indx = np.random.choice(len(append_data), val_samples_per_class, replace=False)
                train_indx = list(set(np.arange(len(append_data))) - set(val_indx))
                val_data.append(append_data[val_indx])
                val_targets.append(append_targets[val_indx])
                train_data.append(append_data[train_indx])
                train_targets.append(append_targets[train_indx])

        train_data, train_targets = np.concatenate(train_data), np.concatenate(train_targets)
        val_data, val_targets = np.concatenate(val_data), np.concatenate(val_targets)

        return (DummyDataset(train_data, train_targets, trsf, self.use_path),
                DummyDataset(val_data, val_targets, trsf, self.use_path))

    def _setup_data(self, dataset_name, shuffle, seed):
        idata = _get_idata(dataset_name, self.args)
        idata.download_data()

        self._train_data, self._train_targets = idata.train_data, idata.train_targets
        self._test_data, self._test_targets = idata.test_data, idata.test_targets
        self.use_path = idata.use_path

        self._train_trsf = idata.train_trsf
        self._test_trsf = idata.test_trsf
        self._common_trsf = idata.common_trsf

        order = [i for i in range(len(np.unique(self._train_targets)))]
        if shuffle:
            np.random.seed(seed)
            order = np.random.permutation(len(order)).tolist()
        else:
            order = idata.class_order
        self._class_order = order
        logging.info(self._class_order)

        self._train_targets = _map_new_class_index(self._train_targets, self._class_order)
        self._test_targets = _map_new_class_index(self._test_targets, self._class_order)

    def _select(self, x, y, low_range, high_range):
        idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[idxes], y[idxes]

    def _select_rmm(self, x, y, low_range, high_range, m_rate):
        assert m_rate is not None
        if m_rate != 0:
            idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
            selected_idxes = np.random.randint(0, len(idxes), size=int((1 - m_rate) * len(idxes)))
            new_idxes = idxes[selected_idxes]
            new_idxes = np.sort(new_idxes)
        else:
            new_idxes = np.where(np.logical_and(y >= low_range, y < high_range))[0]
        return x[new_idxes], y[new_idxes]

    def getlen(self, index):
        y = self._train_targets
        return np.sum(np.where(y == index))


class DummyDataset(Dataset):
    """自定义数据集，继承 jittor.dataset.Dataset
    
    注意：Jittor 的 Dataset 在 __getitem__ 中返回的数据会被自动转为 jt.Var。
    图片预缓存到内存中，避免每次迭代都从磁盘读取（num_workers=0 时尤其重要）。
    """
    _MAX_CACHE = 3000

    def __init__(self, images, labels, trsf, use_path=False):
        super().__init__()
        assert len(images) == len(labels), "Data size error!"
        self.images = images
        self.labels = labels
        self.trsf = trsf
        self.use_path = use_path
        self.set_attrs(total_len=len(images))

        # Pre-cache decoded images in memory to eliminate disk I/O bottleneck
        self._image_cache = None
        if self.use_path and len(images) <= self._MAX_CACHE:
            import time
            t0 = time.time()
            try:
                self._image_cache = [np.asarray(pil_loader(p)) for p in images]
                elapsed = time.time() - t0
                logging.info(f"Pre-cached {len(images)} images in {elapsed:.1f}s")
            except MemoryError:
                self._image_cache = None
                logging.warning(f"Pre-cache skipped due to insufficient RAM for {len(images)} images.")

    def __getitem__(self, idx):
        if self.use_path:
            if self._image_cache is not None:
                image = self.trsf(Image.fromarray(self._image_cache[idx]))
            else:
                image = self.trsf(pil_loader(self.images[idx]))
        else:
            image = self.trsf(Image.fromarray(self.images[idx]))
        label = self.labels[idx]
        return idx, image, label

    def clear_cache(self):
        """Release cached images to free CPU memory."""
        self._image_cache = None


def _map_new_class_index(y, order):
    return np.array(list(map(lambda x: order.index(x), y)))


def _get_idata(dataset_name, args=None):
    name = dataset_name.lower()
    if name == "cifar10":
        return iCIFAR10()
    elif name == "cifar100":
        return iCIFAR100()
    elif name == "imagenet1000":
        return iImageNet1000()
    elif name == "imagenet100":
        return iImageNet100()
    elif name == "cifar224":
        return iCIFAR224(args)
    elif name == "imagenetr":
        return iImageNetR(args)
    elif name == "imageneta":
        return iImageNetA()
    elif name == "cub":
        return CUB()
    elif name == "objectnet":
        return objectnet()
    elif name == "omnibenchmark":
        return omnibenchmark()
    elif name == "vtab":
        return vtab()
    else:
        raise NotImplementedError("Unknown dataset {}.".format(dataset_name))


def pil_loader(path):
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")

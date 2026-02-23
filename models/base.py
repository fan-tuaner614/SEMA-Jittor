"""
Jittor version of BaseLearner for continual learning.
"""
import copy
import logging
import numpy as np
import jittor as jt
from jittor import nn
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 64


class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]  # In Jittor, device is managed differently
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(self._targets_memory), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        return self._network.feature_dim

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes,
                           self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        jt.gc()
        for batch in loader:
            _, inputs, targets = batch[0], batch[1], batch[2]
            with jt.no_grad():
                outputs = model(inputs)["logits"]
            predicts = jt.argmax(outputs, dim=1)[0]
            correct += (predicts.numpy() == targets.numpy()).sum()
            total += len(targets)
            del outputs, predicts, inputs
            jt.gc()
        return np.around(correct * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        jt.gc()
        for batch in loader:
            _, inputs, targets = batch[0], batch[1], batch[2]
            with jt.no_grad():
                outputs = self._network(inputs)["logits"]
            # top-k (Jittor argsort returns (indices, values), opposite to PyTorch)
            sorted_indices = jt.argsort(outputs, dim=1, descending=True)[0]
            predicts = sorted_indices[:, :self.topk]
            y_pred.append(predicts.numpy())
            y_true.append(targets.numpy())
            del outputs, inputs
            jt.gc()
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T
        return np.argsort(scores, axis=1)[:, :self.topk], y_true

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        with jt.no_grad():
            for batch in loader:
                _, _inputs, _targets = batch[0], batch[1], batch[2]
                _targets = _targets.numpy()
                _vectors = self._network.extract_vector(_inputs)
                if isinstance(_vectors, dict):
                    _vectors = _vectors["features"]
                _vectors = _vectors.numpy()
                vectors.append(_vectors)
                targets.append(_targets)
        return np.concatenate(vectors), np.concatenate(targets)

    def save_checkpoint(self, filename):
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        jt.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

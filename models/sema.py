"""
Jittor version of SEMA Learner.
"""
import logging
import numpy as np
import jittor as jt
from jittor import nn, optim
from tqdm import tqdm
import math
from utils.inc_net import SEMAVitNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
from backbone.sema_block import SEMAModules

import os
num_workers = 0 if os.name == 'nt' else 4  # Windows 上多进程共享内存易 OOM


class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = SEMAVitNet(args, True)
        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.weight_decay = args.get("weight_decay", 0.0005)
        self.min_lr = args.get('min_lr', 1e-8)
        self.args = args

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        if self._cur_task == 0:
            self._network.fc = nn.Linear(768, data_manager.nb_classes)
            nn.init.kaiming_uniform_(self._network.fc.weight, a=math.sqrt(5))
            nn.init.zero_(self._network.fc.bias)
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes), source="train", mode="train")
        self.train_dataset = train_dataset
        self.data_manager = data_manager

        # Jittor DataLoader: set_attrs on Dataset for batch_size, shuffle, num_workers
        train_dataset.set_attrs(batch_size=self.batch_size, shuffle=True, num_workers=num_workers)
        self.train_loader = train_dataset

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test")
        test_dataset.set_attrs(batch_size=self.batch_size, shuffle=False, num_workers=num_workers)
        self.test_loader = test_dataset

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        if self._cur_task == 0:
            total_params = sum(p.numel() for p in self._network.parameters())
            print(f'{total_params:,} total parameters.')
            total_trainable_params = sum(
                p.numel() for p in self._network.parameters() if not p.is_stop_grad())
            print(f'{total_trainable_params:,} training parameters.')
            self._train_new(train_loader, test_loader)
        else:
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = True

            # Count max adapters to estimate memory pressure
            max_adapters = 1
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    max_adapters = max(max_adapters, module.num_adapters)

            # Use small detect batch to avoid GPU OOM (8GB GPU)
            # Scale down as adapter count grows: more adapters = more memory per forward
            detect_bs = min(self.args["detect_batch_size"], self.batch_size // 2, 24)
            detect_bs = max(detect_bs // max(1, max_adapters // 3), 4)
            detect_dataset = self.data_manager.get_dataset(
                np.arange(self._known_classes, self._total_classes), source="train", mode="train")
            detect_dataset.set_attrs(
                batch_size=detect_bs, shuffle=True, num_workers=num_workers)

            # Free GPU memory before detection
            jt.sync_all()
            jt.gc()
            added = self._detect_outlier(detect_dataset, train_loader, test_loader, 0)

            # Free detect dataset cache after detection is complete
            if hasattr(detect_dataset, 'clear_cache'):
                detect_dataset.clear_cache()
            del detect_dataset
            jt.sync_all()
            jt.gc()

            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    module.detecting_outlier = False
            if added == 0:
                jt.gc()
                self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
                if self.optimizer is not None:
                    self._init_train(self.args['func_epoch'], train_loader, test_loader,
                                     self.optimizer, self.scheduler, phase='func')
                else:
                    logging.warning("No trainable func params for Task %d, skipping func training.", self._cur_task)

        jt.sync_all()
        jt.gc()
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                module.end_of_task_training()

    def _train_new(self, train_loader, test_loader):
        # Count max adapters across all blocks to determine memory pressure
        max_adapters = 1
        for module in self._network.backbone.modules():
            if isinstance(module, SEMAModules):
                max_adapters = max(max_adapters, module.num_adapters)

        # For task > 0, scale batch size inversely with adapter count
        # More adapters → more forward-pass memory per sample
        if self._cur_task > 0:
            # Base reduction: /2. Additional reduction for heavy adapter counts.
            divisor = max(2, max_adapters // 2)
            func_bs = max(self.batch_size // divisor, 4)
        else:
            func_bs = self.batch_size
        train_loader.set_attrs(batch_size=func_bs)
        if func_bs != self.batch_size:
            logging.info(f"Func training with batch_size={func_bs} (reduced from {self.batch_size})")

        self.update_optimizer_and_scheduler(num_epoch=self.args['func_epoch'], lr=self.init_lr)
        if self.optimizer is not None:
            self._init_train(self.args['func_epoch'], train_loader, test_loader,
                             self.optimizer, self.scheduler, phase='func')
        else:
            logging.warning("Skipping func training: no trainable functional parameters.")

        # Aggressive cleanup between phases: release func optimizer graph references
        self.optimizer = None
        self.scheduler = None
        jt.sync_all()
        jt.gc()

        # RD uses even smaller batch to reduce peak GPU memory
        rd_bs = max(func_bs // 2, 8)
        train_loader.set_attrs(batch_size=rd_bs)
        logging.info(f"RD training with batch_size={rd_bs}")

        self.update_rd_optimizer_and_scheduler(num_epoch=self.args['rd_epoch'], lr=self.args['rd_lr'])
        if self.rd_optimizer is not None:
            self._init_train(self.args['rd_epoch'], train_loader, test_loader,
                             self.rd_optimizer, self.rd_scheduler, phase='rd')
        else:
            logging.warning("Skipping RD training phase: no trainable RD parameters.")

        # Restore original batch size
        train_loader.set_attrs(batch_size=self.batch_size)
        self.rd_optimizer = None
        self.rd_scheduler = None
        jt.sync_all()
        jt.gc()

    def _detect_outlier(self, detect_loader, train_loader, test_loader, added):
        is_added = False
        self._network.eval()
        jt.gc()
        for i, batch in enumerate(detect_loader):
            _, inputs, targets = batch[0], batch[1], batch[2]
            with jt.no_grad():
                model_outcome = self._network(inputs)
            added_record = model_outcome["added_record"]
            # Free forward-pass intermediates immediately
            del model_outcome, inputs, targets
            jt.gc()

            if sum(added_record) > 0:
                added += 1
                is_added = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = False

                # Free detect_loader image cache before training to reclaim CPU memory
                if hasattr(detect_loader, 'clear_cache'):
                    detect_loader.clear_cache()
                jt.sync_all()
                jt.gc()

                self._train_new(train_loader, test_loader)
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.detecting_outlier = True
                for module in self._network.backbone.modules():
                    if isinstance(module, SEMAModules):
                        module.freeze_functional()
                        module.freeze_rd()
                        module.reset_newly_added_status()
                jt.sync_all()
                jt.gc()

        if is_added:
            return self._detect_outlier(detect_loader, train_loader, test_loader, added)
        else:
            return added

    def _init_train(self, total_epoch, train_loader, test_loader, optimizer, scheduler, phase='func'):
        prog_bar = tqdm(range(total_epoch))
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0

            for i, batch in enumerate(train_loader):
                _, inputs, targets = batch[0], batch[1], batch[2]
                outcome = self._network(inputs)
                logits = outcome["logits"]
                logits = logits[:, :self._total_classes]
                if self._cur_task > 0:
                    # Use large negative value instead of -inf to avoid NaN
                    # in Jittor's cross_entropy (exp(-inf) can cause NaN gradients)
                    logits[:, :self._known_classes] = -1e9

                if phase == "func":
                    loss = nn.cross_entropy_loss(logits, targets.long())
                elif phase == "rd":
                    loss = outcome["rd_loss"]

                optimizer.step(loss)

                losses += loss.item()

                with jt.no_grad():
                    preds = jt.argmax(logits, dim=1)[0]
                    correct += (preds.numpy() == targets.numpy()).sum()
                    total += len(targets)

                # Periodic GPU memory cleanup to prevent OOM
                if (i + 1) % 10 == 0:
                    jt.sync_all()
                    jt.gc()

            scheduler.step()
            jt.sync_all()
            jt.gc()
            train_acc = np.around(correct * 100 / total, decimals=2) if total > 0 else 0

            # Only evaluate on test set every 5 epochs or the last epoch to save time
            eval_interval = 5
            if (epoch + 1) % eval_interval == 0 or epoch + 1 == total_epoch:
                test_acc = self._compute_accuracy(self._network, test_loader)
                self._network.train()  # restore training mode after eval
            else:
                test_acc = -1  # skipped
            info = "{} Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                phase, self._cur_task, epoch + 1, total_epoch,
                losses / max(i + 1, 1), train_acc,
                test_acc if test_acc >= 0 else float('nan'))
            prog_bar.set_description(info)

        logging.info(info)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        jt.gc()
        for batch in loader:
            _, inputs, targets = batch[0], batch[1], batch[2]
            with jt.no_grad():
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
            sorted_indices = jt.argsort(outputs, dim=1, descending=True)[0]
            predicts = sorted_indices[:, :self.topk]
            y_pred.append(predicts.numpy())
            y_true.append(targets.numpy())
            del outcome, logits, outputs, inputs
            jt.gc()
        return np.concatenate(y_pred), np.concatenate(y_true)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        jt.gc()
        with jt.no_grad():
            for batch in loader:
                _, inputs, targets = batch[0], batch[1], batch[2]
                outcome = self._network(inputs)
                logits = outcome["logits"]
                outputs = logits[:, :self._total_classes]
                predicts = jt.argmax(outputs, dim=1)[0]
                correct += (predicts.numpy() == targets.numpy()).sum()
                total += len(targets)
                # Free each batch's intermediates
                del outcome, logits, outputs, predicts, inputs
                jt.gc()
        return np.around(correct * 100 / total, decimals=2) if total > 0 else 0

    def _unfreeze_trainable_params(self, phase='func'):
        """Explicitly unfreeze params that should be trainable for this training phase.
        
        Jittor may silently freeze params during optimizer.step() or graph cleanup.
        This ensures the correct params are unfrozen before each training phase.
        Only unfreeze addition-layer (adapt_start ~ adapt_end) routers to avoid
        zero-gradient warnings for non-addition-layer routers.
        """
        adapt_start = self.args.get('adapt_start_layer', 0)
        adapt_end = self.args.get('adapt_end_layer', 11)
        if phase == 'func':
            # Always unfreeze the fc (classification head)
            if self._network.fc is not None:
                for p in self._network.fc.parameters():
                    p.start_grad()
            # Unfreeze newly-added adapters' functional params and addition-layer routers
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    for adapter in module.adapters:
                        if adapter.newly_added:
                            for p in adapter.functional.parameters():
                                p.start_grad()
                    # Only unfreeze router for addition layers (where router is actually used)
                    if adapt_start <= module.layer_id <= adapt_end:
                        for p in module.router.parameters():
                            p.start_grad()
                        if module.new_router is not None:
                            for p in module.new_router.parameters():
                                p.start_grad()
        elif phase == 'rd':
            # Unfreeze newly-added adapters' RD params
            for module in self._network.backbone.modules():
                if isinstance(module, SEMAModules):
                    for adapter in module.adapters:
                        if adapter.newly_added and adapter.rd is not None:
                            for p in adapter.rd.parameters():
                                p.start_grad()

    def _is_addition_layer_router(self, param_name):
        """Check if a router param belongs to an addition layer (adapt_start ~ adapt_end).
        Non-addition-layer routers never get gradients and should be excluded from optimizer."""
        if 'router' not in param_name:
            return False
        # Extract block index from name like 'backbone.blocks.9.adapter_module.router.weight'
        parts = param_name.split('.')
        for i, part in enumerate(parts):
            if part == 'blocks' and i + 1 < len(parts):
                try:
                    block_idx = int(parts[i + 1])
                    start = self.args.get('adapt_start_layer', 0)
                    end = self.args.get('adapt_end_layer', 11)
                    return start <= block_idx <= end
                except ValueError:
                    pass
        return True  # fallback: include

    def update_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["init_lr"] if lr is None else lr
        self._unfreeze_trainable_params(phase='func')
        # Only include router params from addition layers (blocks adapt_start ~ adapt_end).
        # Non-addition-layer routers never participate in forward pass, so they have no gradient.
        func_params = []
        for n, p in self._network.named_parameters():
            if p.is_stop_grad():
                continue
            if 'functional' in n or 'fc' in n:
                func_params.append(p)
            elif 'router' in n:
                if self._is_addition_layer_router(n):
                    func_params.append(p)
                else:
                    # Freeze non-addition-layer router to avoid zero-gradient warnings
                    p.stop_grad()
        logging.info(f"Func trainable params: {sum(p.numel() for p in func_params)}")
        if len(func_params) == 0:
            logging.warning("No trainable func params found! Skipping func training phase.")
            self.optimizer = None
            self.scheduler = None
            return
        if self.args['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(func_params, lr=lr, momentum=0.9,
                                       weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.optimizer = optim.Adam(func_params, lr=lr,
                                        weight_decay=self.args["weight_decay"])

        self.scheduler = CosineAnnealingLR(self.optimizer, T_max=num_epoch, eta_min=self.min_lr)

    def update_rd_optimizer_and_scheduler(self, num_epoch=20, lr=None):
        lr = self.args["rd_lr"] if lr is None else lr
        self._unfreeze_trainable_params(phase='rd')
        rd_params = [p for n, p in self._network.named_parameters()
                     if 'rd' in n and not p.is_stop_grad()]
        logging.info(f"RD trainable params: {sum(p.numel() for p in rd_params)}")
        if len(rd_params) == 0:
            logging.warning("No trainable RD params found! Skipping RD training phase.")
            self.rd_optimizer = None
            self.rd_scheduler = None
            return

        if self.args['optimizer'] == 'sgd':
            self.rd_optimizer = optim.SGD(rd_params, lr=lr, momentum=0.9,
                                          weight_decay=self.args["weight_decay"])
        elif self.args['optimizer'] == 'adam':
            self.rd_optimizer = optim.Adam(rd_params, lr=lr,
                                           weight_decay=self.args["weight_decay"])

        self.rd_scheduler = CosineAnnealingLR(self.rd_optimizer, T_max=num_epoch, eta_min=self.min_lr)

    def save_checkpoint(self, filename):
        state_dict = self._network.state_dict()
        save_dict = {}
        for k, v in state_dict.items():
            if 'adapter' in k or ('fc' in k and 'block' not in k):
                save_dict[k] = v
        jt.save(save_dict, "{}.pth".format(filename))

    def load_checkpoint(self, filename):
        state_dict = jt.load(filename)
        self._network.load_state_dict(state_dict)


class CosineAnnealingLR:
    """Simple cosine annealing learning rate scheduler for Jittor."""
    def __init__(self, optimizer, T_max, eta_min=0):
        self.optimizer = optimizer
        self.T_max = T_max
        self.eta_min = eta_min
        self.base_lr = optimizer.lr
        self.current_step = 0

    def step(self):
        self.current_step += 1
        t = self.current_step
        T = self.T_max
        new_lr = self.eta_min + (self.base_lr - self.eta_min) * (1 + math.cos(math.pi * t / T)) / 2
        self.optimizer.lr = new_lr

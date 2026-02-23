"""
Jittor version of SEMA Block: AdapterModule and SEMAModules.
"""
import jittor as jt
from jittor import nn
from typing import List
import copy
import logging
from backbone.sema_components import Adapter, AE, Records


class AdapterModule(nn.Module):
    """Single adapter module containing a functional adapter and a representation descriptor (RD)."""
    def __init__(self, config, adapter_id, writer=None):
        super().__init__()
        self.config = config
        self.functional = Adapter(
            self.config, adapter_id, dropout=0.1, bottleneck=self.config.ffn_num,
            init_option=self.config.ffn_adapter_init_option,
            adapter_scalar=self.config.ffn_adapter_scalar,
            adapter_layernorm_option=self.config.ffn_adapter_layernorm_option,
        )
        layer_id = int(adapter_id.split('.')[0])
        self.not_addition_layer = layer_id < config.adapt_start_layer or layer_id > config.adapt_end_layer
        if self.not_addition_layer:
            self.rd = None
        else:
            self.rd = AE(self.config)
        self.activation = nn.ReLU()
        self.newly_added = True
        self.adapter_id = adapter_id
        self.writer = writer
        self.rd_loss_record = Records(max_len=config.buffer_size)

    def execute(self, x):
        func_out = self.functional(x)
        batch_zeros = jt.zeros((x.shape[0],), dtype=x.dtype)
        if self.not_addition_layer:
            return func_out, batch_zeros, batch_zeros

        # Skip RD computation for frozen adapters during training:
        # their rd_loss is never used in the loss function, and
        # rd_loss_record.updating is False, so recording is a no-op anyway.
        # This saves significant GPU memory and compute.
        if not self.newly_added and self.is_training() and not self.rd_loss_record.updating:
            return func_out, batch_zeros, batch_zeros

        rd_loss = self.rd.compute_reconstruction_loss(x)
        z_score = self.get_z_score_deviation(rd_loss)
        if self.is_training() and self.rd_loss_record.updating:
            self.add_z_score_record(rd_loss)
        return func_out, rd_loss, z_score

    def get_z_score_deviation(self, rd_loss):
        mean, stddev = self.rd_loss_record.mean, self.rd_loss_record.stddev
        if not self.rd_loss_record.length > 2:
            return jt.zeros_like(rd_loss)
        z_score = (rd_loss - mean) / stddev
        z_score = jt.abs(z_score)
        return z_score

    def add_z_score_record(self, rd_loss):
        self.rd_loss_record.add_record(rd_loss.detach().numpy())


class SEMAModules(nn.Module):
    """SEMA module managing multiple adapters and routing logic per transformer block."""
    def __init__(self, config, layer_id, writer=None):
        super().__init__()
        self.adapters: List[AdapterModule] = nn.ModuleList()
        self.config = config
        self.act_func = nn.ReLU()
        self.layer_id = layer_id
        self.writer = writer
        self.newly_added = True
        self.added_for_task = True
        self.adapt_start_layer = config.adapt_start_layer
        self.adapt_end_layer = config.adapt_end_layer
        # initialize with one adapter
        self.add_adapter(initialize=True)
        self.added_adapter = 0

        self.router = nn.Linear(config.d_model, 1)
        self.new_router = None
        self.detecting_outlier = False

        # Top-k routing
        self.use_topk_routing = getattr(config, 'use_topk_routing', False)
        self.top_k_adapters = getattr(config, 'top_k_adapters', 1)

    @property
    def num_adapters(self):
        return len(self.adapters)

    def set_new_router(self):
        self.new_router = nn.Linear(self.config.d_model, 1)

    def fix_router(self):
        trained_router = nn.Linear(self.config.d_model, len(self.adapters))
        old_router = self.router
        weight = copy.deepcopy(old_router.weight.data)
        new_weight = copy.deepcopy(self.new_router.weight.data)
        weight = jt.concat([weight, new_weight], dim=0)
        trained_router.weight = weight
        bias = copy.deepcopy(old_router.bias.data)
        new_bias = copy.deepcopy(self.new_router.bias.data)
        bias = jt.concat([bias, new_bias], dim=0)
        trained_router.bias = bias
        self.router = trained_router
        self.new_router = None

    def _topk_routing(self, logits, func_outs):
        """Top-k routing: sparse adapter activation (vectorized over batch)."""
        batch_size = logits.shape[0]
        num_adapters = logits.shape[1]
        k = min(self.top_k_adapters, num_adapters)

        # argsort returns (indices, sorted_values) in Jittor, opposite to PyTorch
        sorted_indices, sorted_values = jt.argsort(logits, dim=1, descending=True)
        topk_indices = sorted_indices[:, :k]            # (B, k)
        topk_scores = sorted_values[:, :k]              # (B, k)
        topk_weights = nn.softmax(topk_scores, dim=1)   # (B, k)

        # func_outs: (num_adapters, B, seq_len, dim)
        func_out = jt.zeros_like(func_outs[0])           # (B, seq_len, dim)
        adapter_range = jt.arange(num_adapters).reshape(1, -1)  # (1, num_adapters)

        for j in range(k):
            idx = topk_indices[:, j].reshape(-1, 1)      # (B, 1)
            # Build one-hot mask via broadcast comparison
            mask = (idx == adapter_range)                  # (B, num_adapters)
            # Expand mask for 4D broadcast: (num_adapters, B, 1, 1)
            mask_4d = mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)
            selected = (func_outs * mask_4d).sum(dim=0)   # (B, S, D)
            w = topk_weights[:, j].unsqueeze(-1).unsqueeze(-1)  # (B, 1, 1)
            func_out = func_out + w * selected
        return func_out

    def add_adapter(self, initialize=False):
        adapter_id = f"{self.layer_id}.{len(self.adapters)}"
        new_adapter = AdapterModule(self.config, adapter_id, self.writer)
        self.newly_added = True
        self.added_for_task = True
        self.adapters.append(new_adapter)
        if not initialize:
            self.set_new_router()
        logging.info(f"Adapter {adapter_id} added at block {self.layer_id}")

    def execute(self, x):
        rd_loss = jt.array(0.0)
        added = False
        not_addition_layer = self.layer_id < self.adapt_start_layer or self.layer_id > self.adapt_end_layer

        if not_addition_layer:
            func_out, _, _ = self.adapters[-1](x)
        else:
            func_outs, rd_losses, z_scores = [], [], []
            for adapter in self.adapters:
                func_out, rd_l, z_score = adapter(x)
                # Detach frozen adapter outputs: gradients don't need to flow
                # through their computation graph (params already stop_grad).
                # This prevents Jittor from keeping huge intermediate tensors
                # in GPU memory as the number of adapters grows.
                if not adapter.newly_added:
                    func_out = func_out.detach()
                func_outs.append(func_out)
                rd_losses.append(rd_l)
                z_scores.append(z_score)

            func_outs = jt.stack(func_outs)
            rd_losses = jt.stack(rd_losses)
            z_scores = jt.stack(z_scores)

            # Convert z-score min to Python float SAFELY before comparison
            # to avoid implicit .item() call that can crash on CUDA OOM
            try:
                jt.sync_all()
                z_min_val = float(z_scores.mean(dim=1).min().numpy())
            except Exception:
                z_min_val = 0.0  # Fallback: don't add adapter if comparison fails

            addition_criteria = (z_min_val > self.config.exp_threshold
                                 and self.layer_id >= self.adapt_start_layer
                                 and self.layer_id <= self.adapt_end_layer
                                 and not self.added_for_task and self.detecting_outlier)

            if addition_criteria:
                self.add_adapter()
                out = {"func_out": jt.zeros_like(func_outs[0]),
                       "rd_loss": jt.array(0.0), "added": True}
                return out
            else:
                logits = self.router(x.mean(dim=1))
                if self.new_router is not None:
                    new_logits = self.new_router(x.mean(dim=1))
                    logits = jt.concat([logits, new_logits], dim=1)

                if self.use_topk_routing and self.num_adapters > 1:
                    func_out = self._topk_routing(logits, func_outs)
                else:
                    mask = nn.softmax(logits, dim=1)
                    func_out = (func_outs * mask.transpose(0, 1).unsqueeze(-1).unsqueeze(-1)).sum(dim=0)

                if self.adapters[-1].newly_added:
                    rd_loss = rd_losses[-1].mean()
                else:
                    rd_loss = jt.array(0.0)

        out = {"func_out": func_out, "rd_loss": rd_loss, "added": added}
        return out

    def end_of_task_training(self):
        self.freeze_functional()
        self.freeze_rd()
        self.reset_newly_added_status()
        self.added_for_task = False

    def reset_newly_added_status(self):
        self.newly_added = False
        for adapter in self.adapters:
            adapter.newly_added = False

    def freeze_functional(self):
        for adapter in self.adapters:
            for param in adapter.functional.parameters():
                param.stop_grad()
        if self.new_router is not None:
            self.fix_router()
        for param in self.router.parameters():
            param.stop_grad()

    def freeze_rd(self):
        for adapter in self.adapters:
            if adapter.rd is not None:
                for param in adapter.rd.parameters():
                    param.stop_grad()
                adapter.rd_loss_record.updating = False

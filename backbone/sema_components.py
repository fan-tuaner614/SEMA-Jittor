"""
Jittor version of SEMA components: Adapter, AE (AutoEncoder as RD), Records.
"""
import jittor as jt
from jittor import nn
import math


class Adapter(nn.Module):
    """Functional adapter: down-project -> ReLU -> up-project."""
    def __init__(self,
                 config=None,
                 adapter_id=None,
                 d_model=None,
                 bottleneck=None,
                 dropout=0.0,
                 init_option="bert",
                 adapter_scalar="1.0",
                 adapter_layernorm_option="in"):
        super().__init__()
        self.adapter_id = adapter_id
        self.n_embd = config.d_model if d_model is None else d_model
        self.down_size = config.attn_bn if bottleneck is None else bottleneck

        self.adapter_layernorm_option = adapter_layernorm_option
        self.adapter_layer_norm_before = None
        if adapter_layernorm_option == "in" or adapter_layernorm_option == "out":
            self.adapter_layer_norm_before = nn.LayerNorm(self.n_embd)

        if adapter_scalar == "learnable_scalar":
            self.scale = jt.ones((1,))
        else:
            self.scale = float(adapter_scalar)

        self.down_proj = nn.Linear(self.n_embd, self.down_size)
        self.non_linear_func = nn.ReLU()
        self.up_proj = nn.Linear(self.down_size, self.n_embd)
        self.dropout = dropout

        if init_option == "lora":
            nn.init.kaiming_uniform_(self.down_proj.weight, a=math.sqrt(5))
            nn.init.zero_(self.up_proj.weight)
            nn.init.zero_(self.down_proj.bias)
            nn.init.zero_(self.up_proj.bias)

    def execute(self, x):
        if self.adapter_layernorm_option == 'in':
            x = self.adapter_layer_norm_before(x)
        down = self.down_proj(x)
        down = self.non_linear_func(down)
        output = self.up_proj(down)
        return output


class AE(nn.Module):
    """AutoEncoder used as Representation Descriptor (RD)."""
    def __init__(self, config):
        super(AE, self).__init__()
        self.input_dim = config.d_model
        self.config = config
        self.encoder = nn.Linear(self.input_dim, config.rd_dim)
        self.decoder = nn.Linear(config.rd_dim, self.input_dim)
        self.weight_initialize()

    def execute(self, x):
        encoded = self.encoder(x)
        reconstruction = self.decoder(encoded)
        return reconstruction

    def compute_reconstruction_loss(self, x):
        # Detach x from backbone computation graph:
        # AE only reconstructs x, gradients should only flow to AE params,
        # not back through the frozen transformer backbone.
        x_in = x.detach().mean(dim=1)
        reconstruction = self.execute(x_in)
        # Vectorized MSE per sample: mean over feature dim
        reconstruction_losses = ((reconstruction - x_in) ** 2).mean(dim=-1)
        return reconstruction_losses

    def reconstruction_loss(self, reconstruction, x):
        return nn.mse_loss(reconstruction, x)

    def weight_initialize(self):
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zero_(self.encoder.bias)
        nn.init.kaiming_uniform_(self.decoder.weight, a=math.sqrt(5))
        nn.init.zero_(self.decoder.bias)


class Records:
    """Buffer for recording RD losses and computing z-score based outlier detection."""
    def __init__(self, max_len=500) -> None:
        self._max_len = max_len
        self._curr_len = 0
        self.record = [0.0] * self._max_len  # use python list for CPU-only tracking
        self._mean = 0.0
        self._var = 0.0
        self.updating = True

    @property
    def length(self):
        return self._curr_len

    @property
    def mean(self):
        return self._mean

    @property
    def stddev(self):
        return math.sqrt(max(self._var, 1e-12))

    def get_stats(self):
        return float(self._mean), float(self._var), int(self._curr_len)

    def set_stats(self, mean, var, length):
        self._mean = float(mean)
        self._var = float(var)
        self._curr_len = int(min(max(length, 0), self._max_len))

    def merge_stats(self, other):
        mean1, var1, n1 = self.get_stats()
        mean2, var2, n2 = other.get_stats()
        n = n1 + n2
        if n <= 0:
            return 0.0, 0.0, 0
        if n1 == 0:
            return mean2, var2, n2
        if n2 == 0:
            return mean1, var1, n1
        mean = (n1 * mean1 + n2 * mean2) / n
        var = (n1 * (var1 + (mean1 - mean) ** 2) + n2 * (var2 + (mean2 - mean) ** 2)) / n
        return float(mean), float(var), int(min(n, self._max_len))

    def add_record(self, v):
        """v is a numpy array or list of values."""
        if not self.updating:
            return
        import numpy as np
        if hasattr(v, 'numpy'):
            v = v.numpy()
        v = np.atleast_1d(v).flatten().tolist()

        if self._curr_len < self._max_len:
            place_left = self._max_len - self._curr_len
            if place_left >= len(v):
                self.record[self._curr_len:self._curr_len + len(v)] = v
                self._curr_len += len(v)
            else:
                self.record[self._curr_len:] = v[:place_left]
                self._curr_len = self._max_len
        else:
            self.record = self.record[len(v):] + v

        arr = np.array(self.record[:self._curr_len], dtype=np.float64)
        self._mean = float(arr.mean())
        self._var = float(arr.var()) if len(arr) > 1 else 0.0

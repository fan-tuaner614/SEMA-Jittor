"""
Jittor version of Vision Transformer with SEMA adapters.
"""
import jittor as jt
from jittor import nn, init
from functools import partial
from collections import OrderedDict
from backbone.sema_block import SEMAModules
import math
import numpy as np


class PatchEmbed(nn.Module):
    """Image to Patch Embedding."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def execute(self, x):
        B, C, H, W = x.shape
        x = self.proj(x)  # (B, embed_dim, H/P, W/P)
        x = x.flatten(2).transpose(0, 2, 1)  # (B, num_patches, embed_dim)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def execute(self, x):
        if not self.is_training() or self.drop_prob == 0.:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jt.rand(shape)
        random_tensor = jt.floor(random_tensor + keep_prob)
        output = x / keep_prob * random_tensor
        return output


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def _shape(self, tensor, seq_len, bsz):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)

    def execute(self, x):
        B, N, C = x.shape

        q = self.q_proj(x)
        k = self._shape(self.k_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)
        v = self._shape(self.v_proj(x), -1, B).reshape(B * self.num_heads, -1, self.head_dim)
        q = self._shape(q, N, B).reshape(B * self.num_heads, -1, self.head_dim)

        attn_weights = jt.bmm(q, k.transpose(0, 2, 1)) * self.scale
        attn_weights = nn.softmax(attn_weights, dim=-1)
        attn_probs = self.attn_drop(attn_weights)
        attn_output = jt.bmm(attn_probs, v)

        attn_output = attn_output.view(B, self.num_heads, N, self.head_dim)
        attn_output = attn_output.transpose(0, 2, 1, 3)
        attn_output = attn_output.reshape(B, N, C)

        x = self.proj(attn_output)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, config=None,
                 layer_id=None, writer=None):
        super().__init__()
        self.config = config
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.fc1 = nn.Linear(dim, mlp_hidden_dim)
        self.fc2 = nn.Linear(mlp_hidden_dim, dim)
        self.act = act_layer()
        self.mlp_drop = nn.Dropout(drop)

        if config.ffn_adapt:
            self.adapter_module = SEMAModules(self.config, layer_id=layer_id, writer=writer)
        self.layer_id = layer_id

    def execute(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        out = self.adapter_module(x)
        adapt_x = out["func_out"]

        residual = x
        x = self.mlp_drop(self.act(self.fc1(self.norm2(x))))
        x = self.drop_path(self.mlp_drop(self.fc2(x)))

        if self.config.ffn_adapt:
            if self.config.ffn_option == 'sequential':
                out = self.adapter_module(x)
                x = out["func_out"]
            elif self.config.ffn_option == 'parallel':
                x = x + adapt_x
            else:
                raise ValueError(self.config.ffn_adapt)

        x = residual + x
        out.update({"blk_out": x})
        return out


class VisionTransformer(nn.Module):
    """Vision Transformer with SEMA adapters (Jittor version)."""
    def __init__(self, global_pool=False, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=None, act_layer=None, tuning_config=None, writer=None):
        super().__init__()

        print("I'm using ViT with SEMA adapters (Jittor).")
        self.tuning_config = tuning_config
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = jt.zeros((1, 1, embed_dim))
        self.dist_token = jt.zeros((1, 1, embed_dim)) if distilled else None
        self.pos_embed = jt.zeros((1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = np.linspace(0, drop_path_rate, depth).tolist()
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, act_layer=act_layer,
                config=tuning_config, layer_id=i, writer=writer
            )
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        # Representation layer
        if representation_size and not distilled:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(OrderedDict([
                ('fc', nn.Linear(embed_dim, representation_size)),
                ('act', nn.Tanh())
            ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()

        self.global_pool = global_pool
        if self.global_pool:
            self.fc_norm = norm_layer(embed_dim)

        if tuning_config.vpt_on:
            assert tuning_config.vpt_num > 0
            self.embeddings = [jt.empty((1, self.tuning_config.vpt_num, embed_dim)) for _ in range(depth)]
            for eee in self.embeddings:
                init.xavier_uniform_(eee)

    @property
    def feature_dim(self):
        return self.out_dim

    def execute_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = jt.concat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        rd_losses = 0
        added_record = []

        for idx, blk in enumerate(self.blocks):
            if self.tuning_config.vpt_on:
                eee = self.embeddings[idx].expand(B, -1, -1)
                x = jt.concat([eee, x], dim=1)
            blk_ret = blk(x)
            x = blk_ret["blk_out"]
            rd_loss, added = blk_ret["rd_loss"], blk_ret["added"]
            rd_losses += rd_loss
            added_record.append(added)
            if self.tuning_config.vpt_on:
                x = x[:, self.tuning_config.vpt_num:, :]
            if added:
                break

        if self.global_pool:
            x = x[:, 1:, :].mean(dim=1)
            outcome = self.fc_norm(x)
        else:
            x = self.norm(x)
            outcome = x[:, 0]

        out = {"features": outcome, "rd_loss": rd_losses, "added_record": added_record}
        return out

    def execute(self, x):
        out = self.execute_features(x)
        x = out["features"]
        if self.head_dist is not None:
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            if self.is_training():
                return x, x_dist
            else:
                return (x + x_dist) / 2
        else:
            x = self.head(x)
        out.update({"features": x})
        return out


def _load_pretrained_weights(model, pretrained_path_or_url=None):
    """Load pretrained ViT weights, adapting qkv -> q/k/v and mlp naming.
    
    Uses torch to load timm weights, then converts to Jittor.
    """
    import torch
    import timm as timm_torch
    
    checkpoint_model = timm_torch.create_model("vit_base_patch16_224", pretrained=True, num_classes=0)
    state_dict = checkpoint_model.state_dict()
    
    for key in list(state_dict.keys()):
        if 'qkv.weight' in key:
            qkv_weight = state_dict.pop(key)
            q_weight = qkv_weight[:768]
            k_weight = qkv_weight[768:768*2]
            v_weight = qkv_weight[768*2:]
            state_dict[key.replace('qkv.weight', 'q_proj.weight')] = q_weight
            state_dict[key.replace('qkv.weight', 'k_proj.weight')] = k_weight
            state_dict[key.replace('qkv.weight', 'v_proj.weight')] = v_weight
        elif 'qkv.bias' in key:
            qkv_bias = state_dict.pop(key)
            q_bias = qkv_bias[:768]
            k_bias = qkv_bias[768:768*2]
            v_bias = qkv_bias[768*2:]
            state_dict[key.replace('qkv.bias', 'q_proj.bias')] = q_bias
            state_dict[key.replace('qkv.bias', 'k_proj.bias')] = k_bias
            state_dict[key.replace('qkv.bias', 'v_proj.bias')] = v_bias
    for key in list(state_dict.keys()):
        if 'mlp.fc' in key:
            fc_weight = state_dict.pop(key)
            state_dict[key.replace('mlp.', '')] = fc_weight

    # Convert torch state_dict to jittor-compatible
    jt_state_dict = {}
    for k, v in state_dict.items():
        jt_state_dict[k] = jt.array(v.cpu().numpy())

    # Load into model; collect missing keys for adapter params
    model_state = model.state_dict()
    missing_keys = []
    for k in model_state.keys():
        if k not in jt_state_dict:
            missing_keys.append(k)
        else:
            model_state[k] = jt_state_dict[k]
    
    model.load_state_dict(model_state)
    print(f"Loaded pretrained weights. Missing keys (adapter params): {missing_keys}")

    # Freeze pretrained params, leave adapter params trainable
    # Use pattern matching (robust against Jittor naming differences)
    adapter_keywords = ['adapter_module', 'adapters', 'functional', 'router', 'rd']
    frozen_count, trainable_count = 0, 0
    for name, p in model.named_parameters():
        if any(kw in name for kw in adapter_keywords):
            # Adapter/RD/Router param — explicitly make trainable
            # Jittor's load_state_dict may reset gradient state, so we must
            # call start_grad() explicitly rather than relying on default.
            p.start_grad()
            trainable_count += 1
        else:
            p.stop_grad()
            frozen_count += 1
    print(f"Frozen {frozen_count} pretrained params, kept {trainable_count} adapter params trainable.")
    
    return model, missing_keys


def vit_base_patch16_224_sema(pretrained=True, **kwargs):
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        model, _ = _load_pretrained_weights(model)
    
    model.out_dim = 768
    return model


def vit_base_patch16_224_in21k_sema(pretrained=True, **kwargs):
    """Load ViT-B/16 pretrained on ImageNet-21k."""
    model = VisionTransformer(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4,
        qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    
    if pretrained:
        import torch
        import timm as timm_torch
        checkpoint_model = timm_torch.create_model("vit_base_patch16_224_in21k", pretrained=True, num_classes=0)
        state_dict = checkpoint_model.state_dict()
        
        for key in list(state_dict.keys()):
            if 'qkv.weight' in key:
                qkv_weight = state_dict.pop(key)
                state_dict[key.replace('qkv.weight', 'q_proj.weight')] = qkv_weight[:768]
                state_dict[key.replace('qkv.weight', 'k_proj.weight')] = qkv_weight[768:768*2]
                state_dict[key.replace('qkv.weight', 'v_proj.weight')] = qkv_weight[768*2:]
            elif 'qkv.bias' in key:
                qkv_bias = state_dict.pop(key)
                state_dict[key.replace('qkv.bias', 'q_proj.bias')] = qkv_bias[:768]
                state_dict[key.replace('qkv.bias', 'k_proj.bias')] = qkv_bias[768:768*2]
                state_dict[key.replace('qkv.bias', 'v_proj.bias')] = qkv_bias[768*2:]
        for key in list(state_dict.keys()):
            if 'mlp.fc' in key:
                fc_weight = state_dict.pop(key)
                state_dict[key.replace('mlp.', '')] = fc_weight

        jt_state_dict = {}
        for k, v in state_dict.items():
            jt_state_dict[k] = jt.array(v.cpu().numpy())

        model_state = model.state_dict()
        missing_keys = []
        for k in model_state.keys():
            if k not in jt_state_dict:
                missing_keys.append(k)
            else:
                model_state[k] = jt_state_dict[k]
        model.load_state_dict(model_state)

        adapter_keywords = ['adapter_module', 'adapters', 'functional', 'router', 'rd']
        for name, p in model.named_parameters():
            if any(kw in name for kw in adapter_keywords):
                p.start_grad()
            else:
                p.stop_grad()
    
    model.out_dim = 768
    return model

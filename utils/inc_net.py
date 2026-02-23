"""
Jittor version of incremental network wrappers.
Only includes SEMAVitNet (the SEMA model wrapper).
"""
import copy
import logging
import jittor as jt
from jittor import nn


def get_backbone(args, pretrained=False):
    name = args["backbone_type"].lower()
    
    if '_adapter' in name and args["model_name"] == "sema":
        from backbone import vit_sema
        from easydict import EasyDict
        ffn_num = args["ffn_num"]
        tuning_config = EasyDict(
            ffn_adapt=True,
            ffn_option="parallel",
            ffn_adapter_layernorm_option="none",
            ffn_adapter_init_option="lora",
            ffn_adapter_scalar="0.1",
            ffn_num=ffn_num,
            ffn_adapter_type=args["ffn_adapter_type"],
            d_model=768,
            attn_bn=ffn_num,  # used as default bottleneck
            vpt_on=False,
            vpt_num=0,
            exp_threshold=args["exp_threshold"],
            adapt_start_layer=args["adapt_start_layer"],
            adapt_end_layer=args["adapt_end_layer"],
            rd_dim=args["rd_dim"],
            buffer_size=args["buffer_size"],
            use_topk_routing=args.get("use_topk_routing", False),
            top_k_adapters=args.get("top_k_adapters", 1),
        )
        if name == "pretrained_vit_b16_224_adapter":
            model = vit_sema.vit_base_patch16_224_sema(
                num_classes=0, global_pool=False, drop_path_rate=0.0,
                tuning_config=tuning_config)
            model.out_dim = 768
        elif name == "pretrained_vit_b16_224_in21k_adapter":
            model = vit_sema.vit_base_patch16_224_in21k_sema(
                num_classes=0, global_pool=False, drop_path_rate=0.0,
                tuning_config=tuning_config)
            model.out_dim = 768
        else:
            raise NotImplementedError("Unknown type {}".format(name))
        model.eval()
        return model
    else:
        raise NotImplementedError(
            "Only SEMA adapter backbone is supported in Jittor version. Got: {}".format(name))


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()
        print('BaseNet initialization...')
        self.backbone = get_backbone(args, pretrained)
        print('After BaseNet initialization.')
        self.fc = None
        self._device = args["device"][0]
        self.model_type = 'vit'

    @property
    def feature_dim(self):
        return self.backbone.out_dim

    def extract_vector(self, x):
        return self.backbone(x)

    def execute(self, x):
        x = self.backbone(x)
        out = self.fc(x)
        out.update({"features": x})
        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.stop_grad()
        self.eval()
        return self


class SEMAVitNet(BaseNet):
    def __init__(self, args, pretrained):
        super().__init__(args, pretrained)
        self.fc = None
        self.args = args

    def extract_vector(self, x):
        return self.backbone(x)

    def execute(self, x):
        out = self.backbone(x)
        x = out["features"]
        out.update({"logits": self.fc(x)})
        return out

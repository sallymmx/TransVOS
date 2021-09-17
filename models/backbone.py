"""
Backbone modules.
"""

import torch
import torch.nn.functional as F
from torch import nn
from collections import OrderedDict
from typing import Dict, List
from utils.misc import NestedTensor, is_main_process
from models.position_encoding import build_position_encoding
from models import resnet as resnet_module


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.
    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()  # rsqrt(x): 1/sqrt(x), r: reciprocal
        bias = b - rm * scale
        return x * scale + bias


class IntermediateLayerGetter(nn.ModuleDict):
    """
    Module wrapper that returns intermediate layers from a model
    It has a strong assumption that the modules have been registered
    into the model in the same order as they are used.
    This means that one should **not** reuse the same nn.Module
    twice in the forward if you want this to work.
    Additionally, it is only able to query submodules that are directly
    assigned to the model. So if `model` is passed, `model.feature1` can
    be returned, but not `model.feature1.layer2`.
    Arguments:
        model (nn.Module): model on which we will extract the features
        return_layers (Dict[name, new_name]): a dict containing the names
            of the modules for which the activations will be returned as
            the key of the dict, and the value of the dict is the name
            of the returned activation (which the user can specify).
    """
    
    def __init__(self, model, return_layers):
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model")
 
        orig_return_layers = return_layers
        return_layers = {k: v for k, v in return_layers.items()}
        layers = OrderedDict()
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break
 
        super(IntermediateLayerGetter, self).__init__(layers)
        self.return_layers = orig_return_layers
        self.fg_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bg_conv = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
 
    def forward(self, x):
        out = OrderedDict()
        if x.shape[1] == 4:
            x, m = x[:,:3], x[:,3:4]
            fg, bg = m, 1-m
            for name, module in self.named_children():
                if name == 'conv1':
                    x = module(x) + self.fg_conv(fg) + self.bg_conv(bg)
                elif name not in ['fg_conv', 'bg_conv']:
                    x = module(x)
                if name in self.return_layers:
                    out_name = self.return_layers[name]
                    out[out_name] = x
        else:
            for name, module in self.named_children():
                if name not in ['fg_conv', 'bg_conv']:
                    x = module(x)
                    if name in self.return_layers:
                        out_name = self.return_layers[name]
                        out[out_name] = x
        return out


class BackboneBase(nn.Module):

    def __init__(self, backbone: nn.Module, train_backbone: bool, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if not train_backbone or 'layer2' not in name and 'layer3' not in name:
                parameter.requires_grad_(False)  # here should allow users to specify which layers to freeze !
                # print('freeze %s'%name)
        if return_interm_layers:
            return_layers = {"layer1": "0", "layer2": "1", "layer3": "2"}  # stride = 4, 8, 16
            self.strides = [4, 8, 16]
            self.num_channels = num_channels
        else:
            return_layers = {'layer3': "0"}  # stride = 16
            self.strides = [16]
            self.num_channels = [num_channels[-1]]
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)  # method in torchvision

    def forward(self, tensor_list: NestedTensor):
        xs = self.body(tensor_list.tensors)
        out: Dict[str, NestedTensor] = {}
        for name, x in xs.items():
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
            out[name] = NestedTensor(x, mask)
        return out


class Backbone(BackboneBase):
    """ResNet backbone with frozen BatchNorm."""
    def __init__(self, name: str,
                 train_backbone: bool,
                 return_interm_layers: bool,
                 dilation: bool,
                 freeze_bn: bool):
        norm_layer = FrozenBatchNorm2d if freeze_bn else nn.BatchNorm2d
        # here is different from the original DETR because we use feature from block3
        backbone = getattr(resnet_module, name)(
            replace_stride_with_dilation=[False, dilation, False],
            pretrained=is_main_process(), norm_layer=norm_layer, last_layer='layer3')
        num_channels = [64, 128, 256] if name in ('resnet18', 'resnet34') else [256, 512, 1024]
        super().__init__(backbone, train_backbone, num_channels, return_interm_layers)


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor, mode=None):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_backbone(cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.TRAIN.BACKBONE_MULTIPLIER > 0
    return_interm_layers = cfg.MODEL.PREDICT_MASK
    backbone = Backbone(cfg.MODEL.BACKBONE.TYPE, train_backbone, return_interm_layers,
                        cfg.MODEL.BACKBONE.DILATION, cfg.TRAIN.FREEZE_BACKBONE_BN)
    model = Joiner(backbone, position_embedding)
    return model
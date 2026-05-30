""" MobileNet V3 / V4 Common Modules

Compatible with MobileNetV4 architectures.
"""
from functools import partial
from typing import Callable, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import SelectAdaptivePool2d, Linear, LayerType, PadType, create_conv2d, get_norm_act_layer
from timm.models._builder import build_model_with_cfg, pretrained_cfg_for_features
from models.blocks import SqueezeExcite
from models.model_utils import BlockArgs, EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer
from timm.models._features import FeatureInfo, FeatureHooks, feature_take_indices
from timm.models._manipulate import checkpoint_seq
from timm.models._registry import generate_default_cfgs, register_model
from models.extra_attention_block import MultiScaleAttentionGate

__all__ = ['MobileNetV4', 'MobileNetV4Features']


class MobileNetV4(nn.Module):
    """ MobiletNet-V3 / V4

    Based on EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3/v4 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.
    """

    def __init__(
            self,
            block_args: BlockArgs,
            num_classes: int = 1000,
            in_chans: int = 3,
            stem_size: int = 16,
            fix_stem: bool = False,
            num_features: int = 1280,
            head_bias: bool = True,
            head_norm: bool = False,
            pad_type: str = '',
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            se_from_exp: bool = True,
            round_chs_fn: Callable = round_channels,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = None,
            global_pool: str = 'avg',
            extra_attention_block: bool = False,
            **kwargs
    ):
        super(MobileNetV4, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.grad_checkpointing = False
        self.extra_attention_block = MultiScaleAttentionGate(960) if extra_attention_block else None

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        self.stage_ends = [f['stage'] for f in self.feature_info]
        self.num_features = builder.in_chs  # features of last stage, output of forward_features()
        self.head_hidden_size = num_features  # features of conv_head, pre_logits output

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = self.num_features * self.global_pool.feat_mult()
        if head_norm:
            # mobilenet-v4 post-pooling PW conv is followed by a norm+act layer
            self.conv_head = create_conv2d(num_pooled_chs, self.head_hidden_size, 1, padding=pad_type)  # never bias
            self.norm_head = norm_act_layer(self.head_hidden_size)
            self.act2 = nn.Identity()
        else:
            # mobilenet-v3 and others only have an activation after final PW conv
            self.conv_head = create_conv2d(num_pooled_chs, self.head_hidden_size, 1, padding=pad_type, bias=head_bias)
            self.norm_head = nn.Identity()
            self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)

    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.norm_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)

    @torch.jit.ignore
    def group_matcher(self, coarse: bool = False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    @torch.jit.ignore
    def get_classifier(self) -> nn.Module:
        return self.classifier

    def reset_classifier(self, num_classes: int, global_pool: str = 'avg'):
        self.num_classes = num_classes
        # NOTE: cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.head_hidden_size, num_classes) if num_classes > 0 else nn.Identity()

    def forward_intermediates(
            self,
            x: torch.Tensor,
            indices: Optional[Union[int, List[int]]] = None,
            norm: bool = False,
            stop_early: bool = False,
            output_fmt: str = 'NCHW',
            intermediates_only: bool = False,
            extra_blocks: bool = False,
    ) -> Union[List[torch.Tensor], Tuple[torch.Tensor, List[torch.Tensor]]]:
        assert output_fmt in ('NCHW',), 'Output shape must be NCHW.'
        if stop_early:
            assert intermediates_only, 'Must use intermediates_only for early stopping.'
        intermediates = []
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            take_indices = [self.stage_ends[i] for i in take_indices]
            max_index = self.stage_ends[max_index]

        # forward pass
        feat_idx = 0  # stem is index 0
        x = self.conv_stem(x)
        x = self.bn1(x)
        if feat_idx in take_indices:
            intermediates.append(x)

        if torch.jit.is_scripting() or not stop_early:  # can't slice blocks in torchscript
            blocks = self.blocks
        else:
            blocks = self.blocks[:max_index]
        for blk in blocks:
            feat_idx += 1
            x = blk(x)
            if feat_idx in take_indices:
                intermediates.append(x)

        if intermediates_only:
            return intermediates

        return x, intermediates

    def prune_intermediate_layers(
            self,
            indices: Union[int, List[int]] = 1,
            prune_norm: bool = False,
            prune_head: bool = True,
            extra_blocks: bool = False,
    ):
        """ Prune layers not required for specified intermediates.
        """
        if extra_blocks:
            take_indices, max_index = feature_take_indices(len(self.blocks) + 1, indices)
        else:
            take_indices, max_index = feature_take_indices(len(self.stage_ends), indices)
            max_index = self.stage_ends[max_index]
        self.blocks = self.blocks[:max_index]  # truncate blocks w/ stem as idx 0
        if max_index < len(self.blocks):
            self.conv_head = nn.Identity()
            self.norm_head = nn.Identity()
        if prune_head:
            self.conv_head = nn.Identity()
            self.norm_head = nn.Identity()
            self.reset_classifier(0, '')
        return take_indices

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        if self.extra_attention_block:
            x = self.extra_attention_block(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.norm_head(x)
        x = self.act2(x)
        x = self.flatten(x)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        if pre_logits:
            return x
        return self.classifier(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


class MobileNetV4Features(nn.Module):
    """ MobileNetV4 Feature Extractor """

    def __init__(
            self,
            block_args: BlockArgs,
            out_indices: Tuple[int, ...] = (0, 1, 2, 3, 4),
            feature_location: str = 'bottleneck',
            in_chans: int = 3,
            stem_size: int = 16,
            fix_stem: bool = False,
            output_stride: int = 32,
            pad_type: PadType = '',
            round_chs_fn: Callable = round_channels,
            se_from_exp: bool = True,
            act_layer: Optional[LayerType] = None,
            norm_layer: Optional[LayerType] = None,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[LayerType] = None,
            drop_rate: float = 0.,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = None,
    ):
        super(MobileNetV4Features, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        se_layer = se_layer or SqueezeExcite
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_layer(stem_size)
        self.act1 = act_layer(inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=output_stride,
            pad_type=pad_type,
            round_chs_fn=round_chs_fn,
            se_from_exp=se_from_exp,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            drop_path_rate=drop_path_rate,
            layer_scale_init_value=layer_scale_init_value,
            feature_location=feature_location,
        )
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = FeatureInfo(builder.features, out_indices)
        self._stage_out_idx = {f['stage']: f['index'] for f in self.feature_info.get_dicts()}

        efficientnet_init_weights(self)

        # Register feature extraction hooks with FeatureHooks helper
        self.feature_hooks = None
        if feature_location != 'bottleneck':
            hooks = self.feature_info.get_dicts(keys=('module', 'hook_type'))
            self.feature_hooks = FeatureHooks(hooks, self.named_modules())

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_checkpointing = enable

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act1(x)
        if self.feature_hooks is None:
            features = []
            if 0 in self._stage_out_idx:
                features.append(x)  # add stem out
            for i, b in enumerate(self.blocks):
                if self.grad_checkpointing and not torch.jit.is_scripting():
                    x = checkpoint(b, x)
                else:
                    x = b(x)
                if i + 1 in self._stage_out_idx:
                    features.append(x)
            return features
        else:
            self.blocks(x)
            out = self.feature_hooks.get_output(x.device)
            return list(out.values())


def _create_mnv4(variant: str, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    features_mode = ''
    model_cls = MobileNetV4
    kwargs_filter = None
    if kwargs.pop('features_only', False):
        if 'feature_cfg' in kwargs or 'feature_cls' in kwargs:
            features_mode = 'cfg'
        else:
            kwargs_filter = ('num_classes', 'num_features', 'head_conv', 'head_bias', 'head_norm', 'global_pool')
            model_cls = MobileNetV4Features
            features_mode = 'cls'

    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        features_only=features_mode == 'cfg',
        pretrained_strict=features_mode != 'cls',
        kwargs_filter=kwargs_filter,
        **kwargs,
    )
    if features_mode == 'cls':
        model.default_cfg = pretrained_cfg_for_features(model.default_cfg)
    return model


def _cfg(url: str = '', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }

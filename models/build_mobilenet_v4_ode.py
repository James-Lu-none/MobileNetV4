""" MobileNet V4 ODE Models

Compatible with MobileNetV4 architectures.
"""
from functools import partial
import torch.nn as nn

from timm.models._registry import register_model
from models.model_utils import decode_arch_def, round_channels_to_perfect_square, resolve_bn_args, resolve_act_layer
from .build_mobilenet_v4_common import MobileNetV4, _create_mnv4


def _gen_ode_mobilenet_v4(
        variant: str, channel_multiplier: float = 1.0, group_size=None, pretrained=False,
        pretrained_cfg=None, pretrained_cfg_overlay=None, ode_num_steps: int = 10, **kwargs,
) -> MobileNetV4:
    """Creates a MobileNet-V4 model with ODE Conv2d blocks (ouir)."""
    num_features = 1296
    if 'hybrid' in variant:
        layer_scale_init_value = 1e-5
        if 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48'  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'ouir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'ouir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'ouir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'ouir_r1_a0_k0_s1_e2_c160',  # FFN
                    'ouir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'ouir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'ouir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'ouir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'ouir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'ouir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'ouir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'ouir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'ouir_r1_a0_k0_s1_e2_c256',  # FFN
                    'ouir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'ouir_r1_a0_k0_s1_e2_c256',  # FFN
                    'ouir_r1_a0_k0_s1_e4_c256',  # FFN
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'ouir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'ouir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960'  # Conv
                ],
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'gelu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'ouir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'ouir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'ouir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'ouir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'ouir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'ouir_r2_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'ouir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'ouir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'ouir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'ouir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'ouir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'ouir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'ouir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'ouir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        else:
            assert False, f'Unknown variant {variant}.'
    else:
        layer_scale_init_value = None
        if 'small' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'cn_r1_k3_s2_e1_c32',  # Conv
                    'cn_r1_k1_s1_e1_c32',  # Conv
                ],
                # stage 1, 56x56 in
                [
                    'cn_r1_k3_s2_e1_c96',  # Conv
                    'cn_r1_k1_s1_e1_c64',  # Conv
                ],
                # stage 2, 28x28 in
                [
                    'ouir_r1_a5_k5_s2_e3_c96',  # ExtraDW
                    'ouir_r4_a0_k3_s1_e2_c96',  # IR
                    'ouir_r1_a3_k0_s1_e4_c96',  # ConvNeXt
                ],
                # stage 3, 14x14 in
                [
                    'ouir_r1_a3_k3_s2_e6_c128',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c128',  # ExtraDW
                    'ouir_r1_a0_k5_s1_e4_c128',  # IR
                    'ouir_r1_a0_k5_s1_e3_c128',  # IR
                    'ouir_r2_a0_k3_s1_e4_c128',  # IR
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        elif 'medium' in variant:
            stem_size = 32
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'ouir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'ouir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'ouir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'ouir_r2_a3_k3_s1_e4_c160',  # ExtraDW
                    'ouir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'ouir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'ouir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'ouir_r1_a0_k0_s1_e2_c160',  # ExtraDW
                    'ouir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'ouir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'ouir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'ouir_r1_a0_k0_s1_e4_c256',  # FFN
                    'ouir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'ouir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'ouir_r2_a0_k0_s1_e4_c256',  # FFN
                    'ouir_r1_a5_k0_s1_e2_c256',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        elif 'large' in variant:
            stem_size = 24
            act_layer = resolve_act_layer(kwargs, 'relu')
            arch_def = [
                # stage 0, 112x112 in
                [
                    'er_r1_k3_s2_e4_c48',  # FusedIB (EdgeResidual)
                ],
                # stage 1, 56x56 in
                [
                    'ouir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'ouir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'ouir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'ouir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'ouir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'ouir_r5_a5_k3_s1_e4_c192',  # ExtraDW
                    'ouir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'ouir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'ouir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'ouir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'ouir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'ouir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'ouir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'ouir_r3_a5_k0_s1_e4_c512',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960',  # Conv
                ],
            ]
        else:
            assert False, f'Unknown variant {variant}.'

    model_kwargs = dict(
        block_args=decode_arch_def(arch_def, group_size=group_size),
        head_bias=False,
        head_norm=True,
        num_features=num_features,
        stem_size=stem_size,
        fix_stem=channel_multiplier < 1.0,
        round_chs_fn=partial(round_channels_to_perfect_square, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        layer_scale_init_value=layer_scale_init_value,
        **kwargs,
    )
    model = _create_mnv4(variant, pretrained, **model_kwargs)
    return model


@register_model
def mobilenetv4_ode_conv_small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 ODE Conv Small """
    model = _gen_ode_mobilenet_v4('mobilenetv4_ode_conv_small', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_ode_conv_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 ODE Conv Medium """
    model = _gen_ode_mobilenet_v4('mobilenetv4_ode_conv_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_ode_conv_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 ODE Conv Large """
    model = _gen_ode_mobilenet_v4('mobilenetv4_ode_conv_large', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_ode_hybrid_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 ODE Hybrid Medium """
    model = _gen_ode_mobilenet_v4('mobilenetv4_ode_hybrid_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_ode_hybrid_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 ODE Hybrid Large """
    model = _gen_ode_mobilenet_v4('mobilenetv4_ode_hybrid_large', 1.0, pretrained=pretrained, **kwargs)
    return model

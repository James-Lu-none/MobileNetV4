""" MobileNet V4 Base Models

Compatible with MobileNetV4 architectures.
"""
from functools import partial
import torch.nn as nn

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models._registry import generate_default_cfgs, register_model
from models.model_utils import decode_arch_def, round_channels, resolve_bn_args, resolve_act_layer
from .build_mobilenet_v4_common import MobileNetV4, _create_mnv4, _cfg


def _gen_mobilenet_v4_rw(
        variant: str, channel_multiplier: float = 1.0, pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs
) -> MobileNetV4:
    """Creates a MobileNet-V4 model.
    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    arch_def = [
        # stage 0, 112x112 in
        ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
        # stage 1, 112x112 in
        ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
        # stage 2, 56x56 in
        ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
        # stage 3, 28x28 in
        ['ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80', 'ir_r2_k3_s1_e2.3_c80'],  # hard-swish
        # stage 4, 14x14in
        ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
        # stage 5, 14x14in
        ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
        # stage 6, 7x7 in
        ['cn_r1_k1_s1_c960'],  # hard-swish
    ]
    model_kwargs = dict(
        block_args=decode_arch_def(arch_def),
        head_bias=False,
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=resolve_act_layer(kwargs, 'hard_swish'),
        se_layer=partial(SqueezeExcite=None, gate_layer='hard_sigmoid') if 'se_layer' not in kwargs else kwargs.pop('se_layer'),  # SqueezeExcite inside blocks
        **kwargs,
    )
    # If se_layer is not customized, use default
    if 'se_layer' not in model_kwargs or model_kwargs['se_layer'] is None:
        from models.blocks import SqueezeExcite
        model_kwargs['se_layer'] = partial(SqueezeExcite, gate_layer='hard_sigmoid')
    model = _create_mnv4(variant, pretrained, **model_kwargs)
    return model


def _gen_mobilenet_v4(
        variant: str, channel_multiplier: float = 1.0, group_size=None, pretrained=False,
        pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs,
) -> MobileNetV4:
    """Creates a MobileNet-V4 model.

    Args:
      channel_multiplier: multiplier to number of channels per layer.
    """
    num_features = 1280
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
                    'uir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'uir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c160',  # FFN
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'mqa_r1_k3_h4_s1_v2_d64_c160',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c256',  # FFN
                    'uir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e2_c256',  # FFN
                    'uir_r1_a0_k0_s1_e4_c256',  # FFN
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'mqa_r1_k3_h4_s1_d64_c256',  # MQA
                    'uir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                    'mqa_r1_k3_h4_s1_d64_c256', # MQA
                    'uir_r1_a5_k0_s1_e4_c256',  # ConvNeXt
                ],
                # stage 4, 7x7 in
                [
                    'cn_r1_k1_s1_c960' # Conv
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
                    'uir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'uir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'uir_r2_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a5_k3_s1_e4_c192',  # ExtraDW
                    'mqa_r1_k3_h8_s1_v2_d48_c192',  # MQA w/ KV downsample
                    'uir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'mqa_r1_k3_h8_s1_d64_c512',  # MQA
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
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
                    'uir_r1_a5_k5_s2_e3_c96',  # ExtraDW
                    'uir_r4_a0_k3_s1_e2_c96',  # IR
                    'uir_r1_a3_k0_s1_e4_c96',  # ConvNeXt
                ],
                # stage 3, 14x14 in
                [
                    'uir_r1_a3_k3_s2_e6_c128',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c128',  # ExtraDW
                    'uir_r1_a0_k5_s1_e4_c128',  # IR
                    'uir_r1_a0_k5_s1_e3_c128',  # IR
                    'uir_r2_a0_k3_s1_e4_c128',  # IR
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
                    'uir_r1_a3_k5_s2_e4_c80',  # ExtraDW
                    'uir_r1_a3_k3_s1_e2_c80',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e6_c160',  # ExtraDW
                    'uir_r2_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c160',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                    'uir_r1_a0_k0_s1_e2_c160',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c160',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r1_a5_k5_s2_e6_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a3_k5_s1_e4_c256',  # ExtraDW
                    'uir_r1_a0_k0_s1_e4_c256',  # FFN
                    'uir_r1_a3_k0_s1_e4_c256',  # ConvNeXt
                    'uir_r1_a3_k5_s1_e2_c256',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c256',  # ExtraDW
                    'uir_r2_a0_k0_s1_e4_c256',  # FFN
                    'uir_r1_a5_k0_s1_e2_c256',  # ConvNeXt
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
                    'uir_r1_a3_k5_s2_e4_c96',  # ExtraDW
                    'uir_r1_a3_k3_s1_e4_c96',  # ExtraDW
                ],
                # stage 2, 28x28 in
                [
                    'uir_r1_a3_k5_s2_e4_c192',  # ExtraDW
                    'uir_r3_a3_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k5_s1_e4_c192',  # ExtraDW
                    'uir_r5_a5_k3_s1_e4_c192',  # ExtraDW
                    'uir_r1_a3_k0_s1_e4_c192',  # ConvNeXt
                ],
                # stage 3, 14x14in
                [
                    'uir_r4_a5_k5_s2_e4_c512',  # ExtraDW
                    'uir_r1_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r2_a5_k0_s1_e4_c512',  # ConvNeXt
                    'uir_r1_a5_k3_s1_e4_c512',  # ExtraDW
                    'uir_r1_a5_k5_s1_e4_c512',  # ExtraDW
                    'uir_r3_a5_k0_s1_e4_c512',  # ConvNeXt
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
        round_chs_fn=partial(round_channels, multiplier=channel_multiplier),
        norm_layer=partial(nn.BatchNorm2d, **resolve_bn_args(kwargs)),
        act_layer=act_layer,
        layer_scale_init_value=layer_scale_init_value,
        **kwargs,
    )
    model = _create_mnv4(variant, pretrained, **model_kwargs)
    return model


default_cfgs = generate_default_cfgs({
    'mobilenetv4_conv_small_035.untrained': _cfg(
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small_050.e3000_r224_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e2400_r224_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e1200_r224_in1k': _cfg(
        hf_hub_id='timm/',
        test_input_size=(3, 256, 256), test_crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_small.e3600_r256_in1k': _cfg(
        hf_hub_id='timm/',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD,
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95,
        test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e500_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_conv_medium.e250_r384_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e180_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e180_ad_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_medium.e250_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_conv_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_large.e500_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),

    'mobilenetv4_hybrid_medium.e200_r256_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.ix_e550_r256_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.ix_e550_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium.e200_r256_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 256, 256), pool_size=(8, 8),
        crop_pct=0.95, test_input_size=(3, 320, 320), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_large.ix_e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),

    # experimental
    'mobilenetv4_conv_aa_medium.untrained': _cfg(
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_conv_blur_medium.e500_r224_in1k': _cfg(
        hf_hub_id='timm/',
        crop_pct=0.95, test_input_size=(3, 256, 256), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r448_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 448, 448), pool_size=(14, 14),
        crop_pct=0.95, test_input_size=(3, 544, 544), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r384_in12k_ft_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 480, 480), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e600_r384_in1k': _cfg(
        hf_hub_id='timm/',
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 480, 480), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_conv_aa_large.e230_r384_in12k': _cfg(
        hf_hub_id='timm/',
        num_classes=11821,
        input_size=(3, 384, 384), pool_size=(12, 12),
        crop_pct=0.95, test_input_size=(3, 448, 448), test_crop_pct=1.0, interpolation='bicubic'),
    'mobilenetv4_hybrid_medium_075.untrained': _cfg(
        crop_pct=0.95, interpolation='bicubic'),
    'mobilenetv4_hybrid_large_075.untrained': _cfg(
        input_size=(3, 256, 256), pool_size=(8, 8), crop_pct=0.95, interpolation='bicubic'),
})


@register_model
def mobilenetv4_conv_small_035(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small_035', 0.35, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_small_050(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small_050', 0.50, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_small(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_small', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 """
    model = _gen_mobilenet_v4('mobilenetv4_conv_large', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 Hybrid """
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_medium', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 Hybrid"""
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_large', 1.0, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_conv_aa_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 w/ AvgPool AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_aa_medium', 1.0, pretrained=pretrained, aa_layer='avg', **kwargs)
    return model


@register_model
def mobilenetv4_conv_blur_medium(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 Conv w/ Blur AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_blur_medium', 1.0, pretrained=pretrained, aa_layer='blurpc', **kwargs)
    return model


@register_model
def mobilenetv4_conv_aa_large(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 w/ AvgPool AA """
    model = _gen_mobilenet_v4('mobilenetv4_conv_aa_large', 1.0, pretrained=pretrained, aa_layer='avg', **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_medium_075(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 Hybrid """
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_medium_075', 0.75, pretrained=pretrained, **kwargs)
    return model


@register_model
def mobilenetv4_hybrid_large_075(pretrained=False, pretrained_cfg=None, pretrained_cfg_overlay=None, **kwargs) -> MobileNetV4:
    """ MobileNet V4 Hybrid"""
    model = _gen_mobilenet_v4('mobilenetv4_hybrid_large_075', 0.75, pretrained=pretrained, **kwargs)
    return model

from typing import Dict, Optional, Type
import torch.nn as nn
from timm.layers import create_conv2d, DropPath, create_aa, get_norm_act_layer, LayerType
from .blocks_common import make_divisible, num_groups, LayerScale2d, ModuleType
from .blocks_dynamic import Dynamic_conv2d
from .blocks_ode import ChannelwiseODESolver

__all__ = ['DynamicODEUniversalInvertedResidual']


class DynamicODEUniversalInvertedResidual(nn.Module):
    """UIB block with Dynamic_conv2d for DW layers and ChannelwiseODESolver for PW layers."""

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size_start: int = 0,
            dw_kernel_size_mid: int = 3,
            dw_kernel_size_end: int = 0,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
            layer_scale_init_value: Optional[float] = 1e-5,
            dynamic_K: int = 4,
            dynamic_temperature: int = 34,
            dynamic_ratio: float = 0.25,
            dynamic_dw: bool = True,
            dynamic_pw: bool = True,
            ode_num_steps: int = 10,
    ):
        super(DynamicODEUniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_layer_no_act = get_norm_act_layer(norm_layer, None)

        def make_dw(in_c, out_c, k_size, stride_c, groups_c, use_act=True, apply_aa=False):
            padding = pad_type if isinstance(pad_type, int) else (k_size - 1) // 2 * dilation
            use_aa = apply_aa and aa_layer is not None and stride_c > 1
            eff_stride = 1 if use_aa else stride_c
            if dynamic_dw:
                conv = Dynamic_conv2d(
                    in_planes=in_c, out_planes=out_c, kernel_size=k_size,
                    ratio=dynamic_ratio, stride=eff_stride, padding=padding,
                    dilation=dilation, groups=groups_c, bias=False, K=dynamic_K, temperature=dynamic_temperature,
                )
            else:
                conv = create_conv2d(
                    in_c, out_c, kernel_size=k_size,
                    stride=eff_stride, padding=padding,
                    dilation=dilation, groups=groups_c, bias=False, **conv_kwargs,
                )
            norm = norm_act_layer(out_c, inplace=True) if use_act else norm_layer_no_act(out_c)
            if use_aa:
                aa = create_aa(aa_layer, channels=out_c, stride=stride_c, enable=True)
                return nn.Sequential(conv, norm, aa)
            return nn.Sequential(conv, norm)

        def make_pw(in_c, out_c, k_size, stride_c, groups_c, use_act=True):
            padding = pad_type if isinstance(pad_type, int) else (k_size - 1) // 2 * dilation
            conv = ChannelwiseODESolver(
                in_channels=in_c, out_channels=out_c, num_layers=ode_num_steps
            )
            norm = norm_act_layer(out_c, inplace=True) if use_act else norm_layer_no_act(out_c)
            return nn.Sequential(conv, norm)

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = make_dw(in_chs, in_chs, dw_kernel_size_start, dw_start_stride, dw_start_groups, use_act=False, apply_aa=True)
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        mid_chs = int(round(mid_chs ** 0.5)) ** 2
        if mid_chs != in_chs:
            self.pw_exp = make_pw(in_chs, mid_chs, 1, 1, 1, use_act=True)
        else:
            self.pw_exp = nn.Identity()

        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = make_dw(mid_chs, mid_chs, dw_kernel_size_mid, stride, groups, use_act=True, apply_aa=True)
        else:
            self.dw_mid = nn.Identity()

        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.pw_proj = make_pw(mid_chs, out_chs, 1, 1, 1, use_act=False)

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = make_dw(out_chs, out_chs, dw_kernel_size_end, dw_end_stride, dw_end_groups, use_act=False, apply_aa=False)
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':
            conv = self.pw_proj[0]
            num_chs = getattr(conv, 'in_planes', getattr(conv, 'in_channels', None))
            return dict(module='pw_proj.0', hook_type='forward_pre', num_chs=num_chs)
        else:
            conv = self.pw_proj[0]
            num_chs = getattr(conv, 'out_planes', getattr(conv, 'out_channels', None))
            return dict(module='', num_chs=num_chs)

    def forward(self, x):
        shortcut = x
        x = self.dw_start(x)
        x = self.pw_exp(x)
        x = self.dw_mid(x)
        x = self.se(x)
        x = self.pw_proj(x)
        x = self.dw_end(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x

    def update_temperature(self):
        for m in self.modules():
            if isinstance(m, Dynamic_conv2d):
                m.update_temperature()

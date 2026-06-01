import math
from typing import Dict, Optional, Type
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import create_conv2d, DropPath, create_aa, get_norm_act_layer, LayerType, ConvNormAct
from .blocks_common import make_divisible, num_groups, LayerScale2d, ModuleType

__all__ = ['ChannelwiseODESolver', 'ODEUniversalInvertedResidual']


if hasattr(torch, '_dynamo') and hasattr(torch._dynamo, 'config'):
    torch._dynamo.config.recompile_limit = max(torch._dynamo.config.recompile_limit, 128)


class ChannelwiseODESolver(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=10):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = num_layers
        self.cin_sqrt = int(in_channels ** 0.5)
        self.cout_sqrt = int(out_channels ** 0.5)
        
        assert self.cin_sqrt ** 2 == in_channels, f"in_channels {in_channels} must be a perfect square"
        assert self.cout_sqrt ** 2 == out_channels, f"out_channels {out_channels} must be a perfect square"

        # originally, batch normalized is used with a value i where i*i=H*W, but i choose to use layer norm here
        # since layer norm is more suitable for variable batch size 
        self.norm_relu = nn.Sequential(
            nn.LayerNorm(self.out_channels), 
            nn.ReLU6()
        )
        
        self.epsilon = 0.1 / num_layers

        # init phi and delta t
        self.delta_t = nn.Parameter(torch.empty(num_layers, 1).uniform_(1e-4, 1))
        self.phi = nn.Parameter(torch.empty(num_layers, 1, 1, self.cout_sqrt, self.cout_sqrt))
        torch.nn.init.normal_(self.phi, mean=0, std=0.1)
        torch.nn.init.normal_(self.delta_t, mean=0.4, std=0.005)

        # Compile the inner forward logic if torch.compile is supported
        if hasattr(torch, 'compile'):
            try:
                self._compiled_forward = torch.compile(self._forward, dynamic=False)
            except Exception:
                self._compiled_forward = self._forward
        else:
            self._compiled_forward = self._forward

    def feature_reshape(self, x, b, c):
        x_reshaped = x.view(b, c, -1)  # (b, Cin, H*W)
        x_reshaped = x_reshaped.permute(0, 2, 1).contiguous()  # (b, H*W, Cin)
        # print(self.cin_sqrt,self.cout_sqrt,x.shape)
        if self.cin_sqrt != self.cout_sqrt:
            # (b, H*W, sqrt(Cin), sqrt(Cin))
            x_reshaped = x_reshaped.view(b, -1, self.cin_sqrt, self.cin_sqrt)
            # (b, H*W, sqrt(Cout), sqrt(Cout))
            x_expanded = F.interpolate(x_reshaped, size=(self.cout_sqrt, self.cout_sqrt), mode='bilinear', align_corners=False)
        else:
            x_expanded = x_reshaped.view(b, -1, self.cout_sqrt, self.cout_sqrt)
        return x_expanded
    
    def _forward(self, x):
        B, C, H, W = x.size()
        # (b, Cin, H, W)
        
        # (B, H*W, sqrt(Cout), sqrt(Cout))
        x_expanded = self.feature_reshape(x, B, C)  
        y0 = x_expanded

        delta_t = torch.maximum(torch.tensor(self.epsilon, device=x.device), self.delta_t)
        
        for layer in range(self.num_layers):
            dt = delta_t[layer]
            result = torch.matmul(y0, self.phi[layer])
            
            combined = y0 + result
            # reshape to 1D tensor
            combined_flat = combined.view(B, H * W, self.out_channels)
            # perform norm_relu on 1D tensor
            combined_flat = self.norm_relu(combined_flat)
            # reshape back to 2D tensor
            combined = combined_flat.view(B, H * W, self.cout_sqrt, self.cout_sqrt)
            
            dydt = -y0 + combined
            y0 = y0 + dt * dydt
        
        # (B, H, W, Cout)
        y_out = (y0 + x_expanded).contiguous().view(B, H, W, self.out_channels)
        # (b, Cout, H, W)
        y0 = y_out.permute(0, 3, 1, 2).contiguous()

        return y0

    def forward(self, x):
        return self._compiled_forward(x)


class ODEUniversalInvertedResidual(nn.Module):
    """ Universal Inverted Residual Block with ODE Conv2d """

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
            ode_num_steps: int = 10,
            ode_dw: bool = False,
            ode_pw: bool = True,
    ):
        super(ODEUniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_layer_no_act = get_norm_act_layer(norm_layer, None)

        def make_conv_norm_act(in_c, out_c, k_size, stride_c, groups_c, use_act=True, apply_aa=False, use_ode=True):
            padding = pad_type if isinstance(pad_type, int) else (k_size - 1) // 2 * dilation
            use_aa = apply_aa and aa_layer is not None and stride_c > 1
            eff_stride = 1 if use_aa else stride_c
            if use_ode:
                conv = ChannelwiseODESolver(
                    in_channels=in_c, out_channels=out_c, num_layers=ode_num_steps
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

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = make_conv_norm_act(in_chs, in_chs, dw_kernel_size_start, dw_start_stride, dw_start_groups, use_act=False, apply_aa=True, use_ode=ode_dw)
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        mid_chs = int(round(mid_chs ** 0.5)) ** 2
        if mid_chs != in_chs:
            self.pw_exp = make_conv_norm_act(in_chs, mid_chs, 1, 1, 1, use_act=True, apply_aa=False, use_ode=ode_pw)
        else:
            self.pw_exp = nn.Identity()

        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = make_conv_norm_act(mid_chs, mid_chs, dw_kernel_size_mid, stride, groups, use_act=True, apply_aa=True, use_ode=ode_dw)
        else:
            self.dw_mid = nn.Identity()

        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.pw_proj = make_conv_norm_act(mid_chs, out_chs, 1, 1, 1, use_act=False, apply_aa=False, use_ode=ode_pw)

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = make_conv_norm_act(out_chs, out_chs, dw_kernel_size_end, dw_end_stride, dw_end_groups, use_act=False, apply_aa=False, use_ode=ode_dw)
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

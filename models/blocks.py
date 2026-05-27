from typing import Callable, Dict, Optional, Type, List, Optional, Type, Union

import math

import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import create_conv2d, DropPath, create_act_layer, create_aa, to_2tuple, LayerType,\
    ConvNormAct, get_norm_act_layer
from timm.layers import use_fused_attn, to_2tuple

__all__ = [
    'SqueezeExcite', 'ConvBnAct', 'DepthwiseSeparableConv',
    'InvertedResidual', 'CondConvResidual', 'EdgeResidual',
    'UniversalInvertedResidual', 'MobileAttention',
    'DynamicUniversalInvertedResidual', 'ODEUniversalInvertedResidual', 'DynamicODEUniversalInvertedResidual'
]

ModuleType = Type[nn.Module]


def make_divisible(v, divisor=8, min_value=None, round_limit=.9):
    min_value = min_value or divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < round_limit * v:
        new_v += divisor
    return new_v


def num_groups(group_size: Optional[int], channels: int):
    if not group_size:  # 0 or None
        return 1  # normal conv with 1 group
    else:
        # NOTE group_size == 1 -> depthwise conv
        assert channels % group_size == 0
        return channels // group_size


class SqueezeExcite(nn.Module):
    """ Squeeze-and-Excitation w/ specific features for EfficientNet/MobileNet family

    Args:
        in_chs (int): input channels to layer
        rd_ratio (float): ratio of squeeze reduction
        act_layer (nn.Module): activation layer of containing block
        gate_layer (Callable): attention gate function
        force_act_layer (nn.Module): override block's activation fn if this is set/bound
        rd_round_fn (Callable): specify a fn to calculate rounding of reduced chs
    """

    def __init__(
            self,
            in_chs: int,
            rd_ratio: float = 0.25,
            rd_channels: Optional[int] = None,
            act_layer: LayerType = nn.ReLU,
            gate_layer: LayerType = nn.Sigmoid,
            force_act_layer: Optional[LayerType] = None,
            rd_round_fn: Optional[Callable] = None,
    ):
        super(SqueezeExcite, self).__init__()
        if rd_channels is None:
            rd_round_fn = rd_round_fn or round
            rd_channels = rd_round_fn(in_chs * rd_ratio)
        act_layer = force_act_layer or act_layer
        self.conv_reduce = nn.Conv2d(in_chs, rd_channels, 1, bias=True)
        self.act1 = create_act_layer(act_layer, inplace=True)
        self.conv_expand = nn.Conv2d(rd_channels, in_chs, 1, bias=True)
        self.gate = create_act_layer(gate_layer)

    def forward(self, x):
        x_se = x.mean((2, 3), keepdim=True)
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class ConvBnAct(nn.Module):
    """ Conv + Norm Layer + Activation w/ optional skip connection
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            kernel_size: int,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            pad_type: str = '',
            skip: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            drop_path_rate: float = 0.,
    ):
        super(ConvBnAct, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        groups = num_groups(group_size, in_chs)
        self.has_skip = skip and stride == 1 and in_chs == out_chs
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        self.conv = create_conv2d(
            in_chs, out_chs, kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(out_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # output of conv after act, same as block coutput
            return dict(module='bn1', hook_type='forward', num_chs=self.conv.out_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.bn1(x)
        x = self.aa(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class DepthwiseSeparableConv(nn.Module):
    """ Depthwise-separable block
    Used for DS convs in MobileNet-V1 and in the place of IR blocks that have no expansion
    (factor of 1.0). This is an alternative to having a IR with an optional first pw conv.
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            pw_kernel_size: int = 1,
            pw_act: bool = False,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            drop_path_rate: float = 0.,
    ):
        super(DepthwiseSeparableConv, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.has_pw_act = pw_act  # activation after point-wise conv
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        groups = num_groups(group_size, in_chs)

        self.conv_dw = create_conv2d(
            in_chs, in_chs, dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, padding=dw_pad_type, groups=groups)
        self.bn1 = norm_act_layer(in_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=out_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(in_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.conv_pw = create_conv2d(in_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, inplace=True, apply_act=self.has_pw_act)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_dw(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pw(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature%3==1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes!=3:
            hidden_planes = int(in_planes*ratios)+1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()


    def _initialize_weights(self):
        # fc1 with Kaiming initialization since it is followed by ReLU
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        if self.fc1.bias is not None:
            nn.init.constant_(self.fc1.bias, 0)
        # fc2 with small standard deviation initialization since it is followed by softmax
        nn.init.normal_(self.fc2.weight, std=0.01)
        if self.fc2.bias is not None:
            nn.init.constant_(self.fc2.bias, 0)

    def updata_temperature(self):
        # temperature annealing strategy, decrease temperature by 3 every time this function is called, until it reaches 1
        # in the paper, Temperature annealing refers to reducing τ from 30 to 1 linearly in the first 10 epochs to speed up the convergence of the routing weights. After 10 epochs, τ is fixed to 1 for the rest of training.
        if self.temperature!=1:
            self.temperature -=3
            print('Change temperature to:', str(self.temperature))


    def forward(self, x):
        x = self.avgpool(x)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        # pi_k(x) is returned here with softmax and temperature, where pi_k(x) is the weight for the k-th kernel
        return F.softmax(x/self.temperature, 1)


class Dynamic_conv2d(nn.Module):
    """
    https://zhuanlan.zhihu.com/p/208519425
    when training dynamic conv with batch_size > 1, there will be a problem: because the weights of dynamic conv are generated dynamically based on the input, each sample will have different weights, which makes it impossible to directly use standard convolution operations in the batch, because convolution operations require fixed weights.
    1. standard conv input with batch_size: [batch_size, in_channels, W, H]
    2. change view so input size become: [1, batch_size * in_channels, W, H]
    3. perform group convolution where groups = batch_size: [1, batch_size * out_channels, W', H']
    4. change view back to: [batch_size, out_channels, W', H']
    """
    def __init__(self, in_planes, out_planes, kernel_size, ratio=0.25, stride=1, padding=0, dilation=1, groups=1, bias=True, K=4,temperature=34, init_weight=True):
        super(Dynamic_conv2d, self).__init__()
        assert in_planes%groups==0
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = bias
        self.K = K
        self.attention = attention2d(in_planes, ratio, K, temperature)

        self.weight = nn.Parameter(torch.randn(K, out_planes, in_planes//groups, kernel_size, kernel_size), requires_grad=True)
        if bias:
            self.bias = nn.Parameter(torch.zeros(K, out_planes))
        else:
            self.bias = None
        if init_weight:
            self._initialize_weights()
            
    def _initialize_weights(self):
        import math
        for i in range(self.K):
            nn.init.kaiming_uniform_(self.weight[i])
            # times sqrt(k) to prevent variance collapse
            self.weight[i].data.mul_(math.sqrt(self.K))


    def update_temperature(self):
        self.attention.updata_temperature()

    def forward(self, x):
        softmax_attention = self.attention(x)
        batch_size, in_planes, height, width = x.size()
        x = x.view(1, -1, height, width)
        weight = self.weight.view(self.K, -1)

        # aggregate weight and bias using the attention weights
        aggregate_weight = torch.mm(softmax_attention, weight).view(batch_size*self.out_planes, self.in_planes//self.groups, self.kernel_size, self.kernel_size)

        # perform convolution with the aggregated weight and bias
        if self.bias is not None:
            aggregate_bias = torch.mm(softmax_attention, self.bias).view(-1)
            output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups*batch_size)
        else:
            output = F.conv2d(x, weight=aggregate_weight, bias=None, stride=self.stride, padding=self.padding,
                              dilation=self.dilation, groups=self.groups * batch_size)

        output = output.view(batch_size, self.out_planes, output.size(-2), output.size(-1))
        return output


class ODE_conv2d(nn.Module):
    """
    Neural ODE conv2d with COS-style driven dynamics (DDE) using Euler integration.

    Projection conv handles stride and channel mismatch first, then:
        y^0 = proj(x)
        y^{k+1} = y^k + dt_k * g_k(y^0)
        output   = y^L + y^0

    out_planes is factorized as a * b (a <= b, a closest to sqrt(out_planes)).
    Phi has shape (num_steps, a, a) and acts on the a-dimension of the
    (B, a, b, H*W) reshape of the channel axis.  Works for any out_planes.
    """
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, bias=False, num_steps=6):
        super(ODE_conv2d, self).__init__()
        a, b = ODE_conv2d._factorize(out_planes)

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.a = a
        self.b = b
        self.num_steps = num_steps

        self.proj = nn.Conv2d(
            in_planes, out_planes, kernel_size,
            stride=stride, padding=padding,
            dilation=dilation, groups=groups, bias=bias,
        )
        # per-step channelwise weight matrices: (num_steps, a, a)
        self.Phi = nn.Parameter(torch.empty(num_steps, a, a))
        self.dt = nn.Parameter(torch.zeros(num_steps))
        self._initialize_weights()

    @staticmethod
    def _factorize(C: int):
        """Return (a, b) with a*b==C, a<=b, a closest to sqrt(C)."""
        for a in range(int(math.isqrt(C)), 0, -1):
            if C % a == 0:
                return a, C // a
        return 1, C

    def _initialize_weights(self):
        nn.init.kaiming_uniform_(self.proj.weight)
        if self.proj.bias is not None:
            nn.init.zeros_(self.proj.bias)
        for i in range(self.num_steps):
            nn.init.eye_(self.Phi.data[i])

    def _g(self, x: torch.Tensor, k: int) -> torch.Tensor:
        B, C, H, W = x.shape
        x_r = x.reshape(B, self.a, self.b, H * W)
        g = torch.einsum('mn,bnqp->bmqp', self.Phi[k], x_r)
        return g.reshape(B, C, H, W)

    def forward(self, x):
        x_proj = self.proj(x)
        y = x_proj.clone()
        for k in range(self.num_steps):
            y = y + F.relu6(self.dt[k]) * self._g(x_proj, k)
        return y + x_proj


class ChannelwiseODESolver(nn.Module):
    def __init__(self, in_channels, out_channels, num_layers=30):
        super().__init__()
        self.num_layers = num_layers
        self.cin_sqrt=int(in_channels ** 0.5)
        self.cout_sqrt=int(out_channels ** 0.5)
        # originally, batch normalized is used with a value i where i*i=H*W, but i choose to use layer norm here
        # since layer norm is more suitable for variable batch size 
        self.sigma = nn.Sequential(
            nn.LayerNorm([self.cout_sqrt, self.cout_sqrt]),
            nn.ReLU6()
        )
        
        self.epsilon = 1.0 / num_layers

        # init phi and delta t
        self.delta_t = nn.Parameter(torch.empty(num_layers, 1).uniform_(1e-4, 1))
        self.phi = nn.Parameter(torch.empty(num_layers, 1, 1, self.cout_sqrt, self.cout_sqrt))
        torch.nn.init.normal_(self.phi, mean=0, std=0.1)
        torch.nn.init.normal_(self.delta_t, mean=0.4, std=0.005)

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
    
    def forward(self, x):
        B, C, H, W = x.size()
        # (b, Cin, H, W)

        # (B, H*W, sqrt(Cout), sqrt(Cout))
        x_expanded = self.feature_reshape(x, B, C)  
        y0 = x_expanded

        delta_t = torch.maximum(torch.tensor(self.epsilon, device=x.device), self.delta_t)
        
        for layer in range(self.num_layers):
            dt = delta_t[layer]
            result = torch.matmul(y0, self.phi[layer])
            dydt = -y0 + self.sigma(y0 + result)
            y0 = y0 + dt * dydt
        
        # (B, H, W, Cout)
        y_out = (y0 + x_expanded).contiguous().view(B, H, W, self.cout_sqrt * self.cout_sqrt)
        # (b, Cout, H, W)
        y0 = y_out.permute(0, 3, 1, 2).contiguous()

        return y0


class InvertedResidual(nn.Module):
    """ Inverted residual block w/ optional SE

    Originally used in MobileNet-V2 - https://arxiv.org/abs/1801.04381v4, this layer is often
    referred to as 'MBConv' for (Mobile inverted bottleneck conv) and is also used in
      * MNasNet - https://arxiv.org/abs/1807.11626
      * EfficientNet - https://arxiv.org/abs/1905.11946
      * MobileNet-V3 - https://arxiv.org/abs/1905.02244
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            exp_kernel_size: int = 1,
            pw_kernel_size: int = 1,
            s2d: int = 0,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            conv_kwargs: Optional[Dict] = None,
            drop_path_rate: float = 0.,
    ):
        super(InvertedResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Space to depth
        if s2d == 1:
            sd_chs = int(in_chs * 4)
            self.conv_s2d = create_conv2d(in_chs, sd_chs, kernel_size=2, stride=2, padding='same')
            self.bn_s2d = norm_act_layer(sd_chs, sd_chs)
            dw_kernel_size = (dw_kernel_size + 1) // 2
            dw_pad_type = 'same' if dw_kernel_size == 2 else pad_type
            in_chs = sd_chs
            use_aa = False  # disable AA
        else:
            self.conv_s2d = None
            self.bn_s2d = None
            dw_pad_type = pad_type

        mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)

        # Point-wise expansion
        self.conv_pw = create_conv2d(in_chs, mid_chs, exp_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        # Depth-wise convolution
        self.conv_dw = create_conv2d(
            mid_chs, mid_chs, dw_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=dw_pad_type, **conv_kwargs)
        self.bn2 = norm_act_layer(mid_chs, inplace=True)
        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type, **conv_kwargs)
        self.bn3 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        if self.conv_s2d is not None:
            x = self.conv_s2d(x)
            x = self.bn_s2d(x)
        x = self.conv_pw(x)
        x = self.bn1(x)
        x = self.conv_dw(x)
        x = self.bn2(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class LayerScale2d(nn.Module):
    def __init__(self, dim: int, init_values: float = 1e-5, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        gamma = self.gamma.view(1, -1, 1, 1)
        return x.mul_(gamma) if self.inplace else x * gamma


class UniversalInvertedResidual(nn.Module):
    """ Universal Inverted Residual Block (aka Universal Inverted Bottleneck, UIB)

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L778
    """

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
    ):
        super(UniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        # FIXME dilation isn't right w/ extra ks > 1 convs
        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = ConvNormAct(
                in_chs, in_chs, dw_kernel_size_start,
                stride=dw_start_stride,
                dilation=dilation,  # FIXME
                groups=dw_start_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            self.dw_start = nn.Identity()

        # Point-wise expansion
        mid_chs = make_divisible(in_chs * exp_ratio)
        self.pw_exp = ConvNormAct(
            in_chs, mid_chs, 1,
            padding=pad_type,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        # Middle depth-wise convolution
        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = ConvNormAct(
                mid_chs, mid_chs, dw_kernel_size_mid,
                stride=stride,
                dilation=dilation,  # FIXME
                groups=groups,
                padding=pad_type,
                act_layer=act_layer,
                norm_layer=norm_layer,
                aa_layer=aa_layer,
                **conv_kwargs,
            )
        else:
            # keeping mid as identity so it can be hooked more easily for features
            self.dw_mid = nn.Identity()

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.pw_proj = ConvNormAct(
            mid_chs, out_chs, 1,
            padding=pad_type,
            apply_act=False,
            act_layer=act_layer,
            norm_layer=norm_layer,
            **conv_kwargs,
        )

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = ConvNormAct(
                out_chs, out_chs, dw_kernel_size_end,
                stride=dw_end_stride,
                dilation=dilation,
                groups=dw_end_groups,
                padding=pad_type,
                apply_act=False,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **conv_kwargs,
            )
        else:
            self.dw_end = nn.Identity()

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PWL
            return dict(module='pw_proj.conv', hook_type='forward_pre', num_chs=self.pw_proj.conv.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.pw_proj.conv.out_channels)

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


class DynamicUniversalInvertedResidual(nn.Module):
    """ Universal Inverted Residual Block with Dynamic Conv2d """

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
    ):
        super(DynamicUniversalInvertedResidual, self).__init__()
        conv_kwargs = conv_kwargs or {}
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        if stride > 1:
            assert dw_kernel_size_start or dw_kernel_size_mid or dw_kernel_size_end

        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        norm_layer_no_act = get_norm_act_layer(norm_layer, None)

        def make_conv_norm_act(in_c, out_c, k_size, stride_c, groups_c, use_act=True, apply_aa=False, use_dynamic=True):
            padding = pad_type if isinstance(pad_type, int) else (k_size - 1) // 2 * dilation
            eff_stride = 1 if apply_aa and stride_c > 1 else stride_c
            if use_dynamic:
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
            if apply_aa and stride_c > 1:
                aa = create_aa(aa_layer, channels=out_c, stride=stride_c, enable=True)
                return nn.Sequential(conv, norm, aa)
            return nn.Sequential(conv, norm)

        if dw_kernel_size_start:
            dw_start_stride = stride if not dw_kernel_size_mid else 1
            dw_start_groups = num_groups(group_size, in_chs)
            self.dw_start = make_conv_norm_act(in_chs, in_chs, dw_kernel_size_start, dw_start_stride, dw_start_groups, use_act=False, apply_aa=True, use_dynamic=dynamic_dw)
        else:
            self.dw_start = nn.Identity()

        mid_chs = make_divisible(in_chs * exp_ratio)
        if mid_chs != in_chs:
            self.pw_exp = make_conv_norm_act(in_chs, mid_chs, 1, 1, 1, use_act=True, apply_aa=False, use_dynamic=dynamic_pw)
        else:
            self.pw_exp = nn.Identity()

        if dw_kernel_size_mid:
            groups = num_groups(group_size, mid_chs)
            self.dw_mid = make_conv_norm_act(mid_chs, mid_chs, dw_kernel_size_mid, stride, groups, use_act=True, apply_aa=True, use_dynamic=dynamic_dw)
        else:
            self.dw_mid = nn.Identity()

        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        self.pw_proj = make_conv_norm_act(mid_chs, out_chs, 1, 1, 1, use_act=False, apply_aa=False, use_dynamic=dynamic_pw)

        if dw_kernel_size_end:
            dw_end_stride = stride if not dw_kernel_size_start and not dw_kernel_size_mid else 1
            dw_end_groups = num_groups(group_size, out_chs)
            if dw_end_stride > 1:
                assert not aa_layer
            self.dw_end = make_conv_norm_act(out_chs, out_chs, dw_kernel_size_end, dw_end_stride, dw_end_groups, use_act=False, apply_aa=False, use_dynamic=dynamic_dw)
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
            ode_num_steps: int = 6,
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
            eff_stride = 1 if apply_aa and stride_c > 1 else stride_c
            if use_ode:
                conv = ODE_conv2d(
                    in_planes=in_c, out_planes=out_c, kernel_size=k_size,
                    stride=eff_stride, padding=padding,
                    dilation=dilation, groups=groups_c, bias=False, num_steps=ode_num_steps,
                )
            else:
                conv = create_conv2d(
                    in_c, out_c, kernel_size=k_size,
                    stride=eff_stride, padding=padding,
                    dilation=dilation, groups=groups_c, bias=False, **conv_kwargs,
                )
            norm = norm_act_layer(out_c, inplace=True) if use_act else norm_layer_no_act(out_c)
            if apply_aa and stride_c > 1:
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


class DynamicODEUniversalInvertedResidual(nn.Module):
    """UIB block with Dynamic_conv2d for DW layers and ODE_conv2d for PW layers."""

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
            eff_stride = 1 if apply_aa and stride_c > 1 else stride_c
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
            if apply_aa and stride_c > 1:
                aa = create_aa(aa_layer, channels=out_c, stride=stride_c, enable=True)
                return nn.Sequential(conv, norm, aa)
            return nn.Sequential(conv, norm)

        def make_pw(in_c, out_c, k_size, stride_c, groups_c, use_act=True):
            padding = pad_type if isinstance(pad_type, int) else (k_size - 1) // 2 * dilation
            conv = ODE_conv2d(
                in_planes=in_c, out_planes=out_c, kernel_size=k_size,
                stride=stride_c, padding=padding,
                dilation=dilation, groups=groups_c, bias=False, num_steps=ode_num_steps,
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


class MultiQueryAttention2d(nn.Module):
    """Multi Query Attention with spatial downsampling.

     3 parameters are introduced for the spatial downsampling:
     1. kv_stride: downsampling factor on Key and Values only.
     2. query_strides: horizontal & vertical strides on Query only.

    This is an optimized version.
    1. Projections in Attention is explicit written out as 1x1 Conv2D.
    2. Additional reshapes are introduced to bring a up to 3x speed up.
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            dim_out: Optional[int] = None,
            num_heads: int = 8,
            key_dim: Optional[int] = None,
            value_dim: Optional[int] = None,
            query_strides: int = 1,
            kv_stride: int = 1,
            dw_kernel_size: int = 3,
            dilation: int = 1,
            padding: Union[str, int, List[int]] = '',
            attn_drop: float = 0.,
            proj_drop: float = 0.,
            norm_layer: Type[nn.Module] = nn.BatchNorm2d,
            use_bias: bool = False,
            device=None,
            dtype=None,
    ):
        """Initializer.

        Args:
          num_heads: Number of attention heads.
          key_dim: Size of the attention key dimension.
          value_dim: Size of the attention value dimension.
          query_strides: Vertical stride size for query only.
          kv_stride: Key and value stride size.
          dw_kernel_size: Spatial dimension of the depthwise kernel.
        """
        dd = {'device': device, 'dtype': dtype}
        super().__init__()
        dim_out = dim_out or dim
        self.num_heads = num_heads
        self.key_dim = key_dim or dim // num_heads
        self.value_dim = value_dim or dim // num_heads
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_strides = any([s > 1 for s in self.query_strides])
        self.scale = self.key_dim ** -0.5
        self.fused_attn = use_fused_attn()
        self.drop = attn_drop

        self.query = nn.Sequential()
        if self.has_query_strides:
            # FIXME dilation
            if padding == 'same':
                self.query.add_module('down_pool', create_pool2d(
                    'avg',
                    kernel_size=self.query_strides,
                    padding='same',
                ))
            else:
                # no pad if not 'same' as kern=stride=even
                self.query.add_module('down_pool', nn.AvgPool2d(kernel_size=query_strides))
            self.query.add_module('norm', norm_layer(dim, **dd))
        self.query.add_module('proj', create_conv2d(
            dim,
            self.num_heads * self.key_dim,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))

        self.key = nn.Sequential()
        if kv_stride > 1:
            self.key.add_module('down_conv', create_conv2d(
                dim,
                dim,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                dilation=dilation,
                padding=padding,
                depthwise=True,
                **dd,
            ))
            self.key.add_module('norm', norm_layer(dim, **dd))
        self.key.add_module('proj', create_conv2d(
            dim,
            self.key_dim,
            kernel_size=1,
            padding=padding,
            bias=use_bias,
            **dd,
        ))

        self.value = nn.Sequential()
        if kv_stride > 1:
            self.value.add_module('down_conv', create_conv2d(
                dim,
                dim,
                kernel_size=dw_kernel_size,
                stride=kv_stride,
                dilation=dilation,
                padding=padding,
                depthwise=True,
                **dd,
            ))
            self.value.add_module('norm', norm_layer(dim, **dd))
        self.value.add_module('proj', create_conv2d(
            dim,
            self.value_dim,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))

        self.attn_drop = nn.Dropout(attn_drop)

        self.output = nn.Sequential()
        if self.has_query_strides:
            self.output.add_module('upsample', nn.Upsample(
                scale_factor=self.query_strides,
                mode='bilinear',
                align_corners=False
            ))
        self.output.add_module('proj', create_conv2d(
            self.value_dim * self.num_heads,
            dim_out,
            kernel_size=1,
            bias=use_bias,
            **dd,
        ))
        self.output.add_module('drop', nn.Dropout(proj_drop))

        self.einsum = False
        self.init_weights()

    def init_weights(self):
        # using xavier appeared to improve stability for mobilenetv4 hybrid w/ this layer
        nn.init.xavier_uniform_(self.query.proj.weight)
        nn.init.xavier_uniform_(self.key.proj.weight)
        nn.init.xavier_uniform_(self.value.proj.weight)
        if self.kv_stride > 1:
            nn.init.xavier_uniform_(self.key.down_conv.weight)
            nn.init.xavier_uniform_(self.value.down_conv.weight)
        nn.init.xavier_uniform_(self.output.proj.weight)

    def _reshape_input(self, t: torch.Tensor):
        """Reshapes a tensor to three dimensions, keeping the batch and channels."""
        s = t.shape
        t = t.reshape(s[0], s[1], -1).transpose(1, 2)
        if self.einsum:
            return t
        else:
            return t.unsqueeze(1).contiguous()

    def _reshape_projected_query(self, t: torch.Tensor, num_heads: int, key_dim: int):
        """Reshapes projected query: [b, n, n, h x k] -> [b, n x n, h, k]."""
        s = t.shape
        t = t.reshape(s[0], num_heads, key_dim, -1)
        if self.einsum:
            return t.permute(0, 3, 1, 2).contiguous()
        else:
            return t.transpose(-1, -2).contiguous()

    def _reshape_output(self, t: torch.Tensor, num_heads: int, h_px: int, w_px: int):
        """Reshape output:[b, n x n x h, k] -> [b, n, n, hk]."""
        s = t.shape
        feat_dim = s[-1] * num_heads
        if not self.einsum:
            t = t.transpose(1, 2)
        return t.reshape(s[0], h_px, w_px, feat_dim).permute(0, 3, 1, 2).contiguous()

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        """Run layer computation."""
        B, C, H, W = s = x.shape

        q = self.query(x)
        # desired q shape: [b, h, k, n x n] - [b, l, h, k]
        q = self._reshape_projected_query(q, self.num_heads, self.key_dim)

        k = self.key(x)
        # output shape of k: [b, k, p], p = m x m
        k = self._reshape_input(k)

        v = self.value(x)
        # output shape of v: [ b, p, k], p = m x m
        v = self._reshape_input(v)

        # desired q shape: [b, n x n, h, k]
        # desired k shape: [b, m x m, k]
        # desired logits shape: [b, n x n, h, m x m]
        if self.einsum:
            attn = torch.einsum('blhk,bpk->blhp', q, k) * self.scale
            if attn_mask is not None:
                # NOTE: assumes mask is float and in correct shape
                attn = attn + attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            o = torch.einsum('blhp,bpk->blhk', attn, v)
        else:
            if self.fused_attn:
                o = F.scaled_dot_product_attention(
                    q, k, v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.
                )
            else:
                q = q * self.scale
                attn = q @ k.transpose(-1, -2)
                if attn_mask is not None:
                    # NOTE: assumes mask is float and in correct shape
                    attn = attn + attn_mask
                attn = attn.softmax(dim=-1)
                attn = self.attn_drop(attn)
                o = attn @ v

        # reshape o into [b, hk, n, n,]
        o = self._reshape_output(o, self.num_heads, H // self.query_strides[0], W // self.query_strides[1])
        x = self.output(o)
        return x


class MobileAttention(nn.Module):
    """ Mobile Attention Block

    For MobileNetV4 - https://arxiv.org/abs/, referenced from
    https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L1504
    """
    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            stride: int = 1,
            dw_kernel_size: int = 3,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            num_heads: int = 8,
            key_dim: int = 64,
            value_dim: int = 64,
            use_multi_query: bool = False,
            query_strides: int = (1, 1),
            kv_stride: int = 1,
            cpe_dw_kernel_size: int = 3,
            noskip: bool = False,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            drop_path_rate: float = 0.,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
            layer_scale_init_value: Optional[float] = 1e-5,
            use_bias: bool = False,
            use_cpe: bool = False,
    ):
        super(MobileAttention, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        self.has_skip = (stride == 1 and in_chs == out_chs) and not noskip
        self.query_strides = to_2tuple(query_strides)
        self.kv_stride = kv_stride
        self.has_query_stride = any([s > 1 for s in self.query_strides])

        # This CPE is different than the one suggested in the original paper.
        # https://arxiv.org/abs/2102.10882
        # 1. Rather than adding one CPE before the attention blocks, we add a CPE
        #    into every attention block.
        # 2. We replace the expensive Conv2D by a Seperable DW Conv.
        if use_cpe:
            self.conv_cpe_dw = create_conv2d(
                in_chs, in_chs,
                kernel_size=cpe_dw_kernel_size,
                dilation=dilation,
                depthwise=True,
                bias=True,
            )
        else:
            self.conv_cpe_dw = None

        self.norm = norm_act_layer(in_chs, apply_act=False)

        if num_heads is None:
            assert in_chs % key_dim == 0
            num_heads = in_chs // key_dim

        if use_multi_query:
            self.attn = MultiQueryAttention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                key_dim=key_dim,
                value_dim=value_dim,
                query_strides=query_strides,
                kv_stride=kv_stride,
                dilation=dilation,
                padding=pad_type,
                dw_kernel_size=dw_kernel_size,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                #bias=use_bias, # why not here if used w/ mhsa?
            )
        else:
            self.attn = Attention2d(
                in_chs,
                dim_out=out_chs,
                num_heads=num_heads,
                attn_drop=attn_drop,
                proj_drop=proj_drop,
                bias=use_bias,
            )

        if layer_scale_init_value is not None:
            self.layer_scale = LayerScale2d(out_chs, layer_scale_init_value)
        else:
            self.layer_scale = nn.Identity()

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, input to PW
            return dict(module='conv_pw', hook_type='forward_pre', num_chs=self.conv_pw.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pw.out_channels)

    def forward(self, x):
        if self.conv_cpe_dw is not None:
            x_cpe = self.conv_cpe_dw(x)
            x = x + x_cpe

        shortcut = x
        x = self.norm(x)
        x = self.attn(x)
        x = self.layer_scale(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut

        return x


class CondConvResidual(InvertedResidual):
    """ Inverted residual block w/ CondConv routing"""

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            dw_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 1,
            pad_type: str = '',
            noskip: bool = False,
            exp_ratio: float = 1.0,
            exp_kernel_size: int = 1,
            pw_kernel_size: int = 1,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            num_experts: int = 0,
            drop_path_rate: float = 0.,
    ):

        self.num_experts = num_experts
        conv_kwargs = dict(num_experts=self.num_experts)
        super(CondConvResidual, self).__init__(
            in_chs,
            out_chs,
            dw_kernel_size=dw_kernel_size,
            stride=stride,
            dilation=dilation,
            group_size=group_size,
            pad_type=pad_type,
            noskip=noskip,
            exp_ratio=exp_ratio,
            exp_kernel_size=exp_kernel_size,
            pw_kernel_size=pw_kernel_size,
            act_layer=act_layer,
            norm_layer=norm_layer,
            aa_layer=aa_layer,
            se_layer=se_layer,
            conv_kwargs=conv_kwargs,
            drop_path_rate=drop_path_rate,
        )
        self.routing_fn = nn.Linear(in_chs, self.num_experts)

    def forward(self, x):
        shortcut = x
        pooled_inputs = F.adaptive_avg_pool2d(x, 1).flatten(1)  # CondConv routing
        routing_weights = torch.sigmoid(self.routing_fn(pooled_inputs))
        x = self.conv_pw(x, routing_weights)
        x = self.bn1(x)
        x = self.conv_dw(x, routing_weights)
        x = self.bn2(x)
        x = self.se(x)
        x = self.conv_pwl(x, routing_weights)
        x = self.bn3(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x


class EdgeResidual(nn.Module):
    """ Residual block with expansion convolution followed by pointwise-linear w/ stride

    Originally introduced in `EfficientNet-EdgeTPU: Creating Accelerator-Optimized Neural Networks with AutoML`
        - https://ai.googleblog.com/2019/08/efficientnet-edgetpu-creating.html

    This layer is also called FusedMBConv in the MobileDet, EfficientNet-X, and EfficientNet-V2 papers
      * MobileDet - https://arxiv.org/abs/2004.14525
      * EfficientNet-X - https://arxiv.org/abs/2102.05610
      * EfficientNet-V2 - https://arxiv.org/abs/2104.00298
    """

    def __init__(
            self,
            in_chs: int,
            out_chs: int,
            exp_kernel_size: int = 3,
            stride: int = 1,
            dilation: int = 1,
            group_size: int = 0,
            pad_type: str = '',
            force_in_chs: int = 0,
            noskip: bool = False,
            exp_ratio: float = 1.0,
            pw_kernel_size:  int = 1,
            act_layer: LayerType = nn.ReLU,
            norm_layer: LayerType = nn.BatchNorm2d,
            aa_layer: Optional[LayerType] = None,
            se_layer: Optional[ModuleType] = None,
            drop_path_rate: float = 0.,
    ):
        super(EdgeResidual, self).__init__()
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        if force_in_chs > 0:
            mid_chs = make_divisible(force_in_chs * exp_ratio)
        else:
            mid_chs = make_divisible(in_chs * exp_ratio)
        groups = num_groups(group_size, mid_chs)  # NOTE: Using out_chs of conv_exp for groups calc
        self.has_skip = (in_chs == out_chs and stride == 1) and not noskip
        use_aa = aa_layer is not None and stride > 1  # FIXME handle dilation

        # Expansion convolution
        self.conv_exp = create_conv2d(
            in_chs, mid_chs, exp_kernel_size,
            stride=1 if use_aa else stride,
            dilation=dilation, groups=groups, padding=pad_type)
        self.bn1 = norm_act_layer(mid_chs, inplace=True)

        self.aa = create_aa(aa_layer, channels=mid_chs, stride=stride, enable=use_aa)

        # Squeeze-and-excitation
        self.se = se_layer(mid_chs, act_layer=act_layer) if se_layer else nn.Identity()

        # Point-wise linear projection
        self.conv_pwl = create_conv2d(mid_chs, out_chs, pw_kernel_size, padding=pad_type)
        self.bn2 = norm_act_layer(out_chs, apply_act=False)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def feature_info(self, location):
        if location == 'expansion':  # after SE, before PWL
            return dict(module='conv_pwl', hook_type='forward_pre', num_chs=self.conv_pwl.in_channels)
        else:  # location == 'bottleneck', block output
            return dict(module='', num_chs=self.conv_pwl.out_channels)

    def forward(self, x):
        shortcut = x
        x = self.conv_exp(x)
        x = self.bn1(x)
        x = self.aa(x)
        x = self.se(x)
        x = self.conv_pwl(x)
        x = self.bn2(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x
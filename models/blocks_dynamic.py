import math
from typing import Dict, Optional, Type
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import create_conv2d, DropPath, create_aa, get_norm_act_layer, LayerType
from .blocks_common import make_divisible, num_groups, LayerScale2d, ModuleType

__all__ = ['attention2d', 'Dynamic_conv2d', 'DynamicUniversalInvertedResidual']


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
            use_aa = apply_aa and aa_layer is not None and stride_c > 1
            eff_stride = 1 if use_aa else stride_c
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
            if use_aa:
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

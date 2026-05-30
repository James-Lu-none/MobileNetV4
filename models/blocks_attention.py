from typing import Optional, Union, List, Type
import torch
import torch.nn as nn
from torch.nn import functional as F

from timm.layers import use_fused_attn, to_2tuple, create_pool2d, create_conv2d, DropPath, get_norm_act_layer, LayerType, Attention2d
from .blocks_common import LayerScale2d, ModuleType

__all__ = ['MultiQueryAttention2d', 'MobileAttention']


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

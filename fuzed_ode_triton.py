import triton
import triton.language as tl

from torch._inductor.runtime import triton_helpers, triton_heuristics
from torch._inductor.runtime.triton_helpers import libdevice, math as tl_math
from torch._inductor.runtime.hints import AutotuneHint, ReductionHint, TileHint, DeviceProperties
triton_helpers.set_driver_to_gpu()

@triton_heuristics.persistent_reduction(
    size_hints={'x': 32768, 'r0_': 64},
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    triton_meta={'signature': {'in_out_ptr0': '*fp32', 'in_ptr0': '*fp32', 'in_ptr1': '*fp32', 'in_ptr2': '*fp32', 'in_ptr3': '*fp32', 'out_ptr2': '*fp32', 'out_ptr3': '*fp32', 'out_ptr4': '*fp32', 'xnumel': 'i32', 'r0_numel': 'i32', 'XBLOCK': 'constexpr'}, 'device': DeviceProperties(type='cuda', index=0, multi_processor_count=188, cc=120, major=12, regs_per_multiprocessor=65536, max_threads_per_multi_processor=1536, warp_size=32), 'constants': {}, 'configs': [{(0,): [['tt.divisibility', 16]], (1,): [['tt.divisibility', 16]], (2,): [['tt.divisibility', 16]], (3,): [['tt.divisibility', 16]], (4,): [['tt.divisibility', 16]], (5,): [['tt.divisibility', 16]], (6,): [['tt.divisibility', 16]], (7,): [['tt.divisibility', 16]], (8,): [['tt.divisibility', 16]], (9,): [['tt.divisibility', 16]]}]},
    inductor_meta={'grid_type': 'Grid1D', 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_neg_1', 'mutated_arg_names': ['in_out_ptr0'], 'optimize_mem': False, 'no_x_dim': False, 'num_load': 5, 'num_reduction': 4, 'backend_hash': '77F10D9EFD0D9A70E44C33B879F1938FAFC5B5BC6E5D56EAB42FA2DB5DF4A410', 'are_deterministic_algorithms_enabled': False, 'assert_indirect_indexing': True, 'autotune_local_cache': True, 'autotune_pointwise': True, 'autotune_remote_cache': None, 'force_disable_caches': False, 'dynamic_scale_rblock': True, 'max_autotune': False, 'max_autotune_pointwise': False, 'min_split_scan_rblock': 256, 'spill_threshold': 16, 'store_cubin': False, 'tiling_scores': {'x': 200704, 'r0_': 51380736}}
)
@triton.jit
def triton_per_fused_add_mul_native_layer_norm_native_layer_norm_backward_neg_1(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, out_ptr3, out_ptr4, xnumel, r0_numel, XBLOCK : tl.constexpr):
    xnumel = 25088
    r0_numel = 64
    R0_BLOCK: tl.constexpr = 64
    rnumel = r0_numel
    RBLOCK: tl.constexpr = R0_BLOCK
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    r0_index = tl.arange(0, R0_BLOCK)[None, :]
    r0_offset = 0
    r0_mask = tl.full([XBLOCK, R0_BLOCK], True, tl.int1)
    roffset = r0_offset
    rindex = r0_index
    r0_1 = r0_index
    x0 = xindex
    
    tmp0 = tl.load(in_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    # load matmul result
    tmp1 = tl.load(in_out_ptr0 + (r0_1 + 64*x0), xmask, other=0.0)
    
    # load LayerNorm weight and bias
    tmp27 = tl.load(in_ptr1 + (r0_1), None, eviction_policy='evict_last')
    tmp29 = tl.load(in_ptr2 + (r0_1), None, eviction_policy='evict_last')
    
    # load ODE time step (dt)
    tmp36 = tl.load(in_ptr3 + (0))
    tmp37 = tl.broadcast_to(tmp36, [XBLOCK, R0_BLOCK])
    
    # fused operation: combined = y0 + result
    tmp2 = tmp0 + tmp1
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, R0_BLOCK])
    tmp5 = tl.where(xmask, tmp3, 0)
    tmp6 = tl.broadcast_to(tmp3, [XBLOCK, R0_BLOCK])
    tmp8 = tl.where(xmask, tmp6, 0)
    
    # LayerNorm: mean
    tmp9 = tl.sum(tmp8, 1)[:, None]
    tmp10 = tl.full([XBLOCK, 1], 64, tl.int32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = (tmp9 / tmp11)
    
    # LayerNorm: variance and rsqrt
    tmp13 = tmp3 - tmp12
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [XBLOCK, R0_BLOCK])
    tmp17 = tl.where(xmask, tmp15, 0)
    tmp18 = tl.sum(tmp17, 1)[:, None]
    tmp19 = tmp2 - tmp12
    tmp20 = 64.0
    tmp21 = (tmp18 / tmp20)
    tmp22 = 1e-05
    tmp23 = tmp21 + tmp22
    tmp24 = libdevice.rsqrt(tmp23)
    
    # LayerNorm
    tmp25 = tmp19 * tmp24
    tmp26 = -tmp0  # -y0
    tmp28 = tmp25 * tmp27
    tmp30 = tmp28 + tmp29
    
    # fused ReLU6
    tmp31 = 0.0
    tmp32 = triton_helpers.maximum(tmp30, tmp31)
    tmp33 = 6.0
    tmp34 = triton_helpers.minimum(tmp32, tmp33)
    
    # fused ODE update
    tmp35 = tmp26 + tmp34
    tmp38 = 0.009999999776482582
    tmp39 = triton_helpers.maximum(tmp38, tmp37)
    tmp40 = tmp39 * tmp35
    tmp41 = tmp0 + tmp40
    tmp42 = 0.015625
    tmp43 = tmp24 * tmp42
    
    # write back to global memory
    tl.store(in_out_ptr0 + (r0_1 + 64*x0), tmp25, xmask)
    tl.store(out_ptr2 + (r0_1 + 64*x0), tmp35, xmask)
    tl.store(out_ptr3 + (r0_1 + 64*x0), tmp41, xmask)
    tl.store(out_ptr4 + (x0), tmp43, xmask)

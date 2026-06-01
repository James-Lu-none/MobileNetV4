import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import matplotlib.patches as mpatches


OI_BOUNDS = [5, 500]

type_markers = {
    'Conv2d':             'o',
    'PW-Conv2d':          'v',
    'DW-Conv2d':          's',
    'FC':                 '^',
    'ODE':                'D',
    'ODE_Fused':          'd',
    'Dynamic-Conv2d':     'p',
    'Dynamic-DW-Conv2d':  '*',
}

try:
    from models.blocks_ode import ChannelwiseODESolver as _ODE_CLS
except ImportError:
    _ODE_CLS = None

try:
    from models.blocks_dynamic import Dynamic_conv2d as _DYNAMIC_CONV_CLS
except ImportError:
    _DYNAMIC_CONV_CLS = None



def _collect_stats(model, input_size, beta_gb_per_s, bytes_per_element=4, use_fused_ode=False):
    stats = []
    hooks = []
    beta = beta_gb_per_s * 1e9

    def make_hook(name, module):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return

            weight_count = 0
            if hasattr(mod, 'weight') and mod.weight is not None:
                weight_count += mod.weight.numel()
            if hasattr(mod, 'bias') and mod.bias is not None:
                weight_count += mod.bias.numel()

            activation_count = out[0].numel() if out.dim() > 1 else out.numel()

            macs = 0
            display_type = type(mod).__name__

            if isinstance(mod, nn.Conv2d):
                out_h, out_w = out.shape[2], out.shape[3]
                macs = (out_h * out_w * mod.out_channels *
                        (mod.in_channels // mod.groups) *
                        mod.kernel_size[0] * mod.kernel_size[1])
                if mod.groups == mod.in_channels:
                    display_type = 'DW-Conv2d'
                elif mod.kernel_size[0] == 1 and mod.kernel_size[1] == 1:
                    display_type = 'PW-Conv2d'
            elif isinstance(mod, nn.ConvTranspose2d):
                out_h, out_w = out.shape[2], out.shape[3]
                macs = (out_h * out_w * mod.out_channels *
                        (mod.in_channels // mod.groups) *
                        mod.kernel_size[0] * mod.kernel_size[1])
            elif isinstance(mod, nn.Linear):
                num_vectors = inp[0][0].numel() // mod.in_features
                macs = num_vectors * mod.in_features * mod.out_features
                display_type = 'FC'
            elif isinstance(mod, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                macs = 2 * activation_count

            if macs == 0:
                return


            memory_bytes = (weight_count + activation_count) * bytes_per_element

            stats.append({
                'name': name,
                'type': display_type,
                'macs': macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': macs / memory_bytes,
            })

        return hook

    def make_ode_hook(name, beta):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            _, _, H, W = out.shape
            num_steps   = mod.num_layers
            cout_sqrt   = mod.cout_sqrt
            out_ch      = mod.out_channels
            cin_ch      = mod.in_channels

            # Bilinear interpolation if cin != out
            if cin_ch != out_ch:
                prep_macs = 4 * H * W * out_ch
                prep_bytes = (cin_ch * H * W + out_ch * H * W) * bytes_per_element
            else:
                prep_macs = 0
                prep_bytes = 2 * out_ch * H * W * bytes_per_element
            
            prep_oi = prep_macs / prep_bytes if prep_bytes > 0 else 0

            # Matmul step
            step_matmul_macs = H * W * (cout_sqrt ** 3)
            step_matmul_bytes = (2 * H * W * out_ch + out_ch) * bytes_per_element
            step_matmul_oi = step_matmul_macs / step_matmul_bytes

            # Norm/relu step
            step_norm_macs = 2 * H * W * out_ch
            step_norm_bytes = (2 * H * W * out_ch + 2 * out_ch) * bytes_per_element
            step_norm_oi = step_norm_macs / step_norm_bytes

            # Residual update step
            # dydt = -y0 + norm_out (1 次減法) 和 y0 = y0 + dt * dydt (1 次乘法，1 次加法)
            step_update_macs = 3 * H * W * out_ch
            step_update_bytes = (3 * H * W * out_ch + 1) * bytes_per_element
            step_update_oi = step_update_macs / step_update_bytes

            # Post-processing
            post_macs = 0
            post_bytes = 3 * out_ch * H * W * bytes_per_element
            post_oi = 0

            total_macs = prep_macs + num_steps * (step_matmul_macs + step_norm_macs + step_update_macs) + post_macs
            total_bytes = prep_bytes + num_steps * (step_matmul_bytes + step_norm_bytes + step_update_bytes) + post_bytes
            effective_oi = total_macs / total_bytes if total_bytes > 0 else 0

            latencies_ns = {}
            for oi_bound in OI_BOUNDS:
                if prep_macs > 0:
                    prep_lat = prep_macs / (beta * min(prep_oi, oi_bound))
                else:
                    prep_lat = prep_bytes / beta
                
                step_matmul_lat = step_matmul_macs / (beta * min(step_matmul_oi, oi_bound))
                step_norm_lat = step_norm_macs / (beta * min(step_norm_oi, oi_bound))
                step_update_lat = step_update_macs / (beta * min(step_update_oi, oi_bound))
                loop_lat = num_steps * (step_matmul_lat + step_norm_lat + step_update_lat)
                
                post_lat = post_bytes / beta
                
                latencies_ns[oi_bound] = (prep_lat + loop_lat + post_lat) * 1e9

            weight_count = (mod.phi.numel()        # num_steps * out_ch
                            + mod.delta_t.numel()  # num_steps
                            + 2 * out_ch)          # LayerNorm weight + bias
            activation_count = H * W * out_ch

            stat_entry = {
                'name': name,
                'type': 'ODE',
                'macs': total_macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': effective_oi,
            }
            for oi_bound in OI_BOUNDS:
                stat_entry[f'latency_ns_oi{oi_bound}'] = latencies_ns[oi_bound]
                stat_entry[f'latency_us_oi{oi_bound}'] = latencies_ns[oi_bound] / 1000.0

            stats.append(stat_entry)
        return hook

    def make_ode_hook_fused(name, beta):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            
            _, _, H, W = out.shape
            num_steps   = mod.num_layers
            cout_sqrt   = mod.cout_sqrt
            out_ch      = mod.out_channels
            cin_ch      = mod.in_channels

            if cin_ch != out_ch:
                prep_macs = 4 * H * W * out_ch
                prep_bytes = (cin_ch * H * W + out_ch * H * W) * bytes_per_element
            else:
                prep_macs = 0
                prep_bytes = 2 * out_ch * H * W * bytes_per_element
            
            prep_oi = prep_macs / prep_bytes if prep_bytes > 0 else 0

            # --- Fully-Fused ODE Kernel ---
            # Matmul step
            step_matmul_macs = H * W * (cout_sqrt ** 3)
            # Norm/relu step
            step_norm_macs   = 2 * H * W * out_ch
            # Update step
            # dydt = -y0 + norm_out (1 次減法) 和 y0 = y0 + dt * dydt (1 次乘法，1 次加法)
            step_update_macs = 3 * H * W * out_ch
            loop_macs = num_steps * (step_matmul_macs + step_norm_macs + step_update_macs)
            
            # y0_final + x_expanded
            post_macs = H * W * out_ch 
            fused_macs = loop_macs + post_macs

            # Memory Bytes
            # only read LN_weight, LN_bias, dt in each loop, and do not write y_n back to VRAM
            weight_bytes_per_step = (out_ch + 2 * out_ch + 1) * bytes_per_element
            loop_weight_bytes = num_steps * weight_bytes_per_step
            
            feature_io_bytes = 3 * H * W * out_ch * bytes_per_element
            
            fused_bytes = feature_io_bytes + loop_weight_bytes
            fused_oi = fused_macs / fused_bytes if fused_bytes > 0 else 0

            total_macs = prep_macs + fused_macs
            total_bytes = prep_bytes + fused_bytes
            effective_oi = total_macs / total_bytes if total_bytes > 0 else 0

            latencies_ns = {}
            for oi_bound in OI_BOUNDS:
                if prep_macs > 0:
                    prep_lat = prep_macs / (beta * min(prep_oi, oi_bound))
                else:
                    prep_lat = prep_bytes / beta
                
                fused_lat = fused_macs / (beta * min(fused_oi, oi_bound))
                latencies_ns[oi_bound] = (prep_lat + fused_lat) * 1e9

            weight_count = (mod.phi.numel()        # num_steps * out_ch
                            + mod.delta_t.numel()  # num_steps
                            + 2 * out_ch)          # LayerNorm weight + bias
            activation_count = H * W * out_ch

            stat_entry = {
                'name': name,
                'type': 'ODE_Fused',
                'macs': total_macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': effective_oi,
            }
            
            for oi_bound in OI_BOUNDS:
                stat_entry[f'latency_ns_oi{oi_bound}'] = latencies_ns[oi_bound]
                stat_entry[f'latency_us_oi{oi_bound}'] = latencies_ns[oi_bound] / 1000.0

            stats.append(stat_entry)
        return hook

    def make_dynamic_conv_hook(name, beta):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            out_h, out_w = out.shape[2], out.shape[3]
            
            in_planes = mod.in_planes
            out_planes = mod.out_planes
            kernel_size = mod.kernel_size
            groups = mod.groups
            K = mod.K
            
            hidden_planes = K
            if hasattr(mod, 'attention') and hasattr(mod.attention, 'fc1'):
                hidden_planes = mod.attention.fc1.out_channels

            # Attention: avgpool + fc1 + fc2 + softmax
            attn_avg_macs = in_planes * out_h * out_w
            attn_avg_bytes = (in_planes * out_h * out_w + in_planes) * bytes_per_element
            attn_avg_oi = attn_avg_macs / attn_avg_bytes if attn_avg_bytes > 0 else 0
            
            attn_fc1_macs = in_planes * hidden_planes
            attn_fc1_bytes = (in_planes + in_planes * hidden_planes + hidden_planes) * bytes_per_element
            attn_fc1_oi = attn_fc1_macs / attn_fc1_bytes if attn_fc1_bytes > 0 else 0
            
            attn_fc2_macs = hidden_planes * K
            attn_fc2_bytes = (hidden_planes + hidden_planes * K + 2 * K) * bytes_per_element
            attn_fc2_oi = attn_fc2_macs / attn_fc2_bytes if attn_fc2_bytes > 0 else 0
            
            attn_soft_macs = K
            attn_soft_bytes = 2 * K * bytes_per_element
            attn_soft_oi = attn_soft_macs / attn_soft_bytes if attn_soft_bytes > 0 else 0
            
            # Weight aggregation
            agg_macs = K * out_planes * (in_planes // groups) * kernel_size * kernel_size
            agg_bytes = (K + (K + 1) * out_planes * (in_planes // groups) * kernel_size * kernel_size) * bytes_per_element
            agg_oi = agg_macs / agg_bytes if agg_bytes > 0 else 0
            
            # Convolution
            conv_macs = out_h * out_w * out_planes * (in_planes // groups) * kernel_size * kernel_size
            conv_bytes = (in_planes * out_h * out_w + 
                          out_planes * (in_planes // groups) * kernel_size * kernel_size + 
                          out_planes * out_h * out_w) * bytes_per_element
            conv_oi = conv_macs / conv_bytes if conv_bytes > 0 else 0

            total_macs = (attn_avg_macs + attn_fc1_macs + attn_fc2_macs + attn_soft_macs + 
                          agg_macs + conv_macs)
            total_bytes = (attn_avg_bytes + attn_fc1_bytes + attn_fc2_bytes + attn_soft_bytes + 
                           agg_bytes + conv_bytes)
            effective_oi = total_macs / total_bytes if total_bytes > 0 else 0

            latencies_ns = {}
            for oi_bound in OI_BOUNDS:
                attn_avg_lat = attn_avg_macs / (beta * min(attn_avg_oi, oi_bound)) if attn_avg_macs > 0 else attn_avg_bytes / beta
                attn_fc1_lat = attn_fc1_macs / (beta * min(attn_fc1_oi, oi_bound)) if attn_fc1_macs > 0 else attn_fc1_bytes / beta
                attn_fc2_lat = attn_fc2_macs / (beta * min(attn_fc2_oi, oi_bound)) if attn_fc2_macs > 0 else attn_fc2_bytes / beta
                attn_soft_lat = attn_soft_macs / (beta * min(attn_soft_oi, oi_bound)) if attn_soft_macs > 0 else attn_soft_bytes / beta
                attn_lat = attn_avg_lat + attn_fc1_lat + attn_fc2_lat + attn_soft_lat
                
                agg_lat = agg_macs / (beta * min(agg_oi, oi_bound)) if agg_macs > 0 else agg_bytes / beta
                conv_lat = conv_macs / (beta * min(conv_oi, oi_bound)) if conv_macs > 0 else conv_bytes / beta
                
                latencies_ns[oi_bound] = (attn_lat + agg_lat + conv_lat) * 1e9

            weight_count = sum(p.numel() for p in mod.parameters())
            activation_count = out[0].numel() if out.dim() > 1 else out.numel()

            display_type = 'Dynamic-Conv2d'
            if groups == in_planes:
                display_type = 'Dynamic-DW-Conv2d'

            stat_entry = {
                'name': name,
                'type': display_type,
                'macs': total_macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': effective_oi,
            }
            for oi_bound in OI_BOUNDS:
                stat_entry[f'latency_ns_oi{oi_bound}'] = latencies_ns[oi_bound]
                stat_entry[f'latency_us_oi{oi_bound}'] = latencies_ns[oi_bound] / 1000.0

            stats.append(stat_entry)
        return hook

    skip_child_ids = set()
    if _ODE_CLS is not None:
        for _, module in model.named_modules():
            if isinstance(module, _ODE_CLS):
                for _, child in module.named_modules():
                    if child is not module:
                        skip_child_ids.add(id(child))
    if _DYNAMIC_CONV_CLS is not None:
        for _, module in model.named_modules():
            if isinstance(module, _DYNAMIC_CONV_CLS):
                for _, child in module.named_modules():
                    if child is not module:
                        skip_child_ids.add(id(child))

    for name, module in model.named_modules():
        if _ODE_CLS is not None and isinstance(module, _ODE_CLS):
            if use_fused_ode:
                h = module.register_forward_hook(make_ode_hook_fused(name, beta))
            else:
                h = module.register_forward_hook(make_ode_hook(name, beta))
            hooks.append(h)
        elif _DYNAMIC_CONV_CLS is not None and isinstance(module, _DYNAMIC_CONV_CLS):
            h = module.register_forward_hook(make_dynamic_conv_hook(name, beta))
            hooks.append(h)
        elif len(list(module.children())) == 0 and id(module) not in skip_child_ids:
            h = module.register_forward_hook(make_hook(name, module))
            hooks.append(h)

    device = next(model.parameters()).device
    dummy = torch.zeros(1, 3, input_size, input_size, device=device)
    model.eval()
    with torch.no_grad():
        model(dummy)

    for h in hooks:
        h.remove()

    return stats


def analyze_layer_stats(model_name, stats, beta_gb_per_s, output_dir, bytes_per_element=4):
    """Single-model roofline analysis: saves CSV and plot."""
    beta = beta_gb_per_s * 1e9

    if not stats:
        print(f"[analyze_model] No supported layers found for {model_name}.")
        return

    output_dir = Path(output_dir)

    csv_rows = []
    for s in stats:
        row = {k: s[k] for k in ('name', 'type', 'macs', 'weight_count', 'activation_count', 'oi_layer')}
        for oi_bound in OI_BOUNDS:
            if f'latency_ns_oi{oi_bound}' in s:
                row[f'latency_ns_oi{oi_bound}'] = s[f'latency_ns_oi{oi_bound}']
            else:
                eff_oi = min(s['oi_layer'], oi_bound)
                row[f'latency_ns_oi{oi_bound}'] = s['macs'] / (beta * eff_oi) * 1e9
        csv_rows.append(row)

    csv_path = output_dir / f"{model_name}_layer_stats.csv"
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=csv_rows[0].keys())
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"[analyze_model] Layer stats saved to {csv_path}")

    import matplotlib.colors as mcolors
    import matplotlib.cm as cm
    import numpy as np

    cmap = mcolors.LinearSegmentedColormap.from_list(
        'black_orange_yellow', ['black', 'darkorange', 'yellow']
    )
    oi_values = np.array([s['oi_layer'] for s in stats if s['type'] in type_markers])
    norm = mcolors.LogNorm(vmin=oi_values.min(), vmax=oi_values.max())

    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    for ax, oi_bound in zip(axes, OI_BOUNDS):
        plotted_types = set()
        scatters = {}
        type_counts = {}
        type_latencies = {}
        total_latency_us = 0.0
        xi = 0
        for s in stats:
            if s['type'] not in type_markers:
                continue
            if f'latency_us_oi{oi_bound}' in s:
                latency_us = s[f'latency_us_oi{oi_bound}']
            else:
                latency_us = s['macs'] / (beta * min(s['oi_layer'], oi_bound)) * 1e6
            marker = type_markers[s['type']]
            color = cmap(norm(s['oi_layer']))
            ax.scatter(xi, latency_us, s=40, marker=marker, color=color, zorder=3)
            
            type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
            type_latencies[s['type']] = type_latencies.get(s['type'], 0.0) + latency_us
            total_latency_us += latency_us
            
            if s['type'] not in plotted_types:
                scatters[s['type']] = marker
                plotted_types.add(s['type'])
            xi += 1

        legend_handles = []
        for t, m in sorted(scatters.items()):
            count = type_counts[t]
            lat = type_latencies[t]
            label = f"{t}: N={count}, {lat:.1f}us"
            legend_handles.append(ax.scatter([], [], marker=m, color='gray', s=40, label=label))
            
        total_label = f"Total Latency: {total_latency_us:.1f}us"
        legend_handles.append(mpatches.Patch(color='none', label=total_label))
        
        ax.legend(handles=legend_handles, loc='upper left', fontsize=8, framealpha=0.7)
        ax.set_xlabel('Layer')
        ax.set_ylabel('Predicted Roofline Latency (us)')
        ax.set_title(f'Model: {model_name} | OI bound = {oi_bound} MAC/byte')
        ax.grid(True, linestyle='--', alpha=0.4)

        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, pad=0.01)
        cbar.set_label('OI (MAC/byte)', fontsize=8)

    plt.tight_layout()
    plot_path = output_dir / f"{model_name}_layer_latency.png"
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyze_model] Latency plot saved to {plot_path}")


def _plot_comparison(all_stats, beta_gb_per_s, output_dir, filename="comparison_latency.png", bytes_per_element=4):
    import matplotlib.colors as mcolors
    import numpy as np

    beta = beta_gb_per_s * 1e9
    output_dir = Path(output_dir)

    model_names = list(all_stats.keys())
    n_models = len(model_names)
    import matplotlib.cm as cm

    fig, axes = plt.subplots(2, n_models, figsize=(5 * n_models, 10), sharey='row')
    if n_models == 1:
        axes = [[axes[0]], [axes[1]]]
    cmap = mcolors.LinearSegmentedColormap.from_list(
        'black_orange_yellow', ['black', 'darkorange', 'yellow']
    )

    for col, model_name in enumerate(model_names):
        stats = all_stats[model_name]
        oi_values = np.array([s['oi_layer'] for s in stats if s['type'] in type_markers])
        norm = mcolors.LogNorm(vmin=oi_values.min(), vmax=oi_values.max())

        for row, oi_bound in enumerate(OI_BOUNDS):
            ax = axes[row][col]
            plotted_types = set()
            type_counts = {}
            type_latencies = {}
            total_latency_us = 0.0
            xi = 0
            for s in stats:
                if s['type'] not in type_markers:
                    continue
                if f'latency_us_oi{oi_bound}' in s:
                    latency_us = s[f'latency_us_oi{oi_bound}']
                else:
                    latency_us = s['macs'] / (beta * min(s['oi_layer'], oi_bound)) * 1e6
                marker = type_markers[s['type']]
                color = cmap(norm(s['oi_layer']))
                ax.scatter(xi, latency_us, s=40, marker=marker, color=color, zorder=3)
                
                type_counts[s['type']] = type_counts.get(s['type'], 0) + 1
                type_latencies[s['type']] = type_latencies.get(s['type'], 0.0) + latency_us
                total_latency_us += latency_us
                
                plotted_types.add(s['type'])
                xi += 1

            legend_handles = []
            for t in sorted(plotted_types):
                count = type_counts[t]
                lat = type_latencies[t]
                marker = type_markers[t]
                label = f"{t}: N={count}, {lat:.1f}us"
                legend_handles.append(ax.scatter([], [], marker=marker, color='gray', s=40, label=label))
                
            total_label = f"Total: {total_latency_us:.1f}us"
            legend_handles.append(mpatches.Patch(color='none', label=total_label))
            
            ax.legend(handles=legend_handles, loc='upper left', fontsize=6, framealpha=0.7)
            if row == 0:
                ax.set_title(model_name, fontsize=9, fontweight='bold')
            ax.set_xlabel('Layer')
            ax.set_ylabel('Latency (us)')
            ax.grid(True, linestyle='--', alpha=0.4)

            sm = cm.ScalarMappable(cmap=cmap, norm=norm)
            sm.set_array([])
            cbar = fig.colorbar(sm, ax=ax, pad=0.01)
            cbar.set_label('OI (MAC/byte)', fontsize=7)

    row_labels = [
        f'Roofline Ops ({OI_BOUNDS[0]:.2f} MACs/byte - Slow CPU)',
        f'Roofline Ops ({OI_BOUNDS[1]:.2f} MACs/byte - Accelerator)',
    ]
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.55, top=0.90)
    fig.canvas.draw()
    bbox0 = axes[0][0].get_position()
    bbox1 = axes[1][0].get_position()
    gap_center = (bbox0.y0 + bbox1.y1) / 2
    fig.text(0.01, bbox0.y1 + 0.03, row_labels[0],
             va='center', ha='left', fontsize=9, fontweight='bold')
    fig.text(0.01, gap_center, row_labels[1],
             va='center', ha='left', fontsize=9, fontweight='bold')

    plot_path = output_dir / filename
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    print(f"[analyze_model] Comparison plot saved to {plot_path}")


if __name__ == '__main__':
    import argparse
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    from models import *
    from timm.models import create_model

    parser = argparse.ArgumentParser(description='Multi-model roofline latency comparison')
    parser.add_argument('--models', nargs='+', default=['mobilenetv4_conv_small', 'mobilenetv4_dynamic_conv_small', 'mobilenetv4_ode_conv_small', 'mobilenetv4_dynamic_ode_conv_small'])
    parser.add_argument('--nb-classes', default=5, type=int)
    parser.add_argument('--input-size', default=384, type=int)
    parser.add_argument('--memory-bandwidth', default=47.0, type=float,
                        help='Memory bandwidth in GB/s')
    parser.add_argument('--output-dir', default='./output/analysis')
    parser.add_argument('--ode-num-steps', default=10, type=int)
    parser.add_argument('--extra-attention-block', action='store_true', default=False)
    args = parser.parse_args()

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    all_stats = {}
    for model_name in args.models:
        print(f"[analyze_model] Creating model: {model_name}")
        model = create_model(
            model_name,
            extra_attention_block=args.extra_attention_block,
            ode_num_steps=args.ode_num_steps,
            args=args,
        )
        model.reset_classifier(num_classes=args.nb_classes)
        model.to(device)
        
        # Standard collection
        stats = _collect_stats(model, args.input_size, args.memory_bandwidth, use_fused_ode=False)
        all_stats[model_name] = stats
        print(f"[analyze_model]   -> {len(stats)} layers collected for {model_name}")
        analyze_layer_stats(model_name, stats, args.memory_bandwidth, args.output_dir)
        
        # If it is an ODE model, also collect and analyze the fused variant
        if 'ode' in model_name:
            fused_name = model_name + "_fused"
            print(f"[analyze_model] Collecting fused variant: {fused_name}")
            stats_fused = _collect_stats(model, args.input_size, args.memory_bandwidth, use_fused_ode=True)
            all_stats[fused_name] = stats_fused
            print(f"[analyze_model]   -> {len(stats_fused)} layers collected for {fused_name}")
            analyze_layer_stats(fused_name, stats_fused, args.memory_bandwidth, args.output_dir)

    # Plot traditional comparison (original models, using standard ODE if any)
    traditional_stats = {m: all_stats[m] for m in args.models if m in all_stats}
    _plot_comparison(traditional_stats, args.memory_bandwidth, args.output_dir, filename="comparison_latency.png")

    # Plot fused comparison (replacing ODE models with their fused variants)
    fused_stats = {}
    for m in args.models:
        fused_m = m + "_fused"
        if fused_m in all_stats:
            fused_stats[fused_m] = all_stats[fused_m]
        elif m in all_stats:
            fused_stats[m] = all_stats[m]
    _plot_comparison(fused_stats, args.memory_bandwidth, args.output_dir, filename="comparison_latency_fused.png")

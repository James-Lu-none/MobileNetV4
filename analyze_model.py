import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import csv
from pathlib import Path
import matplotlib.patches as mpatches


OI_BOUNDS = [5, 500]

type_markers = {
    'Conv2d':             'o',
    'DW-Conv2d':          's',
    'FC':                 '^',
    'ODE':                'D',
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



def _collect_stats(model, input_size, bytes_per_element=4):
    stats = []
    hooks = []

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

    def make_ode_hook(name):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            _, _, H, W = out.shape
            num_steps   = mod.num_layers
            cout_sqrt   = mod.cout_sqrt
            out_ch      = mod.out_channels

            # matmul: (H*W, cout_sqrt, cout_sqrt) @ (cout_sqrt, cout_sqrt) per step
            matmul_macs = H * W * cout_sqrt ** 3
            # LayerNorm inside norm_relu per step
            ln_macs     = 2 * H * W * out_ch
            macs        = num_steps * (matmul_macs + ln_macs)

            weight_count = (mod.phi.numel()        # num_steps * out_ch
                            + mod.delta_t.numel()  # num_steps
                            + 2 * out_ch)          # LayerNorm weight + bias
            activation_count = H * W * out_ch

            memory_bytes = (weight_count + activation_count) * bytes_per_element
            if memory_bytes == 0:
                return

            stats.append({
                'name': name,
                'type': 'ODE',
                'macs': macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': macs / memory_bytes,
            })
        return hook

    def make_dynamic_conv_hook(name):
        def hook(mod, inp, out):
            if not isinstance(out, torch.Tensor):
                return
            out_h, out_w = out.shape[2], out.shape[3]
            
            # The main convolution MACs per sample:
            conv_macs = (out_h * out_w * mod.out_planes *
                         (mod.in_planes // mod.groups) *
                         mod.kernel_size * mod.kernel_size)
            
            # Attention overhead:
            attn_macs = 0
            if hasattr(mod, 'attention'):
                attn = mod.attention
                if hasattr(attn, 'fc1') and hasattr(attn, 'fc2'):
                    attn_macs += attn.fc1.weight.numel() + attn.fc2.weight.numel()
            
            # Weight aggregation overhead:
            agg_macs = mod.K * mod.out_planes * (mod.in_planes // mod.groups) * mod.kernel_size * mod.kernel_size
            
            total_macs = conv_macs + attn_macs + agg_macs
            
            # Weight count of the entire Dynamic_conv2d module
            weight_count = sum(p.numel() for p in mod.parameters())
            
            # Activation count (output features size)
            activation_count = out[0].numel() if out.dim() > 1 else out.numel()
            
            memory_bytes = (weight_count + activation_count) * bytes_per_element
            if memory_bytes == 0:
                return

            display_type = 'Dynamic-Conv2d'
            if mod.groups == mod.in_planes:
                display_type = 'Dynamic-DW-Conv2d'

            stats.append({
                'name': name,
                'type': display_type,
                'macs': total_macs,
                'weight_count': weight_count,
                'activation_count': activation_count,
                'oi_layer': total_macs / memory_bytes,
            })
        return hook

    # collect ids of modules that live inside an ODE or Dynamic_conv2d block (to skip them)
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
            h = module.register_forward_hook(make_ode_hook(name))
            hooks.append(h)
        elif _DYNAMIC_CONV_CLS is not None and isinstance(module, _DYNAMIC_CONV_CLS):
            h = module.register_forward_hook(make_dynamic_conv_hook(name))
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


def _plot_comparison(all_stats, beta_gb_per_s, output_dir, bytes_per_element=4):
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

    plot_path = output_dir / "comparison_latency.png"
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
    parser.add_argument('--models', nargs='+', required=True,
                        help='Model names to compare, e.g. --models modelA modelB modelC modelD')
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
        stats = _collect_stats(model, args.input_size)
        all_stats[model_name] = stats
        print(f"[analyze_model]   -> {len(stats)} layers collected")

        # Analyze and save individual model stats (CSV and Plot)
        analyze_layer_stats(model_name, stats, args.memory_bandwidth, args.output_dir)

    _plot_comparison(all_stats, args.memory_bandwidth, args.output_dir)

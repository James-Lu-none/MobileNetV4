import torch
import torch.nn as nn
import time
import sys
sys.path.append("/home/khyehlab/314581029/MobileNetV4")
from models.blocks_ode import ChannelwiseODESolver

def benchmark(fn, x, name, warmup=50, iters=200):
    for _ in range(warmup):
        _ = fn(x)
    torch.cuda.synchronize()
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    start_event.record()
    for _ in range(iters):
        _ = fn(x)
    end_event.record()
    end_event.synchronize()
    
    elapsed_time = start_event.elapsed_time(end_event) / iters
    print(f"{name:<30} | avg time: {elapsed_time:.4f} ms")
    return elapsed_time

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type != "cuda":
        print("This benchmark requires CUDA. Exiting.")
        return
    
    B, C, H, W = 32, 196, 14, 14
    num_layers = 10
    
    print(f"\nConfiguration:")
    print(f"  Batch size: {B}")
    print(f"  Channels: {C} (cout_sqrt: {int(C**0.5)})")
    print(f"  Spatial size: {H}x{W}")
    print(f"  ODE Steps: {num_layers}\n")

    x = torch.randn(B, C, H, W, device=device)

    # original eager mode
    solver = ChannelwiseODESolver(in_channels=C, out_channels=C, num_layers=num_layers).to(device).eval()
    eager_time = benchmark(solver._forward, x, "Eager Mode (Original)")

    # optimized mode
    compiled_fn = torch.compile(solver._forward)
    compiled_time = benchmark(compiled_fn, x, "Compiled Mode (Triton Fused)")

    speedup = eager_time / compiled_time
    print("=" * 55)
    print(f"Single Block Speedup: {speedup:.2f}x")
    print("=" * 55)

if __name__ == "__main__":
    main()

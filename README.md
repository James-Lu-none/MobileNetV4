# Optimizing MobileNetV4: Channelwise ODE Solvers and Dynamic Convolutions

This project explores computational capacity, parameter efficiency, and runtime scheduling constraints in deep neural networks on resource-constrained hardware. Using MobileNetV4 as a baseline, we replace pointwise (1x1) convolutions with a recurrent [Channelwise ODE Solver (COS)](./models/blocks_ode.py) and depthwise convolutions with [Dynamic Convolutions](./models/blocks_dynamic.py). 

Additionally, we perform runtime execution profiling to analyze CPU-to-GPU kernel launch overheads introduced by the recurrent ODE solver, and demonstrate how compilation-driven operator fusion via Triton mitigates this latency bottleneck.

---

## Architectural Modifications & Implementation Details

### 1. Spatial Mixing: Dynamic Convolution
* **Source Module**: [models/blocks_dynamic.py](./models/blocks_dynamic.py)

**Design**: Standard depthwise convolutions are replaced by Dynamic Convolutions. The module routes input activations $x$ through a lightweight attention-based network (`attention2d`) to generate softmax-normalized routing weights $\pi_k(x)$ for $K$ distinct convolutional kernels $\{W_k\}_{k=1}^K$ (and optional biases $\{b_k\}_{k=1}^K$):

$$
\pi(x) = \text{Softmax}\left( \frac{\mathbf{W}_2 \cdot \text{ReLU}(\mathbf{W}_1 \cdot \text{AvgPool}(x))}{\tau} \right)
$$

where $\mathbf{W}_1$ and $\mathbf{W}_2$ represent linear/convolutional projection layers, and $\tau$ is the temperature parameter. The dynamic kernel $\tilde{W}(x)$ and dynamic bias $\tilde{b}(x)$ are aggregated linearly:

$$
\tilde{W}(x) = \sum_{k=1}^K \pi_k(x) \cdot W_k, \quad \tilde{b}(x) = \sum_{k=1}^K \pi_k(x) \cdot b_k
$$

The input feature maps are then convolved using the aggregated parameters:

$$
y = \text{Conv2d}(x; \tilde{W}(x), \tilde{b}(x))
$$

followed by Batch Normalization (BN) and Activation function:

$$
\text{Output} = \text{Activation}(\text{BN}(y))
$$

**Trade-off**: Increases spatial representation capacity and parameter counts (+17%), which yields a significant improvement in top-1 validation accuracy (+2.09%).

### 2. Channel Mixing: Channelwise ODE Solver (COS)
* **Source Module**: [models/blocks_ode.py](./models/blocks_ode.py)

**Design**: Pointwise (1x1) convolutions are replaced with a recurrent Channelwise ODE Solver. The layer-to-layer transformations are modeled as a continuous-depth dynamical system solved via Euler integration over $N$ discrete steps (default $N=10$) with step size $\Delta t$:

$$
\frac{dy}{dt} = -y(t) + f(\text{LN}(y(t) + W_c y(t)))
$$

$$
y_{t+1} = y_t + \Delta t \cdot \left( -y_t + \text{ReLU6}(\text{LayerNorm}(y_t + W_c y_t)) \right)
$$

We substitute traditional Batch Normalization with Layer Normalization to ensure compatibility with variable evaluation batch sizes.

**Trade-off**: The kernel weights $W_c \in \mathbb{R}^{\sqrt{C} \times \sqrt{C}}$ (as opposed to full $C \times C$ pointwise weights) are shared across the recurrent integration steps. This cuts channel-mixing parameter size by 24% while slightly improving validation accuracy (+0.35%).

### 3. Integrated Block: Dynamic ODE Convolution
* **Source Module**: [models/blocks_dynamic_ode.py](./models/blocks_dynamic_ode.py)

**Design**: Combines Dynamic Convolutions in the depthwise (spatial mixing) layers and the Channelwise ODE Solver in the pointwise (channel mixing) layers.

**Trade-off**: Achieves an 8% overall reduction in parameters compared to baseline, but final accuracy gains (+0.47%) are lower than using Dynamic Convolutions alone.

---

## Runtime Profiling & Systems Optimization

### The Kernel Launch Bottleneck
When evaluated in PyTorch's default **Eager Mode**, the recurrent ODE solver incurs severe performance degradation on GPUs, resulting in a **50x slowdown** relative to baseline convolutions.
* **Launch Queues**: In eager mode, every operation within the 10-step integration loop (matrix multiplications, LayerNorm reductions, ReLU6 activations, and element-wise arithmetic) launches an independent CUDA kernel. This leads to more than 60 sequential kernel launches per UIB block.
* **CPU-Bound Scheduling**: Because the GPU execution time of these small kernels is extremely short, the CPU launch latency (driver translation, API overhead, context queues) dominates execution. The GPU sits idle waiting for the next launch call.
* **Memory Traffic**: Eager execution forces the GPU to write intermediate output tensors back to VRAM (DRAM) between consecutive operations, saturating memory bandwidth.

### Operator Fusion via Triton Compilation
To bypass this scheduling bottleneck, we compile the forward logic using PyTorch Inductor:
* **Generated Kernel**: [fuzed_ode_triton.py](/fuzed_ode_triton.py)
* **Mechanics**: Compilation fuses the LayerNorm reductions, ReLU6 thresholds, and Euler updates into a single Triton-optimized GPU kernel per step.
* **Result**: Fused execution retains intermediate feature tensors in GPU registers and SRAM instead of roundtripping them to VRAM. This reduces global memory traffic and minimizes GPU launch queues, yielding a **3.6x speedup** at the block level.

---

## Experimental Results

### 1. Accuracy and Parameter Count Trade-offs
Models were trained and evaluated on the Flower classification dataset with an input resolution of $384 \times 384$.

| Model Configuration | Parameters vs. Baseline | Validation Accuracy | Accuracy Change ($\Delta$) |
| :--- | :---: | :---: | :---: |
| **mobilenetv4_conv_small** (Baseline) | 100% | 81.71% | Baseline |
| **mobilenetv4_ode_conv_small** (COS) | -24% | 82.06% | **+0.35%** |
| **mobilenetv4_dynamic_conv_small** | +17% | 83.80% | **+2.09%** |
| **mobilenetv4_dynamic_ode_conv_small** | -8% | 82.18% | **+0.47%** |

---

### 2. Hyperparameter Ablation: Integration Step-Limit ($\epsilon$)
$\epsilon$ establishes a lower bound for the step size $\Delta t$ to prevent gradient vanishing. We compare coarse steps ($\epsilon=1.0$) against fine steps ($\epsilon=0.1$).

| Model Architecture | Epsilon ($\epsilon$) | Max Validation Accuracy | Empirical Analysis & Observations |
| :--- | :---: | :---: | :--- |
| **ode_conv** | 1.0 | 79.86% | Coarser steps smooth out loss landscapes during training but restrict the model's capacity to fit fine-grained patterns. |
| **ode_conv** | 0.1 | 82.06% | Finer integration steps track complex feature trajectories more accurately, though they can introduce transient extreme loss values. |
| **dynamic_ode_conv** | 1.0 | 76.62% | Coarse step sizes limit optimization convergence when combined with dynamic weights. |
| **dynamic_ode_conv** | 0.1 | 82.18% | Finer steps ensure stable gradient flow and convergence when joint spatial/channel optimization is active. |

---

### 3. End-to-End Latency Benchmark (Eager Mode)
*Measured with Batch Size = 32, Input Resolution = 224x224.*

| Model Configuration | CPU Latency | GPU Latency | GPU Slowdown Factor |
| :--- | :---: | :---: | :---: |
| **mobilenetv4_conv_small** (Baseline) | 38.671 ms | 1.375 ms | 1.00x |
| **mobilenetv4_dynamic_conv_small** | 50.545 ms | 2.606 ms | 1.89x |
| **mobilenetv4_ode_conv_small** | 183.891 ms | 68.259 ms | 49.64x |
| **mobilenetv4_dynamic_ode_conv_small** | 237.170 ms | 69.300 ms | 50.40x |

---

### 4. GPU Speedup via Triton Operator Fusion
*Comparison of Eager GPU latency vs. Compiled GPU latency. Batch Size = 32, ODE Steps = 10.*

| Model Configuration | GPU (Eager Mode) | GPU (Compiled/Fused) | Net Speedup |
| :--- | :---: | :---: | :---: |
| **mobilenetv4_conv_small** (Baseline) | 1.375 ms | - | - |
| **mobilenetv4_dynamic_conv_small** | 2.606 ms | - | - |
| **mobilenetv4_ode_conv_small** | 68.259 ms | 21.701 ms | **3.14x** |
| **mobilenetv4_dynamic_ode_conv_small** | 69.300 ms | 16.840 ms | **4.11x** |

> **Note**: Compiling the recurrent ODE loops provides an average **3.6x speedup** on GPU.

---

### 5. Block-Level Latency Breakdown
*Benchmarked on a Stage 3 ODE Block (Batch Size = 32, Channels = 196, H = 14, W = 14, ODE Steps = 10).*

| Block Implementation | Layer Latency | Block Speedup |
| :--- | :---: | :---: |
| **Original COS** (Eager PyTorch) | 1.5941 ms | 1.00x |
| **Fused COS** (Triton Compiled) | 0.4423 ms | **3.60x** |

---

## Project Repository Structure

```
├── models/
│   ├── blocks_common.py: Core helper utilities and shared layers.
│   ├── blocks_dynamic.py: Dynamic Convolution and attention routing logic.
│   ├── blocks_ode.py: Channelwise ODE Solver block definitions.
│   ├── blocks_dynamic_ode.py: Integrated Dynamic Conv and ODE UIB block.
│   ├── build_mobilenet_v4_base.py: Standard baseline model registry.
│   ├── build_mobilenet_v4_dynamic.py: Dynamic Conv variant registry.
│   ├── build_mobilenet_v4_ode.py: ODE Conv variant registry.
│   ├── build_mobilenet_v4_dynamic_ode.py: Dynamic ODE variant registry.
│   └── model_utils.py: Model arch string parsers and weight initializers.
├── fuzed_ode_triton.py: Compiled LayerNorm + ReLU6 + ODE update Triton kernel.
├── benchmark_single_solver.py: Micro-benchmarking script for eager vs. compiled blocks.
├── cal_inference_time.py: End-to-end model latency measurement utility.
├── analyze_model.py: Custom roofline latency projections and statistical collector.
├── plot_learning_curves.py: Log parser to plot loss curves and training schedules.
├── train_gpu.py: Main training/validation script.
└── environment.yml: Conda virtual environment configuration.
```

---

## Setup & Running Guide

### 1. Build the Virtual Environment
Construct the virtual environment and install PyTorch with CUDA 12.8 support:
```bash
conda env create -f environment.yml
conda activate mobilenetv4
pip install torch==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

### 2. Run Latency Benchmarks
* **Single Block Micro-benchmark**:
  Compare Eager PyTorch vs. Compiled Triton performance on a single ODE block:
  ```bash
  python benchmark_single_solver.py
  ```
* **End-to-End Latency Benchmark**:
  Benchmark inference time for baseline and modified models across CPU and GPU:
  ```bash
  python cal_inference_time.py --batch-size 32
  ```
* **Roofline Model Projections**:
  Run roofline performance estimations on layers:
  ```bash
  python analyze_model.py --models mobilenetv4_conv_small mobilenetv4_ode_conv_small --input-size 384
  ```

### 3. Model Training
To train the Channelwise ODE Solver model on the Flower dataset:
```bash
# download Flower dataset
kaggle datasets download -d alxmamaev/flowers-recognition
unzip flowers.zip -d datasets/flowers/

# base line
python train_gpu.py --model mobilenetv4_conv_small --data_root ./datasets/flowers --batch-size 32
# ode conv
python train_gpu.py --model mobilenetv4_ode_conv_small --data_root ./datasets/flowers --batch-size 32
# dynamic conv
python train_gpu.py --model mobilenetv4_dynamic_conv_small --data_root ./datasets/flowers --batch-size 32
# dynamic ode conv
python train_gpu.py --model mobilenetv4_dynamic_ode_conv_small --data_root ./datasets/flowers --batch-size 32
```

## reference

* Dynamic Convolution: Attention over Convolution Kernels https://arxiv.org/pdf/1912.03458
* MobileODE: An Extra Lightweight Network https://neurips.cc/virtual/2025/loc/san-diego/poster/115654
* MobileNetV2 https://arxiv.org/pdf/1801.04381
* MobileNetV3 https://arxiv.org/pdf/1905.02244
* MobileNetV4 https://arxiv.org/pdf/2404.10518
* PyTorch Image Models https://huggingface.co/timm
* https://medium.com/@tahasamavati/squeeze-and-excitation-explained-387b5981f249
* https://github.com/tensorflow/models/tree/master/official/vision/modeling/backbones
* https://github.com/tensorflow/models/blob/d93c7e932de27522b2fa3b115f58d06d6f640537/official/vision/modeling/layers/nn_blocks.py#L1504
* https://zhuanlan.zhihu.com/p/208519425
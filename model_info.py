import torch
import models
from models.build_mobilenet_v4 import (
    mobilenetv4_conv_small_035,
    mobilenetv4_conv_small_050,
    mobilenetv4_conv_small,
    mobilenetv4_dynamic_conv_small,
    mobilenetv4_ode_conv_small,
    mobilenetv4_dynamic_ode_conv_small
)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params

def main():
    model_funcs = {
        "mobilenetv4_conv_small_035": mobilenetv4_conv_small_035,
        "mobilenetv4_conv_small_050": mobilenetv4_conv_small_050,
        "mobilenetv4_conv_small": mobilenetv4_conv_small,
        "mobilenetv4_dynamic_conv_small": mobilenetv4_dynamic_conv_small,
        "mobilenetv4_ode_conv_small": mobilenetv4_ode_conv_small,
        "mobilenetv4_dynamic_ode_conv_small": mobilenetv4_dynamic_ode_conv_small,
    }
    
    print(f"{'Model Name':<40} | {'Total Params':<15} | {'Trainable Params':<15}")
    print("-" * 76)
    
    for name, func in model_funcs.items():
        try:
            model = func()
            total, trainable = count_parameters(model)
            print(f"{name:<40} | {total:<15,} | {trainable:<15,}")
        except Exception as e:
            print(f"Error loading {name}: {e}")

if __name__ == '__main__':
    main()

import torch
import torch.nn as nn
import models
from models.build_mobilenet_v4 import (
    mobilenetv4_conv_small_035,
    mobilenetv4_conv_small_050,
    mobilenetv4_conv_small,
    mobilenetv4_dynamic_conv_small,
    mobilenetv4_ode_conv_small,
    mobilenetv4_dynamic_ode_conv_small
)
from collections import defaultdict

def analyze_model_parameters(model):
    category_params = defaultdict(int)
    category_modules = defaultdict(set)
    
    for p_name, p in model.named_parameters():
        parts = p_name.split('.')
        mod = model
        path_modules = [mod]
        for part in parts[:-1]:
            if part.isdigit():
                mod = mod[int(part)]
            else:
                mod = getattr(mod, part)
            path_modules.append(mod)
            
        category = None
        owner_mod = None
        
        # Check if any parent is ODE or Dynamic Conv
        for m in reversed(path_modules):
            class_name = type(m).__name__
            if class_name == 'ChannelwiseODESolver':
                category = 'ODE'
                owner_mod = m
                break
            elif class_name == 'Dynamic_conv2d':
                if m.groups == m.in_planes:
                    category = 'Dynamic-DW-Conv2d'
                else:
                    category = 'Dynamic-Conv2d'
                owner_mod = m
                break
                
        if category is None:
            parent = path_modules[-1]
            owner_mod = parent
            if isinstance(parent, nn.Conv2d):
                if parent.groups == parent.in_channels and parent.in_channels > 1:
                    category = 'DW-Conv2d'
                elif parent.kernel_size == (1, 1) or parent.kernel_size[0] == 1:
                    category = 'PW-Conv2d'
                else:
                    category = 'Conv2d'
            elif isinstance(parent, nn.Linear):
                category = 'FC'
            elif isinstance(parent, (nn.BatchNorm2d, nn.BatchNorm1d, nn.LayerNorm, nn.GroupNorm)):
                category = 'Norm'
            else:
                category = 'Other'
                
        category_params[category] += p.numel()
        category_modules[category].add(id(owner_mod))
        
    categories = [
        'Conv2d', 'PW-Conv2d', 'DW-Conv2d', 'ODE', 
        'Dynamic-Conv2d', 'Dynamic-DW-Conv2d', 'FC', 'Norm', 'Other'
    ]
    
    results = {}
    total_counted = 0
    for cat in categories:
        count = len(category_modules[cat])
        params = category_params[cat]
        if count > 0 or params > 0:
            results[cat] = {'count': count, 'params': params}
            total_counted += params
            
    return results, total_counted

def main():
    model_funcs = {
        "mobilenetv4_conv_small": mobilenetv4_conv_small,
        "mobilenetv4_dynamic_conv_small": mobilenetv4_dynamic_conv_small,
        "mobilenetv4_ode_conv_small": mobilenetv4_ode_conv_small,
        "mobilenetv4_dynamic_ode_conv_small": mobilenetv4_dynamic_ode_conv_small,
    }
    
    for name, func in model_funcs.items():
        print(f"\n==================================================")
        print(f" Model: {name}")
        print(f"==================================================")
        try:
            model = func()
            results, total_sum = analyze_model_parameters(model)
            
            print(f"{'Block/Layer Type':<25} | {'Count':<10} | {'Parameters':<15}")
            print("-" * 56)
            for cat, data in results.items():
                print(f"{cat:<25} | {data['count']:<10} | {data['params']:<15,}")
            print("-" * 56)
            print(f"{'Total Parameter Sum':<25} | {'':<10} | {total_sum:<15,}")
        except Exception as e:
            print(f"Error loading {name}: {e}")
    print("\n==================================================")

if __name__ == '__main__':
    main()

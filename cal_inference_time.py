import time
import torch
import torch.nn as nn
import argparse
import models
import numpy as np
from timm.models import create_model


parser = argparse.ArgumentParser(description='PyTorch MobileNetV4 Inference Speed Test')
# Model params
parser.add_argument('--model', default='all', type=str, metavar='MODEL',
                        choices=['all', 'mobilenetv4_conv_small', 'mobilenetv4_dynamic_conv_small', 'mobilenetv4_ode_conv_small', 'mobilenetv4_dynamic_ode_conv_small'],
                        help='Name of model to test')
parser.add_argument('--device', default='all', type=str, choices=['all', 'cpu', 'cuda'],
                        help='Device to run inference (default: all)')
parser.add_argument('--batch-size', default=32, type=int, help='batch size (default: 32)')
parser.add_argument('--img-size', default=224, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--nb-classes', type=int, default=5,
                    help='Number classes in datasets')
parser.add_argument('--ode-num-steps', default=10, type=int, help='Number of ODE steps for ODE conv blocks')


def do_pure_cpu_task():
    x = np.random.randn(1, 3, 512, 512).astype(np.float32)
    x = x * 1024 ** 0.5


@torch.inference_mode()
def cal_time3(model, x, model_name, device, batch_size):
    time_list = []
    is_cuda = 'cuda' in device and torch.cuda.is_available()
    
    num_warmup = 10 if is_cuda else 3
    num_iters = 50 if is_cuda else 15
    discard_first = 5 if is_cuda else 3

    # Warm up
    for _ in range(num_warmup):
        _ = model(x)
    if is_cuda:
        torch.cuda.synchronize()

    if is_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        for _ in range(num_iters):
            start_event.record()
            ret = model(x)
            end_event.record()
            end_event.synchronize()
            time_list.append(start_event.elapsed_time(end_event) / 1000)
    else:
        for _ in range(num_iters):
            t0 = time.perf_counter()
            ret = model(x)
            t1 = time.perf_counter()
            time_list.append(t1 - t0)

    avg_batch_time = sum(time_list[discard_first:]) / len(time_list[discard_first:])
    avg_img_time = avg_batch_time / batch_size
    avg_batch_time_ms = avg_batch_time * 1000
    avg_img_time_ms = avg_img_time * 1000
    print(f"{model_name:<35} | avg batch: {avg_batch_time_ms:.3f} ms | per image: {avg_img_time_ms:.4f} ms")


def main(args):
    if args.device == 'all':
        devices_to_test = ['cpu']
        if torch.cuda.is_available():
            devices_to_test.append('cuda')
    else:
        devices_to_test = [args.device]
        
    if args.model == 'all':
        models_to_test = [
            'mobilenetv4_conv_small',
            'mobilenetv4_dynamic_conv_small',
            'mobilenetv4_ode_conv_small',
            'mobilenetv4_dynamic_ode_conv_small'
        ]
    else:
        models_to_test = [args.model]

    for device in devices_to_test:
        print(f"\n==================================================")
        print(f" DEVICE: {device.upper()}")
        print(f"==================================================")
        for model_name in models_to_test:
            try:
                model = create_model(
                    model_name,
                    num_classes=args.nb_classes,
                    ode_num_steps=args.ode_num_steps
                )
                model.eval().to(device)

                x = torch.randn(size=(args.batch_size, 3, args.img_size, args.img_size), device=device)
                cal_time3(model, x, model_name, device, args.batch_size)
            except Exception as e:
                print(f"Error testing {model_name} on {device}: {e}")
        print(f"==================================================")
    print()


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
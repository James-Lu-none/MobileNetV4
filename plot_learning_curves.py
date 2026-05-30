import argparse
import os
import glob
import json
import math
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def plot_learning_curves(log_path, output_dir):
    epochs = []
    train_loss = []
    test_loss = []
    test_acc1 = []
    test_acc5 = []
    train_lr = []

    if not os.path.exists(log_path):
        return

    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                epochs.append(entry.get('epoch', 0))
                train_loss.append(entry.get('train_loss'))
                test_loss.append(entry.get('test_loss'))
                test_acc1.append(entry.get('test_acc1'))
                test_acc5.append(entry.get('test_acc5'))
                train_lr.append(entry.get('train_lr'))
            except Exception:
                continue

    def sanitize(val_list):
        res = []
        for v in val_list:
            if v is None:
                res.append(float('nan'))
            elif isinstance(v, float) and math.isnan(v):
                res.append(float('nan'))
            elif isinstance(v, str) and v.lower() == 'nan':
                res.append(float('nan'))
            else:
                res.append(float(v))
        return res

    train_loss = sanitize(train_loss)
    test_loss = sanitize(test_loss)
    test_acc1 = sanitize(test_acc1)
    test_acc5 = sanitize(test_acc5)
    train_lr = sanitize(train_lr)

    if not epochs:
        return

    # Find min/max values for annotation
    valid_train_loss = [x for x in train_loss if not math.isnan(x)]
    valid_test_loss = [x for x in test_loss if not math.isnan(x)]
    valid_test_acc1 = [x for x in test_acc1 if not math.isnan(x)]
    valid_test_acc5 = [x for x in test_acc5 if not math.isnan(x)]

    min_train_loss = min(valid_train_loss) if valid_train_loss else None
    min_test_loss = min(valid_test_loss) if valid_test_loss else None
    max_test_acc1 = max(valid_test_acc1) if valid_test_acc1 else None
    max_test_acc5 = max(valid_test_acc5) if valid_test_acc5 else None

    fig, axs = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Loss
    axs[0].plot(epochs, train_loss, label='Train Loss', color='royalblue', marker='o', markersize=4)
    if not all(math.isnan(x) for x in test_loss):
        axs[0].plot(epochs, test_loss, label='Test Loss', color='orange', marker='s', markersize=4)
    if min_train_loss is not None:
        axs[0].axhline(y=min_train_loss, color='royalblue', linestyle='--', alpha=0.5, label=f'Min Train Loss: {min_train_loss:.4f}')
    if min_test_loss is not None:
        axs[0].axhline(y=min_test_loss, color='orange', linestyle='--', alpha=0.5, label=f'Min Test Loss: {min_test_loss:.4f}')
    axs[0].set_title('Loss Curve')
    axs[0].set_xlabel('Epoch')
    axs[0].set_ylabel('Loss')
    axs[0].grid(True, linestyle='--', alpha=0.6)
    axs[0].legend()

    # Plot 2: Accuracy
    axs[1].plot(epochs, test_acc1, label='Test Acc@1', color='forestgreen', marker='^', markersize=4)
    axs[1].plot(epochs, test_acc5, label='Test Acc@5', color='crimson', marker='v', markersize=4)
    if max_test_acc1 is not None:
        axs[1].axhline(y=max_test_acc1, color='forestgreen', linestyle='--', alpha=0.5, label=f'Max Acc@1: {max_test_acc1:.2f}%')
    if max_test_acc5 is not None:
        axs[1].axhline(y=max_test_acc5, color='crimson', linestyle='--', alpha=0.5, label=f'Max Acc@5: {max_test_acc5:.2f}%')
    axs[1].set_title('Validation Accuracy')
    axs[1].set_xlabel('Epoch')
    axs[1].set_ylabel('Accuracy (%)')
    axs[1].grid(True, linestyle='--', alpha=0.6)
    axs[1].legend()

    # Plot 3: Learning Rate
    axs[2].plot(epochs, train_lr, label='Learning Rate', color='purple', marker='d', markersize=4)
    axs[2].set_title('Learning Rate Schedule')
    axs[2].set_xlabel('Epoch')
    axs[2].set_ylabel('LR')
    axs[2].grid(True, linestyle='--', alpha=0.6)
    axs[2].legend()

    plt.tight_layout()
    save_img_path = os.path.join(output_dir, 'learning_curves.png')
    plt.savefig(save_img_path, dpi=150)
    plt.close()
    print(f"Successfully generated learning curves at: {save_img_path}")


def main():
    parser = argparse.ArgumentParser(description="Standalone script to plot learning curves from model training logs.")
    parser.add_argument('--output-dir', default='./output', type=str, help='Path to the directory containing output logs (default: ./output)')
    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        print(f"Error: Output directory '{args.output_dir}' does not exist.")
        return

    # Find all log.txt files under output-dir
    log_files = glob.glob(os.path.join(args.output_dir, '**', 'log.txt'), recursive=True)
    
    if not log_files:
        print(f"No log.txt files found in '{args.output_dir}'")
        return

    print(f"Found {len(log_files)} training run logs. Plotting...")
    for log_path in log_files:
        run_dir = os.path.dirname(log_path)
        plot_learning_curves(log_path, run_dir)

if __name__ == '__main__':
    main()

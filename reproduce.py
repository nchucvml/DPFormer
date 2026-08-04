"""
Reproduce incremental accuracy from saved checkpoints.

The checkpoint root is a setting directory that contains one or more `orderN`
subfolders (e.g. checkpoints/cifar100/10 -> order1, order2, order3). Each order is
evaluated with its matching class order from options/order/<dataset>_orderN.yaml,
and the final numbers are averaged over all orders found.

Usage (dataset and increment are inferred from the --ckpt_root path):
    python reproduce.py --ckpt_root ./checkpoints/cifar100/10 --data_path datasets/cifar100
    python reproduce.py --ckpt_root ./checkpoints/imagenet100/10 --data_path datasets/imagenet1000
    python reproduce.py --ckpt_root ./checkpoints/imagenet1000/100 --data_path datasets/imagenet1000
"""
import argparse
import os
import re
import statistics
from types import SimpleNamespace

import torch
import yaml
from torch.utils.data import DataLoader

from continual import backbone
from continual.datasets import build_dataset
from continual.dpt import DPT


# total_classes and default input_size per dataset
DATASET_INFO = {
    'cifar100': {'total_classes': 100, 'input_size': 32},
    'imagenet100': {'total_classes': 100, 'input_size': 224},
    'imagenet1000': {'total_classes': 1000, 'input_size': 224},
}


def get_args():
    parser = argparse.ArgumentParser('Reproduce incremental accuracy, averaged over task orders')
    parser.add_argument('--data_path', required=True, help='Path to the dataset')
    parser.add_argument('--ckpt_root', required=True,
                        help='Setting directory like ./checkpoints/<dataset>/<increment> '
                             '(dataset and increment are inferred from the last two path parts)')
    parser.add_argument('--dataset', default=None,
                        help='cifar100 | imagenet100 | imagenet1000 (default: inferred from --ckpt_root)')
    parser.add_argument('--increment', default=None, type=int,
                        help='Classes per task (default: inferred from --ckpt_root)')
    parser.add_argument('--ckpt_prefix', default='checkpoint', help='Checkpoint filename prefix')
    parser.add_argument('--total_classes', default=None, type=int, help='Override total classes')
    parser.add_argument('--input_size', default=None, type=int, help='Override input size')
    # Architecture (defaults match options/config.yaml)
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--num_heads', default=12, type=int)
    parser.add_argument('--stages', default=3, type=int)
    parser.add_argument('--num_extractor', default=1, type=int)
    parser.add_argument('--class_to_task', default=2, type=int)
    parser.add_argument('--pretrained_model', default=None)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def find_order_dirs(ckpt_root):
    """Return sorted [(order_idx, path), ...] for orderN subfolders under ckpt_root.
    If none exist, treat ckpt_root itself as a single order-1 run.
    """
    entries = []
    if os.path.isdir(ckpt_root):
        for name in os.listdir(ckpt_root):
            m = re.match(r'(?i)^order(\d+)$', name)
            full = os.path.join(ckpt_root, name)
            if m and os.path.isdir(full):
                entries.append((int(m.group(1)), full))
    if not entries:
        return [(1, ckpt_root)]
    entries.sort()
    return entries


def find_order_yaml(dataset, order_idx):
    """Locate options/order/<dataset>_order<idx>.yaml (case-insensitive)."""
    data_dir = os.path.join('options', 'order')
    want = f'{dataset}_order{order_idx}.yaml'.lower()
    for f in os.listdir(data_dir):
        if f.lower() == want:
            return os.path.join(data_dir, f)
    return None


def load_class_order(yaml_path):
    with open(yaml_path, 'r') as f:
        return yaml.safe_load(f)['class_order']


def build_scenario(args, class_order):
    ds_args = SimpleNamespace(
        dataset=args.dataset,
        data_path=args.data_path,
        input_size=args.input_size,
        increment=args.increment,
        augmentation='none',
        class_order=class_order,
    )
    return build_dataset(is_train=False, args=ds_args)


def build_model(args, nb_tasks, device):
    transformer = backbone.Blocks(
        input_size=args.input_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        stages=args.stages,
        num_extractor=args.num_extractor,
        class_to_task=args.class_to_task,
        pretrained_model=args.pretrained_model,
    )
    model = DPT(transformer, nb_classes=args.increment, head_div=True)
    for _ in range(nb_tasks - 1):
        model.add_model(args.increment)
    return model.to(device)


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    model.load_state_dict(state_dict, strict=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        preds = model(images)['logits'].argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def run_order(args, order_dir, class_order, nb_tasks, device):
    """Evaluate one order. Returns (avg_incremental_acc, last_acc) or None if no checkpoints."""
    scenario = build_scenario(args, class_order)
    accs = []
    for task_id in range(nb_tasks):
        ckpt_path = os.path.join(order_dir, f'{args.ckpt_prefix}_{task_id}.pth')
        if not os.path.exists(ckpt_path):
            print(f"    [SKIP] task {task_id}: missing {ckpt_path}")
            continue
        model = build_model(args, task_id + 1, device)
        load_checkpoint(model, ckpt_path)
        loader = DataLoader(scenario[:task_id + 1], batch_size=args.batch_size,
                            shuffle=False, num_workers=args.num_workers, pin_memory=True)
        acc = evaluate(model, loader, device)
        accs.append(acc)
        print(f"    task {task_id} ({(task_id + 1) * args.increment:>3} classes): {acc:.2f}%")
        del model
    if not accs:
        return None
    return statistics.mean(accs), accs[-1]


def main():
    args = get_args()

    parts = os.path.normpath(args.ckpt_root).split(os.sep)
    if args.dataset is None:
        args.dataset = parts[-2]
    if args.increment is None:
        args.increment = int(parts[-1])

    key = args.dataset.lower()
    if key not in DATASET_INFO:
        raise ValueError(f'Unknown dataset {args.dataset}. Choose from {list(DATASET_INFO)}.')
    if args.total_classes is None:
        args.total_classes = DATASET_INFO[key]['total_classes']
    if args.input_size is None:
        args.input_size = DATASET_INFO[key]['input_size']

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nb_tasks = args.total_classes // args.increment

    order_dirs = find_order_dirs(args.ckpt_root)
    print(f"Dataset: {args.dataset} | increment: {args.increment} | tasks: {nb_tasks} "
          f"| input_size: {args.input_size}")
    print(f"Found {len(order_dirs)} order(s) under {args.ckpt_root}")
    print(f"Device: {device}\n")

    avg_scores, last_scores = [], []
    for order_idx, order_dir in order_dirs:
        yaml_path = find_order_yaml(key, order_idx)
        if yaml_path is None:
            print(f"[SKIP] order{order_idx}: no options/order/{args.dataset}_order{order_idx}.yaml")
            continue
        print(f"=== order{order_idx}  ({order_dir}) ===")
        print(f"    class order from: {yaml_path}")
        class_order = load_class_order(yaml_path)
        result = run_order(args, order_dir, class_order, nb_tasks, device)
        if result is None:
            print(f"    [SKIP] no checkpoints found\n")
            continue
        avg_acc, last_acc = result
        avg_scores.append(avg_acc)
        last_scores.append(last_acc)
        print(f"    -> Avg Incremental Acc: {avg_acc:.2f}%  |  Last: {last_acc:.2f}%\n")

    if not avg_scores:
        print("No orders evaluated.")
        return

    avg_rounded = [round(a, 2) for a in avg_scores]
    last_rounded = [round(a, 2) for a in last_scores]
    print('=' * 55)
    print(f"Orders evaluated: {len(avg_scores)}")
    print(f"Per-order Avg : {avg_rounded}")
    print(f"Per-order Last: {last_rounded}")
    print(f"Mean Avg Incremental Acc: {statistics.mean(avg_rounded):.2f}%")
    print(f"Mean Last Acc          : {statistics.mean(last_rounded):.2f}%")
    print('=' * 55)


if __name__ == '__main__':
    main()

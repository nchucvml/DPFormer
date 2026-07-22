"""
Reproduce incremental accuracy on CIFAR-100 test set.
For each task t, loads checkpoint_t.pth and evaluates on all classes seen so far (tasks 0~t).

Usage:
    python test.py --ckpt_dir ./checkpoints/cifar100/10/2024-01-01/exp_name
    python test.py --ckpt_dir . --ckpt_prefix checkpoint --data_path datasets/cifar100
"""
import argparse
import os
import statistics
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from continuum import ClassIncremental
from continuum.datasets import CIFAR100

from continual import backbone
from continual.dpt import DPT


CLASS_ORDER_1 = [
    87, 0, 52, 58, 44, 91, 68, 97, 51, 15, 94, 92, 10, 72, 49, 78, 61, 14, 8, 86,
    84, 96, 18, 24, 32, 45, 88, 11, 4, 67, 69, 66, 77, 47, 79, 93, 29, 50, 57, 83,
    17, 81, 41, 12, 37, 59, 25, 20, 80, 73, 1, 28, 6, 46, 62, 82, 53, 9, 31, 75,
    38, 63, 33, 74, 27, 22, 36, 3, 16, 21, 60, 19, 70, 90, 89, 43, 5, 42, 65, 76,
    40, 30, 23, 85, 2, 95, 56, 48, 71, 64, 98, 13, 99, 7, 34, 55, 54, 26, 35, 39
]


def get_args():
    parser = argparse.ArgumentParser('Reproduce incremental accuracy on CIFAR-100')
    parser.add_argument('--ckpt_dir', default='.', help='Directory containing checkpoint_0.pth, checkpoint_1.pth, ...')
    parser.add_argument('--ckpt_prefix', default='checkpoint', help='Checkpoint filename prefix (default: checkpoint)')
    parser.add_argument('--data_path', default='datasets/cifar100', help='Path to CIFAR-100 dataset')
    parser.add_argument('--increment', default=10, type=int, help='Classes per task')
    parser.add_argument('--total_classes', default=100, type=int, help='Total number of classes')
    parser.add_argument('--embed_dim', default=384, type=int)
    parser.add_argument('--num_heads', default=12, type=int)
    parser.add_argument('--stages', default=3, type=int)
    parser.add_argument('--num_extractor', default=1, type=int)
    parser.add_argument('--class_to_task', default=2, type=int)
    parser.add_argument('--input_size', default=32, type=int)
    parser.add_argument('--batch_size', default=100, type=int)
    parser.add_argument('--num_workers', default=2, type=int)
    parser.add_argument('--device', default='cuda')
    return parser.parse_args()


def build_scenario(data_path, increment, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    dataset = CIFAR100(data_path, train=False, download=True)
    scenario = ClassIncremental(
        dataset,
        initial_increment=increment,
        increment=increment,
        transformations=transform.transforms,
        class_order=CLASS_ORDER_1,
    )
    return scenario


def build_model(args, nb_tasks, device):
    transformer = backbone.Blocks(
        input_size=args.input_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        stages=args.stages,
        num_extractor=args.num_extractor,
        class_to_task=args.class_to_task,
        pretrained_model=None,
    )
    model = DPT(transformer, nb_classes=args.increment, head_div=True)
    for _ in range(nb_tasks - 1):
        model.add_model(args.increment)
    model = model.to(device)
    return model


def remap_state_dict(state_dict):
    """Rename tfbs -> febs to match DPT model (old checkpoints saved with PTT used tfbs)."""
    remapped = {}
    renamed_count = 0
    for k, v in state_dict.items():
        new_k = k.replace('tfbs.', 'febs.', 1)
        if new_k != k:
            renamed_count += 1
        remapped[new_k] = v
    if renamed_count:
        print(f"  Remapped {renamed_count} keys: tfbs.* -> febs.*")
    return remapped


def load_checkpoint(model, ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    state_dict = ckpt['model'] if isinstance(ckpt, dict) and 'model' in ckpt else ckpt
    state_dict = remap_state_dict(state_dict)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [WARN] Missing keys  ({len(missing)}): {missing[:3]}{'...' if len(missing) > 3 else ''}")
    if unexpected:
        print(f"  [WARN] Unexpected keys ({len(unexpected)}): {unexpected[:3]}{'...' if len(unexpected) > 3 else ''}")
    if not missing and not unexpected:
        print(f"  All keys matched.")


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    for images, targets, _ in loader:
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        preds = outputs['logits'].argmax(dim=-1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    return 100.0 * correct / total


def main():
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    nb_tasks = args.total_classes // args.increment

    print(f"Config: {nb_tasks} tasks x {args.increment} classes = {args.total_classes} total classes")
    print(f"Checkpoint dir: {args.ckpt_dir}")
    print(f"Device: {device}\n")

    scenario = build_scenario(args.data_path, args.increment, args.batch_size, args.num_workers)

    accuracy_list = []

    for task_id in range(nb_tasks):
        ckpt_path = os.path.join(args.ckpt_dir, f'{args.ckpt_prefix}_{task_id}.pth')
        if not os.path.exists(ckpt_path):
            print(f"[SKIP] Task {task_id}: checkpoint not found at {ckpt_path}")
            continue

        print(f"--- Task {task_id} ({(task_id + 1) * args.increment} classes seen) ---")
        print(f"  Loading: {ckpt_path}")

        # Build a fresh model sized for this task
        model = build_model(args, task_id + 1, device)
        load_checkpoint(model, ckpt_path)

        # Evaluate on all classes seen so far (tasks 0 ~ task_id)
        test_set = scenario[:task_id + 1]
        loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        acc = evaluate(model, loader, device)
        accuracy_list.append(acc)
        print(f"  Accuracy (tasks 0~{task_id}): {acc:.2f}%")

    if accuracy_list:
        print(f"\n{'='*45}")
        print(f"  All accuracies:  {[round(a, 2) for a in accuracy_list]}")
        print(f"  Average Incremental Accuracy: {statistics.mean(accuracy_list):.2f}%")
        print(f"{'='*45}")


if __name__ == '__main__':
    main()

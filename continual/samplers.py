import math
import torch
import torch.distributed as dist

import continual.utils as utils

# Number of times each sample is repeated for Repeated Augmentation (RA)
NUM_AUGMENT_REPEATS = 3


class RASampler(torch.utils.data.Sampler):
    """Sampler that restricts data loading to a subset of the dataset for distributed
    training with repeated augmentation.
    It ensures that each augmented version of a sample will be visible to a
    different process (GPU)
    Heavily based on torch.utils.data.DistributedSampler
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, batch_size=256):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * NUM_AUGMENT_REPEATS / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        # align selected samples to batch_size * num_replicas for consistent steps per epoch
        align = batch_size * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // align * align / self.num_replicas))
        self.shuffle = shuffle

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # repeat each sample NUM_AUGMENT_REPEATS times for repeated augmentation
        indices = [ele for ele in indices for _ in range(NUM_AUGMENT_REPEATS)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class RASamplerNoDist(RASampler):
    """Single-GPU variant of RASampler that simulates multi-GPU repeated augmentation
    by alternating rank between iterations."""

    def __init__(self, dataset, num_replicas=None, shuffle=True, batch_size=256):
        if num_replicas is None:
            num_replicas = 2
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * NUM_AUGMENT_REPEATS / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        align = batch_size * self.num_replicas
        self.num_selected_samples = int(math.floor(len(self.dataset) // align * align / self.num_replicas))
        self.shuffle = shuffle
        self.rank = 0

    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))

        # repeat each sample NUM_AUGMENT_REPEATS times for repeated augmentation
        indices = [ele for ele in indices for _ in range(NUM_AUGMENT_REPEATS)]
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample for current rank, then rotate rank to alternate subsets each epoch
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples
        self.rank = (self.rank + 1) % self.num_replicas

        return iter(indices[:self.num_selected_samples])

    def __len__(self):
        return self.num_selected_samples


def get_sampler(dataset_train, dataset_val, args):
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            batch_size=args.batch_size)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    else:
        sampler_train = RASamplerNoDist(dataset_train, num_replicas=2, shuffle=True,
                                        batch_size=args.batch_size)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    return sampler_train, sampler_val


def get_train_sampler(dataset_train, args):
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        sampler_train = RASampler(
            dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True,
            batch_size=args.batch_size)
    else:
        sampler_train = torch.utils.data.RandomSampler(dataset_train)

    return sampler_train

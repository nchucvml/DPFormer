import torch

from continual import samplers, backbone, dpt


def get_backbone(args):
    print(f"Creating backbone.")
    model = backbone.Blocks(
        input_size=args.input_size,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        stages=args.stages,
        num_extractor=args.num_extractor,
        class_to_task=args.class_to_task,
        pretrained_model=args.pretrained_model)

    return model.to(args.device)


def build_dpt(base, task_id, args, dataset=None):
    """Build or expand the DPFormer (DPT) model for the given task.
    :param base: backbone (Blocks) for task 0, or the existing DPT for task > 0.
    :param dataset: Optional. Current task's training data for prototype initialization.
    """
    if task_id == 0:
        print(f'Creating DPFormer.')
        model = dpt.DPT(
            base,
            nb_classes=args.increment,
            head_div=args.head_div > 0.,
            dataset=dataset,
            args=args)
    else:
        print(f'Expanding DPFormer.')
        base.add_model(args.increment, dataset=dataset, args=args)
        model = base

    return model.to(args.device)


class InfiniteLoader:
    def __init__(self, loader):
        self.loader = loader
        self.reset()

    def reset(self):
        self.it = iter(self.loader)

    def get(self):
        try:
            return next(self.it)
        except StopIteration:
            self.reset()
            return next(self.it)


def _make_train_loader(dataset_train, args, batch_size=None, sampler=None, drop_last=True):
    return torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler,
        batch_size=batch_size or args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def get_loaders(dataset_train, dataset_val, args, finetuning=False):
    sampler_train, sampler_val = samplers.get_sampler(dataset_train, dataset_val, args)

    # finetuning uses a pre-balanced dataset, so no distributed sampler needed
    loader_train = _make_train_loader(
        dataset_train, args,
        sampler=None if finetuning else sampler_train,
        drop_last=not finetuning,
    )
    loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    return loader_train, loader_val


def get_train_loaders(dataset_train, args, batch_size=None):
    sampler_train = samplers.get_train_sampler(dataset_train, args)
    return _make_train_loader(dataset_train, args, batch_size=batch_size, sampler=sampler_train)

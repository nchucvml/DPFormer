from timm.utils import dispatch_clip_grad
import torch


class ContinualScaler:
    state_dict_key = "scaler"

    def __init__(self, disable_amp):
        self._scaler = torch.cuda.amp.GradScaler(enabled=not disable_amp)

    def __call__(
        self, loss, optimizer, model_without_ddp, parameters=None, create_graph=False,
        clip_grad=None, clip_mode='norm', hook=True
    ):
        self.pre_step(loss, optimizer, parameters, create_graph, clip_grad, clip_mode)
        self.post_step(optimizer, model_without_ddp, hook)

    def pre_step(self, loss, optimizer, parameters=None, create_graph=False, clip_grad=None, clip_mode='norm'):
        # create_graph=True needed for second-order optimizers (e.g. adahessian)
        self._scaler.scale(loss).backward(create_graph=create_graph)
        self._scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
        if clip_grad is not None:
            assert parameters is not None
            dispatch_clip_grad(parameters, clip_grad, mode=clip_mode)

    def post_step(self, optimizer, model_without_ddp, hook=True):
        if hook and hasattr(model_without_ddp, 'hook_before_update'):
            model_without_ddp.hook_before_update()

        self._scaler.step(optimizer)

        if hook and hasattr(model_without_ddp, 'hook_after_update'):
            model_without_ddp.hook_after_update()

        self.update()

    def update(self):
        self._scaler.update()

    def state_dict(self):
        return self._scaler.state_dict()

    def load_state_dict(self, state_dict):
        self._scaler.load_state_dict(state_dict)

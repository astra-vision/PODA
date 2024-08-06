# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.runner.hooks import HOOKS, Hook
from torch import autograd

@HOOKS.register_module()
class IgnoreInvalidLossHook(Hook):
    """Check invalid loss hook.

    This hook will regularly check whether the loss is valid
    during training.

    Args:
        interval (int): Checking interval (every k iterations).
            Default: 50.
    """

    def after_train_iter(self, runner):
        if self.every_n_iters(runner, 1):
            with autograd.detect_anomaly():
                runner.optimizer.zero_grad()
                runner.outputs['loss'].backward()
                allreduce_grads(runner.model, self.coalesce, self.bucket_size_mb)
                if self.grad_clip is not None:
                    self.clip_grads(runner.model.parameters())
                runner.optimizer.step()

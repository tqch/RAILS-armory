import torch
import torch.nn as nn


class PGD:
    
    def __init__(self, model, eps=60 / 255., step_size=20 / 255., max_iter=10, random_init=True,
                 targeted=False, loss_fn=nn.CrossEntropyLoss(), batch_size=64):
        if callable(model):
            self.model = model
        else:
            self.model = model._model
        self.model.eval()
        self.device = next(iter(self.model.parameters())).device
        self.eps = eps
        self.step_size = step_size
        self.max_iter = max_iter
        self.random_init = random_init
        self.targeted = targeted
        self.loss_fn = loss_fn
        self.batch_size = batch_size

    def attack(self, model, x, y, x_adv=None, targets=None):
        if x_adv is None:
            if self.random_init:
                x_adv = 2 * self.eps * (torch.rand_like(x) - 0.5) + x
                x_adv = x_adv.clamp(0.0, 1.0)
            else:
                x_adv = torch.clone(x).detach()
        x_adv.requires_grad_(True)
        pred_adv = model(x_adv)
        if isinstance(pred_adv, (list, tuple)):
            pred_adv = pred_adv[-1]
        if self.targeted:
            assert targets is not None, "Target labels not found!"
            loss = self.loss_fn(pred_adv, targets)
        else:
            loss = self.loss_fn(pred_adv, y)
        grad = torch.autograd.grad(loss, x_adv)[0]
        pert = self.step_size * grad.sign()
        x_adv = (x_adv + pert).clamp(0.0, 1.0).detach()
        pert = (x_adv - x).clamp(-self.eps, self.eps)
        return x + pert

    def generate(self, x, y=None, targets=None):
        if not isinstance(x, torch.Tensor):
            x = torch.FloatTensor(x)
        if y is not None and not isinstance(y, torch.Tensor):
            y = torch.LongTensor(y)
        x_adv = []
        for i in range(0, x.size(0), self.batch_size):
            x_batch = x[i: i + self.batch_size].to(self.device)
            if y is None:
                y_batch = self.model(x_batch)
                if isinstance(y_batch, tuple):
                    y_batch = y_batch[-1]
                y_batch = y_batch.max(dim=-1)[1].to(self.device)
            else:
                y_batch = y[i: i + self.batch_size].to(self.device)
            for j in range(self.max_iter):
                if j == 0:
                    x_adv_batch = self.attack(self.model, x_batch, y_batch, targets=targets)
                else:
                    x_adv_batch = self.attack(self.model, x_batch, y_batch, x_adv_batch, targets=targets)
            x_adv.append(x_adv_batch)
        return torch.cat(x_adv, dim=0).cpu().numpy()

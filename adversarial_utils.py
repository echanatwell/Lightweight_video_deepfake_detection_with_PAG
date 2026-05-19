import torch
from torch import nn


def pgd_attack(x: torch.Tensor, y_target: torch.Tensor, model: nn.Module, eps:float, loss_fn=nn.CrossEntropyLoss(), pgd_iterations=3):
    device = next(model.parameters()).device
    
    x_grad = x.detach().clone().requires_grad_().to(device)
    model.requires_grad_(False)

    for _ in range(pgd_iterations):
        if x_grad.grad is not None:
            x_grad.grad = None
        y_pred = model(x_grad)

        loss = loss_fn(y_pred, y_target)
        loss.backward()

        with torch.no_grad():
            x_grad = torch.clamp(x_grad + torch.clamp(torch.sign(x_grad.grad), -eps, eps), -1, 1).requires_grad_()

    model.requires_grad_()
    return x_grad.detach()

def fgsm_attack(x: torch.Tensor, y_target: torch.Tensor, model: nn.Module, eps:float, loss_fn=nn.CrossEntropyLoss()):
    x_grad = x.detach().clone().requires_grad_().to(model.device)

    y_pred = model.requires_grad_(False)(x_grad)
    loss_fn(y_pred, y_target).backward()
    x_grad = torch.clamp(x_grad + eps * torch.sign(x_grad.grad), -1, 1)

    return x_grad.detach()
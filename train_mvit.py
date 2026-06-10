import torch
from torch import nn
from torch.utils.data import DataLoader
from torchmetrics import F1Score
from torchvision.transforms import v2 as T

import cv2
import numpy as np
import time
from sklearn.metrics import classification_report

import matplotlib.pyplot as plt
from datasets.ffpp import FFPPDataset
from datasets.celebdf import CelebDFDataset
from model.model import Model
import torchvision
from torchinfo import summary


# ---------------------------------------------------------------------------
# Constants (module-level is fine — no side-effects)
# ---------------------------------------------------------------------------

DEVICE = 'cuda:0'
EPOCHS = 5
LR_0 = 0.0003
LR_N = 0.00001
BATCH_SIZE = 12
MAX_FRAMES_PER_VIDEO = 16
NUM_CLASSES = 2
IMG_SIZE = 224

GRADIENT_ACCUMULATION_STEPS = 4

EXP_NAME = "MViT_CelebDF"


# ---------------------------------------------------------------------------
# Entry point — required on Windows (spawn) to prevent recursive worker launch
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    train_loss_history = list()
    val_loss_history = list()
    val_f1_history = list()
    val_adv_loss_history = list()
    val_adv_f1_history = list()

    model = torchvision.models.video.mvit_v2_s(torchvision.models.video.MViT_V2_S_Weights.DEFAULT)
    model.eval()
    model.head = nn.Sequential(
        nn.Dropout(0.1),
        nn.Linear(768, 2)
    )
    model.to(DEVICE)

    summary(model, input_size=(BATCH_SIZE, 3, MAX_FRAMES_PER_VIDEO, IMG_SIZE, IMG_SIZE))

    transforms = torchvision.models.video.MViT_V2_S_Weights.DEFAULT.transforms()
    train_transforms = transforms
    test_transforms = transforms

    dataset_path = '/home/peter/faigc/data/Celeb-DF-v2'
    # dataset_path = '/home/peter/faigc/data/ff++'

    train_dataset = CelebDFDataset(dataset_path=dataset_path, transforms=train_transforms,
                                       frames_per_video=MAX_FRAMES_PER_VIDEO, split='train')
    val_dataset = CelebDFDataset(dataset_path=dataset_path, transforms=test_transforms,
                                     frames_per_video=MAX_FRAMES_PER_VIDEO, split='validation')
    test_dataset = CelebDFDataset(dataset_path=dataset_path, transforms=test_transforms,
                                      frames_per_video=MAX_FRAMES_PER_VIDEO, split='test')

    train_loader = DataLoader(train_dataset, BATCH_SIZE, shuffle=True, num_workers=10, drop_last=True, pin_memory=True)
    val_loader = DataLoader(val_dataset, BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True, pin_memory=True)
    test_loader = DataLoader(test_dataset, BATCH_SIZE, shuffle=False, num_workers=8, drop_last=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    f1_score_fn = F1Score(task="multiclass", num_classes=NUM_CLASSES).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_0, weight_decay=0.05)

    warmup_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.1)
    regular_iters = int(len(train_loader) * EPOCHS / GRADIENT_ACCUMULATION_STEPS * 0.9)

    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, 0.1, 1, total_iters=warmup_iters)
    regular_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=regular_iters, eta_min=LR_N)
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [warmup_scheduler, regular_scheduler], [warmup_iters])

    step_losses = list()
    step_lrs = list()
    step_grad_norms = list()

    start = time.time()
    iterator = iter(train_loader)

    for _ in range(10):
        next(iterator)

    print(f'[TEST] Fetched 10 batches in {round(time.time() - start, 4)} seconds')

    for epoch in range(EPOCHS):
        model.train()

        train_loss = 0
        val_loss = 0
        val_f1 = 0

        epoch_loss_history = list()
        epoch_grad_norm_history = list()

        start = time.time()

        for step, (x, attention_masks, y) in enumerate(train_loader):
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            y_pred = model(x)

            loss = criterion(y_pred, y)
            # since we use gradient accumation which is additive, loss should be scaled by GRADIENT_ACUMULATION_STEPS
            (loss / GRADIENT_ACCUMULATION_STEPS).backward()

            total_norm = nn.utils.get_total_norm([p.grad for p in model.parameters() if p.grad is not None]).mean().item()

            epoch_loss_history.append(loss.item())
            epoch_grad_norm_history.append(total_norm)

            step_grad_norms.append(total_norm)
            step_lrs.append(lr_scheduler.get_last_lr()[0])
            step_losses.append(loss.item())

            train_loss += loss.item()
            if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            if step % 50 == 49:
                print(f'step: {step}, loss smoothed: {round(sum(epoch_loss_history[-100:]) / min(100, step + 1), 4)}, grad_norm smoothed: {round(sum(epoch_grad_norm_history[-100:]) / min(100, step + 1), 4)}, lr: {"{:0.2e}".format(lr_scheduler.get_last_lr()[0])}')

        model.eval()

        val_y_target = list()
        val_y_pred = list()
        val_y_adv_pred = list()

        for x, attention_masks, y in val_loader:
            x = x.to(DEVICE, non_blocking=True)
            attention_masks = attention_masks.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            with torch.no_grad():
                y_pred = model(x)

            loss = criterion(y_pred, y)

            val_loss += loss.item()
            val_y_pred.append(y_pred.argmax(-1))
            val_y_target.append(y)

        val_y_pred = torch.cat(val_y_pred)
        val_y_target = torch.cat(val_y_target)

        train_loss_history.append(train_loss / len(train_loader))
        val_loss_history.append(val_loss / len(val_loader))
        val_f1_history.append(f1_score_fn(val_y_pred, val_y_target).item())

        print(f'Epoch {epoch+1}/{EPOCHS}, epoch time: {round(time.time() - start, 2)}.', end='')
        print(f' Train loss: {round(train_loss_history[-1], 6)},', end='')
        print(f' val loss: {round(val_loss_history[-1], 6)}, val f1: {round(val_f1_history[-1], 6)}')

    smoothing_ksize = 100
    step_losses_smoothed = np.convolve(step_losses, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')
    step_grad_norms_smoothed = np.convolve(step_grad_norms, np.ones(smoothing_ksize) / smoothing_ksize, mode='same')

    fig, axes = plt.subplots(ncols=3, figsize=(12, 4))
    axes[0].plot(step_losses, label='loss')
    axes[0].plot(step_losses_smoothed, label='loss smoothed')
    axes[0].legend()
    axes[0].set_title("Train Loss / step")
    axes[1].plot(step_lrs)
    axes[1].set_title("LR / step")
    axes[2].plot(step_grad_norms, label='grad norms')
    axes[2].plot(step_grad_norms_smoothed, label='grad norms smoothed')
    axes[2].legend()
    axes[2].set_title("Grad Norm / step")

    fig.savefig(f'{EXP_NAME}_losses_n_grads.png')

    model.eval()
    test_y_target = list()
    test_y_pred = list()
    for x, attention_masks, y in test_loader:
        x = x.to(DEVICE)
        attention_masks = attention_masks.to(DEVICE)
        y = y.to(DEVICE)

        with torch.no_grad():
            y_pred = model(x)

        test_y_pred.append(y_pred)
        test_y_target.append(y)

    test_y_pred = torch.cat(test_y_pred, dim=0)
    test_y_target = torch.cat(test_y_target)

    print(f'Test loss: {round(criterion(test_y_pred, test_y_target).item(), 6)}')
    print(f'Test f1: {round(f1_score_fn(test_y_pred.argmax(-1), test_y_target).item(), 6)}')
    print(f'Test classification report:\n{classification_report(test_y_target.cpu().numpy(), test_y_pred.argmax(-1).cpu().numpy())}')

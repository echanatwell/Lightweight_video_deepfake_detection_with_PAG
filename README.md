# Inference Guide

## Requirements

Python 3.10+ is required. Install dependencies:

```bash
pip install -r linux_requirements.txt
```

Key packages used at inference time:

| Package | Purpose |
|---|---|
| `torch`, `torchvision` | Model and transforms |
| `transformers` | Qwen3.5 vision encoder blocks |
| `av` (PyAV) | Video decoding |
| `opencv-python` | Frame reading (test.ipynb path) |
| `scikit-learn` | `classification_report` |
| `torchmetrics` | F1 score |

---

## Checkpoints

| File | Description |
|---|---|
| `checkpoints/encoder.pth` | MAE pretraining checkpoint — required only for `test_mae.py` |
| `checkpoints/last.pth` | MAEClassifier fine-tuned weights — used by `infer.py` |

`encoder.pth` format (produced by `pretrain_mae.py`):
```python
{
    'encoder_state_dict': {...},
    'hparams': {
        'encoder_depth': int,
        'd_model': int,
        'num_heads': int,
        'patch_size': int,
        'max_frames': int,
        'img_size': int,
    },
    'epoch': int,
    'loss': float,
}
```

`last.pth` format: plain `state_dict` saved via `torch.save(model.state_dict(), ...)`.

---

## Dataset Format

Both scripts expect videos organised into two subdirectories:

```
dataset_path/
├── real/
│   ├── video_001.mp4
│   ├── video_002.mp4
│   └── ...
└── fake/
    ├── video_101.mp4
    ├── video_102.mp4
    └── ...
```

- Any video format supported by PyAV (`.mp4`, `.avi`, `.mkv`, …) is accepted.
- Labels are derived from the directory name: `real/ → 0`, `fake/ → 1`.
- `test_mae.py` additionally requires a CSV file with columns `obj_id` (video filename without extension) and `label`.

---

## Video-level metrics — `infer.py`

Each video is split into non-overlapping segments of `--frames_per_segment` frames.
The model predicts a class for every segment; the **final video-level prediction is the majority vote** across all segment predictions.

```bash
python infer.py \
    --dataset_path /path/to/dataset \
    --classifier_ckpt checkpoints/last.pth \
    --device cuda
```

All options:

| Argument | Default | Description |
|---|---|---|
| `--dataset_path` | *(required)* | Root dir with `real/` and `fake/` subdirs |
| `--classifier_ckpt` | `checkpoints/last.pth` | MAEClassifier weights |
| `--frames_per_segment` | `16` | Frames per segment fed to the model |
| `--batch_size` | `8` | DataLoader batch size |
| `--num_workers` | `4` | DataLoader worker processes |
| `--device` | auto | `cuda` / `cpu` |
| `--log_file` | `classification_report.log` | Output log file |
| `--encoder_depth` | `6` | Must match checkpoint architecture |
| `--d_model` | `256` | Must match checkpoint architecture |
| `--num_heads` | `8` | Must match checkpoint architecture |
| `--patch_size` | `16` | Must match checkpoint architecture |
| `--max_frames` | `16` | Must match checkpoint architecture |
| `--img_size` | `224` | Input spatial resolution |

The script logs to both stdout and `--log_file`.

---

## Segment-level metrics — `test_mae.py`

Metrics are computed per segment (not aggregated to video level).
Requires both the MAE pretraining checkpoint and the classifier checkpoint.

```bash
python test_mae.py \
    checkpoints/encoder.pth \
    checkpoints/last.pth \
    --frames-per-video 16 \
    --img-size 224 \
    --batch-size 8
```

Positional arguments:

| Argument | Description |
|---|---|
| `pretrained_encoder_checkpoint` | Path to `encoder.pth` |
| `classifier_checkpoint` | Path to `last.pth` |

Optional arguments:

| Argument | Default | Description |
|---|---|---|
| `--frames-per-video` | `16` | Frames per segment |
| `--img-size` | `224` | Input spatial resolution |
| `--batch-size` | `8` | DataLoader batch size |

> **Note:** `labels_path` and `videos_path` are currently hardcoded at the bottom of `test_mae.py`. Edit lines 184–185 before running:
> ```python
> labels_path = "/path/to/labels.csv"   # columns: obj_id, label
> videos_path = "/path/to/videos"       # directory with *.mp4 files
> ```


------------------------------------

# Research

# Идея

1. Взять сильную, но легкую архитектуру (сейчас - легкий энкодер на ооснове блоков Qwen3.5)
2. Взять мощные позиционные многомерные эмбеддинги
3. Научить модель хорошо обращать внимание именно на разницу между кадрами (я ожидаю, что там бывает шум)
4. Сделать модель интерпретируемой без накладных расходов с помощью perceptionally-aligned gradients. Я этот метод откатал для сильно упрощенного сценария, подробности [тут](https://github.com/K0mp0t/perception-aligned-gradients)

# Текущий результат

1. Модель написана, написаны классы датасетов
2. Написан обычный и adversarial train loop
3. Поставлено несколько важных экспериментов и есть несколько выводов

# Самые важные эксперименты и выводы

Здесь далеко не все. Довольно много времени убил на то, чтобы заставить свою модель обучаться, пофиксить в ней баги и т.д. Про это не пишу, только самое важное

## Эксперименты
1. Обучение своей модели с нуля -> 0.63 F1
2. Обучение MViT с нуля -> 0.63 F1
3. Дообучение предобученного MViT с заменой головы и заморозкой всего, кроме новой головы -> 0.69 F1 (logs/celebdf_mvit_f.log)
4. Дообучение предобученного MViT с заменой головы, без заморозки -> 0.95 F1
5. Дообучение предобученного MViT с заменой головы, без заморозки + PAG -> TBD

## Выводы
1. Своя модель учится нормально, просто задача слишком сложная (эксперименты 1 и 2)
2. В процессе дообучения MViT не только подстраивает голову под новую задачу, но и может формировать новые attention-паттерны (эксперименты 3 и 4)
3. Если дообучить свою модель не на object semantics (как в ImageNet), а на frequency-centered задаче, то можно добиться результатов лучше, чем у предобученного на ImageNet MViT

## План экспериментов
1. Предобучение с нуля кастомного энкодера на frequency-centered SSL задачах:
    1. **Frequency-aware MAE**
       - Input: masked RGB (mask ratio=0.75)
       - Target: per-patch normalized Highpass(RGB) = RGB - GaussianBlur(RGB) (kernel=5, σ=1.0)
       - Encoder: 4x CustomTransformerEncoderLayer (d_model=128), Decoder: 2x lightweight transformer (d_dec=128)
       - Pretraining: 200 эпох на CelebDF train split (SSL, без меток), LR 1.5e-4 → 1e-5
       - Finetuning: (5 эпох, LR 0.0003→0.00001)
    2. Frequency-domain MAE. Input: masked FFT(RGB) или DCT(RGB). Target: RGB + FFT(RGB) reconstruction или RGB + DCT(RGB) reconstruction
2. Если все мои предыдущие выводы верны, то предлагаемый эксперимент 1 выйдет удачным (F1 0.95+). Тогда можно будет сделать подобное для своей модели.

"""
Frequency-Aware MAE (Masked Autoencoder) for video deepfake detection pretraining.

Architecture:
  - Encoder: same as Model in model.py (patch_embedding + Qwen3_5LikeVisualPositionEncoding
             + 4x CustomTransformerEncoderLayer)
  - Decoder: lightweight transformer (2x TransformerDecoderLayer, d_dec=128)
  - Target:  per-patch normalized Highpass(RGB) = RGB - GaussianBlur(RGB)
             kernel=5x5, sigma=1.0
  - Loss:    MSE on masked patches only

After pretraining, use FrequencyAwareMAE.encode() to extract features for finetuning.
"""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .model import CustomTransformerEncoderLayer, Qwen3_5LikeVisualPositionEncoding
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig


# ---------------------------------------------------------------------------
# Gaussian blur kernel (no external deps)
# ---------------------------------------------------------------------------

def _gaussian_kernel_2d(kernel_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Returns a 2D Gaussian kernel of shape (kernel_size, kernel_size)."""
    coords = torch.arange(kernel_size, device=device, dtype=dtype) - kernel_size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel = g[:, None] * g[None, :]
    return kernel / kernel.sum()


def gaussian_blur(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """
    Apply 2D Gaussian blur to a video tensor.

    Args:
        x: (B, C, T, H, W) — video frames, any dtype
        kernel_size: odd integer
        sigma: Gaussian sigma

    Returns:
        blurred tensor of same shape
    """
    B, C, T, H, W = x.shape
    k = _gaussian_kernel_2d(kernel_size, sigma, x.device, x.dtype)
    # shape: (1, 1, kH, kW) -> (C, 1, kH, kW) for depthwise conv
    k = k.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)
    pad = kernel_size // 2

    # merge batch and time dims for 2D conv
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    blurred = F.conv2d(x_2d, k, padding=pad, groups=C)
    return blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def highpass_target(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Highpass = x - GaussianBlur(x). Same shape as x."""
    return x - gaussian_blur(x, kernel_size, sigma)


# ---------------------------------------------------------------------------
# Patch utilities  (mirror of Model.encode_visual_features in model.py)
# ---------------------------------------------------------------------------

def patchify(x: torch.Tensor, patch_size: int = 16) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Split video into non-overlapping spatial patches.

    Args:
        x: (B, C, T, H, W)
        patch_size: spatial patch size (same for H and W)

    Returns:
        patches: (B, N, patch_dim)  where N = T * (H/P) * (W/P), patch_dim = C*P*P
        grids:   (B, 3)  [T, H/P, W/P] per sample
    """
    B, C, T, H, W = x.shape
    P = patch_size
    assert H % P == 0 and W % P == 0, f"H={H} and W={W} must be divisible by patch_size={P}"

    # (B, C, T, H/P, P, W/P, P)
    x = x.reshape(B, C, T, H // P, P, W // P, P)
    # (B, T, H/P, W/P, C, P, P) -> (B, T, H/P, W/P, C*P*P)
    x = x.permute(0, 2, 3, 5, 1, 4, 6).flatten(4)
    # (B, N, patch_dim)
    patches = x.flatten(1, 3)

    grids = torch.tensor([[T, H // P, W // P]], device=x.device, dtype=torch.long).expand(B, -1)
    return patches, grids


# ---------------------------------------------------------------------------
# Simple transformer decoder layer (standard pre-norm)
# ---------------------------------------------------------------------------

class TransformerDecoderLayer(nn.Module):
    """Lightweight pre-norm transformer layer (self-attention only, no cross-attention)."""

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # self-attention
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        # FFN
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# FrequencyAwareMAE
# ---------------------------------------------------------------------------

class FrequencyAwareMAE(nn.Module):
    """
    Frequency-Aware Masked Autoencoder for video.

    Pretraining objective:
        Given a video x (B, C, T, H, W):
        1. Compute highpass target = normalize_per_patch(x - GaussianBlur(x))
        2. Patchify x -> (B, N, patch_dim)
        3. Randomly mask 75% of patches
        4. Encode visible patches with the encoder
        5. Decode all patches (visible + mask tokens) with the decoder
        6. Compute MSE loss on masked patches only

    Finetuning:
        Call encode(x) to get (hidden_states, grids) — same interface as
        Model.encode_visual_features() in model.py — then attach a
        classification head on top.
    """

    PATCH_DIM = 768  # C * patch_size * patch_size = 3 * 16 * 16

    def __init__(
        self,
        # encoder
        encoder_depth: int = 4,
        d_model: int = 128,
        num_heads: int = 8,
        # decoder
        decoder_depth: int = 2,
        d_dec: int = 128,
        decoder_num_heads: int = 4,
        # MAE
        mask_ratio: float = 0.75,
        patch_size: int = 16,
        max_frames: int = 16,
        img_size: int = 224,
        # highpass
        blur_kernel: int = 5,
        blur_sigma: float = 1.0,
        add_decoder_projection = False
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.d_model = d_model
        self.d_dec = d_dec

        feedforward_dim = d_model * 4

        # ---- Encoder (same architecture as Model in model.py) ----
        self.vision_config = Qwen3_5VisionConfig(
            depth=encoder_depth,
            hidden_size=d_model,
            intermediate_size=feedforward_dim,
            num_heads=num_heads,
            in_channels=3,
            patch_size=patch_size,
            spatial_merge_size=1,
            temporal_patch_size=1,
            out_hidden_size=d_model,
            num_position_embeddings=(img_size // patch_size) ** 2,
            _attn_implementation='sdpa',
        )

        self.patch_embedding = nn.Linear(self.PATCH_DIM, d_model)
        self.positional_encoding = Qwen3_5LikeVisualPositionEncoding(self.vision_config)
        self.encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(self.vision_config)
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ---- Decoder ----
        # Project encoder output to decoder dimension
        if add_decoder_projection:
            self.decoder_proj = nn.Linear(d_model, d_dec)
        else:
            self.decoder_proj = None
        # Learnable mask token (replaces masked positions in decoder input)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        nn.init.normal_(self.mask_token, std=0.02)

        # Decoder positional embedding (simple learned, for all N positions)
        num_patches = max_frames * (img_size // patch_size) ** 2
        self.decoder_pos_embed = nn.Embedding(num_patches, d_dec)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_dec, decoder_num_heads, d_dec * 4)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(d_dec)

        # Prediction head: reconstruct patch_dim values
        self.pred_head = nn.Linear(d_dec, self.PATCH_DIM)

        self._init_weights()

    def _init_weights(self):
        # Initialize linear layers and layer norms
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                nn.init.ones_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _random_mask(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.

        Args:
            x: (B, N, d_model) — embedded patches

        Returns:
            x_vis:        (B, N_vis, d_model) — visible patches
            mask:         (B, N) bool — True = masked
            ids_restore:  (B, N) long — indices to restore original order
        """
        B, N, D = x.shape
        N_mask = int(N * self.mask_ratio)
        N_vis = N - N_mask

        # Random shuffle
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)          # (B, N)
        ids_restore = ids_shuffle.argsort(dim=1)    # (B, N)

        # Keep first N_vis tokens (lowest noise = "visible")
        ids_keep = ids_shuffle[:, :N_vis]           # (B, N_vis)
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        # Boolean mask: True = masked
        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        return x_vis, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder
    # ------------------------------------------------------------------

    def _encode(
        self,
        patches: torch.Tensor,
        grids: torch.Tensor,
        mask: torch.Tensor | None = None,
        ids_restore: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Run encoder on (optionally masked) patches.

        Args:
            patches: (B, N, PATCH_DIM) raw patches
            grids:   (B, 3) [T, H/P, W/P]
            mask:    (B, N) bool or None — if provided, only visible tokens are encoded
            ids_restore: (B, N) — needed when mask is provided

        Returns:
            encoded: (B, N_vis, d_model) if mask provided, else (B, N, d_model)
        """
        B, N, _ = patches.shape

        # Embed patches
        x = self.patch_embedding(patches)  # (B, N, d_model)

        # Add positional encoding and get cu_seqlens / position_embeddings
        x, cu_seqlens, position_embeddings = self.positional_encoding(x, grids)

        if mask is not None:
            # Keep only visible tokens
            N_vis = (~mask).sum(dim=1)[0].item()
            vis_idx = (~mask).nonzero(as_tuple=False)  # (B*N_vis, 2)
            x = x[~mask].reshape(B, N_vis, self.d_model)

            # Recompute cu_seqlens for visible-only sequences
            vis_per_sample = (~mask).sum(dim=1)  # (B,)
            cu_seqlens_vis = torch.zeros(B + 1, dtype=grids.dtype, device=grids.device)
            cu_seqlens_vis[1:] = vis_per_sample.cumsum(0)

            # Slice position_embeddings to visible positions
            cos_full, sin_full = position_embeddings  # (B*N, head_dim)
            vis_flat = (~mask).reshape(-1)
            cos_vis = cos_full[vis_flat]
            sin_vis = sin_full[vis_flat]
            position_embeddings_vis = (cos_vis, sin_vis)

            for layer in self.encoder_layers:
                x = layer(x, cu_seqlens_vis, None, position_embeddings_vis)
        else:
            for layer in self.encoder_layers:
                x = layer(x, cu_seqlens, None, position_embeddings)

        return self.encoder_norm(x)

    # ------------------------------------------------------------------
    # Public API: encode (for finetuning)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video without masking. Used during finetuning.

        Args:
            x: (B, C, T, H, W)

        Returns:
            hidden_states: (B, N, d_model)
            grids:         (B, 3)
        Same interface as Model.encode_visual_features() in model.py.
        """
        patches, grids = patchify(x, self.patch_size)
        hidden_states = self._encode(patches, grids, mask=None)
        return hidden_states, grids

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def _decode(
        self,
        encoded_vis: torch.Tensor,
        mask: torch.Tensor,
        ids_restore: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reconstruct all N patches from visible encoded tokens + mask tokens.

        Args:
            encoded_vis: (B, N_vis, d_model)
            mask:        (B, N) bool — True = masked
            ids_restore: (B, N) long

        Returns:
            pred: (B, N, PATCH_DIM)
        """
        B, N_vis, _ = encoded_vis.shape
        N = mask.shape[1]
        N_mask = N - N_vis

        # Project encoder output to decoder dim
        if self.decoder_proj is not None:
            vis_tokens = self.decoder_proj(encoded_vis)  # (B, N_vis, d_dec)
        else:
            vis_tokens = encoded_vis

        # Expand mask token
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # (B, N_mask, d_dec)

        # Concatenate: visible first, then mask tokens
        x = torch.cat([vis_tokens, mask_tokens], dim=1)  # (B, N, d_dec)

        # Restore original order using ids_restore
        x = torch.gather(x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.d_dec))

        # Add decoder positional embeddings
        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.decoder_pos_embed(pos_ids)

        # Decoder transformer layers
        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder_norm(x)

        # Predict patch values
        pred = self.pred_head(x)  # (B, N, PATCH_DIM)
        return pred

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-patch normalized highpass target.

        Steps:
          1. highpass = x - GaussianBlur(x)
          2. patchify -> (B, N, PATCH_DIM)
          3. normalize per patch: (t - mean) / (std + eps)

        Args:
            x: (B, C, T, H, W) in [-1, 1]

        Returns:
            target: (B, N, PATCH_DIM) normalized
        """
        hp = highpass_target(x, self.blur_kernel, self.blur_sigma)
        target, _ = patchify(hp, self.patch_size)
        # Per-patch normalization
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True)
        target = (target - mean) / (std + 1e-6)
        return target

    # ------------------------------------------------------------------
    # Forward (pretraining)
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pretraining forward pass.

        Args:
            x: (B, C, T, H, W) — video frames in [-1, 1]

        Returns:
            loss: scalar MSE loss on masked patches
        """
        # 1. Compute highpass target (no grad)
        target = self._compute_target(x)  # (B, N, PATCH_DIM)

        # 2. Patchify input
        patches, grids = patchify(x, self.patch_size)  # (B, N, PATCH_DIM)

        # 3. Embed patches
        x_emb = self.patch_embedding(patches)  # (B, N, d_model)

        # 4. Add positional encoding
        x_emb, cu_seqlens, position_embeddings = self.positional_encoding(x_emb, grids)

        # 5. Random masking
        x_vis, mask, ids_restore = self._random_mask(x_emb)  # (B, N_vis, d_model)

        # 6. Encode visible tokens
        B, N_vis, _ = x_vis.shape
        N = mask.shape[1]

        # Recompute cu_seqlens for visible-only sequences
        vis_per_sample = (~mask).sum(dim=1)  # (B,)
        cu_seqlens_vis = torch.zeros(B + 1, dtype=grids.dtype, device=grids.device)
        cu_seqlens_vis[1:] = vis_per_sample.cumsum(0)

        # Slice position_embeddings to visible positions
        cos_full, sin_full = position_embeddings  # (B*N, head_dim)
        vis_flat = (~mask).reshape(-1)
        cos_vis = cos_full[vis_flat]
        sin_vis = sin_full[vis_flat]
        position_embeddings_vis = (cos_vis, sin_vis)

        for layer in self.encoder_layers:
            x_vis = layer(x_vis, cu_seqlens_vis, None, position_embeddings_vis)
        x_vis = self.encoder_norm(x_vis)

        # 7. Decode
        pred = self._decode(x_vis, mask, ids_restore)  # (B, N, PATCH_DIM)

        # 8. MSE loss on masked patches only
        loss = F.mse_loss(pred[mask], target[mask])
        return loss

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Classifier built on top of the pretrained MAE encoder
# ---------------------------------------------------------------------------

class MAEClassifier(nn.Module):
    """
    Wraps the encoder part of FrequencyAwareMAE and adds:
      - sequence pooling (attention-weighted, same as Model in model.py)
      - classification head (Linear -> BN -> LeakyReLU -> Dropout -> Linear)

    Interface is compatible with the custom Model class in model.py:
        forward(x, attention_mask=None) -> logits (B, num_classes)
    """

    def __init__(
        self,
        mae: FrequencyAwareMAE,
        num_classes: int = 2,
        freeze_encoder: bool = False,
    ):
        super().__init__()

        d_model = mae.d_model

        # Encoder components (shared reference — no copy)
        self.patch_embedding = mae.patch_embedding
        self.positional_encoding = mae.positional_encoding
        self.encoder_layers = mae.encoder_layers
        self.encoder_norm = mae.encoder_norm

        # Sequence pooling weight (same design as Model.seq_pool_weight)
        self.seq_pool_weight = nn.Linear(d_model, 1)

        # Classification head (same design as Model.classification_head)
        self.classification_head = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes),
        )

        if freeze_encoder:
            for p in self.encoder_layers.parameters():
                p.requires_grad_(False)
            for p in self.patch_embedding.parameters():
                p.requires_grad_(False)
            for p in self.positional_encoding.parameters():
                p.requires_grad_(False)

    def sequence_pooling(self, seq: torch.Tensor, attention_mask=None) -> torch.Tensor:
        weights = self.seq_pool_weight(seq).permute(0, 2, 1)  # (B, 1, N)
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        weights = weights.softmax(dim=-1)
        return (weights @ seq).squeeze(1)  # (B, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T, H, W) video frames in [-1, 1]
            attention_mask: (B, T) optional — not used in current encoder,
                            kept for API compatibility with train.py / train_mvit.py

        Returns:
            logits: (B, num_classes)
        """

        patches, grids = patchify(x, patch_size=16)  # (B, N, 768)
        hidden = self.patch_embedding(patches)         # (B, N, d_model)

        hidden, cu_seqlens, position_embeddings = self.positional_encoding(hidden, grids)

        for layer in self.encoder_layers:
            hidden = layer(hidden, cu_seqlens, None, position_embeddings)
        hidden = self.encoder_norm(hidden)             # (B, N, d_model)

        pooled = self.sequence_pooling(hidden)         # (B, d_model)
        return self.classification_head(pooled)        # (B, num_classes)

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_mae_classifier(checkpoint_path: str, num_classes: int = 2) -> MAEClassifier:
    """
    Load a pretrained MAE encoder checkpoint and wrap it in MAEClassifier.

    The checkpoint is produced by pretrain_mae.py and contains:
        {
            'encoder_state_dict': {...},
            'hparams': {encoder_depth, d_model, num_heads, patch_size, ...},
            'epoch': int,
            'loss': float,
        }
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hparams']

    # Reconstruct the full MAE model with the same encoder architecture
    mae = FrequencyAwareMAE(
        encoder_depth=hparams['encoder_depth'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        patch_size=hparams.get('patch_size', 16),
        max_frames=hparams.get('max_frames', 16),
        img_size=hparams.get('img_size', 224),
    )

    # Load only encoder weights (decoder weights are discarded)
    missing, unexpected = mae.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    print(f'Loaded encoder from {checkpoint_path} (epoch {ckpt["epoch"]}, loss {ckpt["loss"]:.6f})')
    if missing:
        print(f'  Missing keys (decoder — expected): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')

    classifier = MAEClassifier(mae, num_classes=num_classes)
    return classifier

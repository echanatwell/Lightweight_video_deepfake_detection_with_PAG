"""
Frequency-Aware MAE with Sparse MoE Encoder for video deepfake detection pretraining.

Architecture:
  - Encoder: patch_embedding + Qwen3_5LikeVisualPositionEncoding
             + 4x MoETransformerEncoderLayer (Attention + SparseMoEFFN)
  - Decoder: lightweight dense transformer (2-4x TransformerDecoderLayer, d_dec=128)
  - Target:  per-patch normalized Highpass(RGB) = RGB - GaussianBlur(RGB)
             kernel=5x5, sigma=1.0
  - Loss:    MSE on masked patches + alpha * aux_load_balancing_loss

After pretraining, use FrequencyAwareMoEMAE.encode() to extract features for finetuning.
"""

import torch
from torch import nn
from torch.nn import functional as F

from .model import Qwen3_5LikeVisualPositionEncoding
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionAttention


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
    k = k.view(1, 1, kernel_size, kernel_size).expand(C, 1, kernel_size, kernel_size)
    pad = kernel_size // 2
    x_2d = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
    blurred = F.conv2d(x_2d, k, padding=pad, groups=C)
    return blurred.reshape(B, T, C, H, W).permute(0, 2, 1, 3, 4)


def highpass_target(x: torch.Tensor, kernel_size: int = 5, sigma: float = 1.0) -> torch.Tensor:
    """Highpass = x - GaussianBlur(x). Same shape as x."""
    return x - gaussian_blur(x, kernel_size, sigma)


# ---------------------------------------------------------------------------
# Patch utilities
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

    x = x.reshape(B, C, T, H // P, P, W // P, P)
    x = x.permute(0, 2, 3, 5, 1, 4, 6).flatten(4)
    patches = x.flatten(1, 3)

    grids = torch.tensor([[T, H // P, W // P]], device=x.device, dtype=torch.long).expand(B, -1)
    return patches, grids


# ---------------------------------------------------------------------------
# Sparse MoE FFN
# ---------------------------------------------------------------------------

class SparseMoEFFN(nn.Module):
    """
    Sparse Mixture-of-Experts FFN layer.

    Replaces the dense FFN in each transformer encoder layer.
    Each token is routed to top_k experts; outputs are weighted by router probabilities.
    Tokens exceeding per-expert capacity are dropped (passed through residual unchanged).

    Returns:
        output:   (B, N, d_model) — weighted sum of expert outputs
        aux_loss: scalar — Switch Transformer load-balancing loss
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.5,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # Router: small linear, no bias, initialised near zero for uniform start
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.normal_(self.router.weight, std=0.01)

        # Experts: each is an independent FFN (same structure as dense baseline)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_model * 4),
                nn.SiLU(),
                nn.Linear(d_model * 4, d_model),
            )
            for _ in range(num_experts)
        ])

        # Monitoring: fraction of tokens dropped due to capacity overflow
        self.last_overflow_frac: float = 0.0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: (B, N, d_model)

        Returns:
            output:   (B, N, d_model)
            aux_loss: scalar tensor
        """
        B, N, D = x.shape
        x_flat = x.reshape(-1, D)       # (T_flat, D) where T_flat = B * N
        T_flat = x_flat.shape[0]

        # ── 1. Router ──────────────────────────────────────────────────────────
        router_logits = self.router(x_flat)                     # (T_flat, E)

        # Noisy Top-K gating: add small noise during training to encourage
        # exploration and prevent early expert collapse
        if self.training:
            noise = torch.randn_like(router_logits) * 1e-2
            router_logits = router_logits + noise

        router_probs = F.softmax(router_logits, dim=-1)         # (T_flat, E)

        # Select top_k experts per token
        top_k_probs, top_k_indices = router_probs.topk(
            self.top_k, dim=-1
        )                                                        # (T_flat, K)

        # Re-normalise weights among selected experts so they sum to 1
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)

        # ── 2. Auxiliary load-balancing loss (Switch Transformer) ──────────────
        # f_i = fraction of tokens dispatched to expert i (over all top_k slots)
        # p_i = mean router probability for expert i across all tokens
        # L_aux = num_experts * sum(f_i * p_i)
        expert_counts = torch.zeros(
            self.num_experts, device=x.device, dtype=x.dtype
        )
        for k in range(self.top_k):
            expert_counts.scatter_add_(
                0,
                top_k_indices[:, k],
                torch.ones(T_flat, device=x.device, dtype=x.dtype),
            )

        # Normalise by total token-expert assignments
        f = expert_counts / T_flat                              # (E,)
        p = router_probs.mean(dim=0)                            # (E,) — NOT scalar
        aux_loss = self.num_experts * (f * p).sum()

        # ── 3. Expert capacity and token dropping ──────────────────────────────
        capacity = int(self.capacity_factor * T_flat * self.top_k / self.num_experts)
        capacity = max(capacity, 1)

        output = torch.zeros_like(x_flat)                       # (T_flat, D)
        total_dropped = 0

        # Track per-expert token counts across all top_k slots to enforce
        # the *total* capacity budget (not per-slot).
        expert_token_counts = [0] * self.num_experts

        for e_idx, expert in enumerate(self.experts):
            # Collect all token positions routed to this expert (any slot)
            all_positions_for_expert = []
            all_weights_for_expert = []

            for k in range(self.top_k):
                token_mask = (top_k_indices[:, k] == e_idx)    # (T_flat,) bool
                positions = token_mask.nonzero(as_tuple=True)[0]
                if positions.numel() == 0:
                    continue
                all_positions_for_expert.append((positions, k))

            if not all_positions_for_expert:
                continue

            # Concatenate positions from all slots, then apply capacity budget
            # to the combined list (total tokens this expert will process)
            combined_positions = torch.cat([p for p, _ in all_positions_for_expert])
            combined_slots     = torch.cat([
                torch.full((p.numel(),), k, device=x.device, dtype=torch.long)
                for p, k in all_positions_for_expert
            ])

            total_for_expert = combined_positions.numel()
            if total_for_expert > capacity:
                total_dropped += total_for_expert - capacity
                combined_positions = combined_positions[:capacity]
                combined_slots     = combined_slots[:capacity]

            expert_input  = x_flat[combined_positions]          # (n, D)
            expert_output = expert(expert_input)                 # (n, D)

            # Weight each token by its router probability for the slot it came from
            weights = top_k_probs[combined_positions, combined_slots].unsqueeze(-1)
            output[combined_positions] += expert_output * weights

        # Track overflow fraction for monitoring in training scripts
        # Denominator = total token-expert assignments attempted
        self.last_overflow_frac = total_dropped / max(T_flat * self.top_k, 1)

        return output.reshape(B, N, D), aux_loss


# ---------------------------------------------------------------------------
# MoE Transformer Encoder Layer
# ---------------------------------------------------------------------------

class MoETransformerEncoderLayer(nn.Module):
    """
    Transformer encoder layer with Sparse MoE FFN.

    Identical to CustomTransformerEncoderLayer in model/model.py except
    the dense FFN is replaced by SparseMoEFFN.

    forward() returns (hidden_states, aux_loss) instead of just hidden_states.
    """

    def __init__(
        self,
        config: Qwen3_5VisionConfig,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.5,
    ):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.attn = Qwen3_5VisionAttention(config)
        self.attn_dropout = nn.Dropout(0.1)

        # MoE replaces the dense FFN
        self.moe = SparseMoEFFN(
            d_model=config.hidden_size,
            num_experts=num_experts,
            top_k=top_k,
            capacity_factor=capacity_factor,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states:       (B, N, d_model)
            cu_seqlens:          cumulative sequence lengths for packed attention
            rotary_pos_emb:      unused (kept for API compatibility)
            position_embeddings: (cos, sin) rotary embeddings
            attention_mask:      optional padding mask

        Returns:
            hidden_states: (B, N, d_model)
            aux_loss:      scalar — MoE load-balancing loss for this layer
        """
        # ── Self-attention block ───────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attn(
            hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
            cu_seqlens=cu_seqlens,
            rotary_pos_emb=rotary_pos_emb,
            position_embeddings=position_embeddings,
        )
        hidden_states = residual + self.attn_dropout(
            hidden_states.reshape(*residual.shape)
        )

        # ── MoE FFN block ──────────────────────────────────────────────────────
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, aux_loss


# ---------------------------------------------------------------------------
# Decoder layer (dense — decoder stays simple)
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
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


# ---------------------------------------------------------------------------
# FrequencyAwareMoEMAE
# ---------------------------------------------------------------------------

class FrequencyAwareMoEMAE(nn.Module):
    """
    Frequency-Aware Masked Autoencoder with Sparse MoE Encoder.

    Pretraining objective:
        Given a video x (B, C, T, H, W):
        1. Compute highpass target = normalize_per_patch(x - GaussianBlur(x))
        2. Patchify x -> (B, N, patch_dim)
        3. Randomly mask mask_ratio of patches
        4. Encode visible patches with the MoE encoder
        5. Decode all patches (visible + mask tokens) with the dense decoder
        6. Compute MSE loss on masked patches + aux_loss_alpha * MoE load-balancing loss

    Finetuning:
        Call encode(x) to get (hidden_states, grids) — same interface as
        Model.encode_visual_features() in model/model.py — then attach a
        classification head on top.
    """

    PATCH_DIM = 768  # C * patch_size * patch_size = 3 * 16 * 16

    def __init__(
        self,
        # Encoder
        encoder_depth: int = 4,
        d_model: int = 128,
        num_heads: int = 8,
        # Decoder
        decoder_depth: int = 2,
        d_dec: int = 128,
        decoder_num_heads: int = 4,
        # MAE
        mask_ratio: float = 0.75,
        patch_size: int = 16,
        max_frames: int = 16,
        img_size: int = 224,
        # Highpass target
        blur_kernel: int = 5,
        blur_sigma: float = 1.0,
        # MoE
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.5,
        aux_loss_alpha: float = 0.01,
    ):
        super().__init__()

        self.mask_ratio = mask_ratio
        self.patch_size = patch_size
        self.max_frames = max_frames
        self.blur_kernel = blur_kernel
        self.blur_sigma = blur_sigma
        self.d_model = d_model
        self.d_dec = d_dec
        self.aux_loss_alpha = aux_loss_alpha

        feedforward_dim = d_model * 4

        # ── Encoder ───────────────────────────────────────────────────────────
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
            MoETransformerEncoderLayer(
                config=self.vision_config,
                num_experts=num_experts,
                top_k=top_k,
                capacity_factor=capacity_factor,
            )
            for _ in range(encoder_depth)
        ])
        self.encoder_norm = nn.LayerNorm(d_model)

        # ── Decoder ───────────────────────────────────────────────────────────
        self.decoder_proj = nn.Linear(d_model, d_dec)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, d_dec))
        nn.init.normal_(self.mask_token, std=0.02)

        num_patches = max_frames * (img_size // patch_size) ** 2
        self.decoder_pos_embed = nn.Embedding(num_patches, d_dec)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_dec, decoder_num_heads, d_dec * 4)
            for _ in range(decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(d_dec)

        self.pred_head = nn.Linear(d_dec, self.PATCH_DIM)

        self._init_weights()

    def _init_weights(self):
        # Collect router Linear objects to skip — they are already initialised
        # with std=0.01 in SparseMoEFFN.__init__ for uniform expert start.
        router_linears: set[int] = {
            id(layer.moe.router)
            for layer in self.encoder_layers
            if isinstance(layer, MoETransformerEncoderLayer)
        }

        for m in self.modules():
            if isinstance(m, nn.Linear):
                if id(m) in router_linears:
                    continue  # router already initialised — do not overwrite
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.LayerNorm, nn.RMSNorm)):
                nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    # Masking helpers
    # ------------------------------------------------------------------

    def _random_mask(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Randomly mask patches.

        Args:
            x: (B, N, d_model)

        Returns:
            x_vis:       (B, N_vis, d_model)
            mask:        (B, N) bool — True = masked
            ids_restore: (B, N) long — indices to restore original order
        """
        B, N, D = x.shape
        N_mask = int(N * self.mask_ratio)
        N_vis = N - N_mask

        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = noise.argsort(dim=1)
        ids_restore = ids_shuffle.argsort(dim=1)

        ids_keep = ids_shuffle[:, :N_vis]
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).expand(-1, -1, D))

        mask = torch.ones(B, N, dtype=torch.bool, device=x.device)
        mask.scatter_(1, ids_keep, False)

        return x_vis, mask, ids_restore

    # ------------------------------------------------------------------
    # Encoder (internal — used by both forward() and encode())
    # ------------------------------------------------------------------

    def _run_encoder(
        self,
        x: torch.Tensor,
        cu_seqlens: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run all encoder layers and accumulate aux_loss.

        Args:
            x:                   (B, N, d_model) — already embedded + pos-encoded
            cu_seqlens:          cumulative sequence lengths
            position_embeddings: (cos, sin)

        Returns:
            x:        (B, N, d_model) after encoder_norm
            aux_loss: scalar — sum of per-layer aux losses
        """
        total_aux_loss = x.new_zeros(())

        for layer in self.encoder_layers:
            x, layer_aux = layer(x, cu_seqlens, None, position_embeddings)
            total_aux_loss = total_aux_loss + layer_aux

        return self.encoder_norm(x), total_aux_loss

    # ------------------------------------------------------------------
    # Public API: encode (for finetuning — no aux_loss needed)
    # ------------------------------------------------------------------

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode video without masking. Used during finetuning.

        Args:
            x: (B, C, T, H, W)

        Returns:
            hidden_states: (B, N, d_model)
            grids:         (B, 3)
        Same interface as Model.encode_visual_features() in model/model.py.
        """
        patches, grids = patchify(x, self.patch_size)
        emb = self.patch_embedding(patches)
        emb, cu_seqlens, position_embeddings = self.positional_encoding(emb, grids)
        hidden_states, _aux = self._run_encoder(emb, cu_seqlens, position_embeddings)
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

        vis_tokens = self.decoder_proj(encoded_vis)                     # (B, N_vis, d_dec)
        mask_tokens = self.mask_token.expand(B, N_mask, -1)             # (B, N_mask, d_dec)

        x = torch.cat([vis_tokens, mask_tokens], dim=1)                 # (B, N, d_dec)
        x = torch.gather(
            x, 1, ids_restore.unsqueeze(-1).expand(-1, -1, self.d_dec)
        )

        pos_ids = torch.arange(N, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.decoder_pos_embed(pos_ids)

        for layer in self.decoder_layers:
            x = layer(x)
        x = self.decoder_norm(x)

        return self.pred_head(x)                                        # (B, N, PATCH_DIM)

    # ------------------------------------------------------------------
    # Target computation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _compute_target(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute per-patch normalized highpass target.

        Args:
            x: (B, C, T, H, W) in [-1, 1]

        Returns:
            target: (B, N, PATCH_DIM) normalized per patch
        """
        hp = highpass_target(x, self.blur_kernel, self.blur_sigma)
        target, _ = patchify(hp, self.patch_size)
        mean = target.mean(dim=-1, keepdim=True)
        std = target.std(dim=-1, keepdim=True)
        return (target - mean) / (std + 1e-6)

    # ------------------------------------------------------------------
    # Forward (pretraining)
    # ------------------------------------------------------------------

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Pretraining forward pass.

        Args:
            x: (B, C, T, H, W) — video frames in [-1, 1]

        Returns:
            total_loss: scalar — mse_loss + aux_loss_alpha * aux_loss
            mse_loss:   scalar — reconstruction loss on masked patches
            aux_loss:   scalar — MoE load-balancing loss (sum over encoder layers)
        """
        # 1. Highpass target (no grad)
        target = self._compute_target(x)                                # (B, N, PATCH_DIM)

        # 2. Patchify + embed
        patches, grids = patchify(x, self.patch_size)
        x_emb = self.patch_embedding(patches)                           # (B, N, d_model)

        # 3. Positional encoding
        x_emb, cu_seqlens, position_embeddings = self.positional_encoding(x_emb, grids)

        # 4. Random masking
        x_vis, mask, ids_restore = self._random_mask(x_emb)            # (B, N_vis, d_model)

        B, N_vis, _ = x_vis.shape
        N = mask.shape[1]

        # Recompute cu_seqlens for visible-only sequences
        vis_per_sample = (~mask).sum(dim=1)                             # (B,)
        cu_seqlens_vis = torch.zeros(B + 1, dtype=grids.dtype, device=grids.device)
        cu_seqlens_vis[1:] = vis_per_sample.cumsum(0)

        # Slice position_embeddings to visible positions
        cos_full, sin_full = position_embeddings
        vis_flat = (~mask).reshape(-1)
        position_embeddings_vis = (cos_full[vis_flat], sin_full[vis_flat])

        # 5. Encode visible tokens (accumulates aux_loss across all layers)
        x_vis, total_aux_loss = self._run_encoder(
            x_vis, cu_seqlens_vis, position_embeddings_vis
        )

        # 6. Decode
        pred = self._decode(x_vis, mask, ids_restore)                   # (B, N, PATCH_DIM)

        # 7. MSE loss on masked patches only
        mse_loss = F.mse_loss(pred[mask], target[mask])

        total_loss = mse_loss + self.aux_loss_alpha * total_aux_loss

        return total_loss, mse_loss, total_aux_loss

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# MAE Classifier (for finetuning)
# ---------------------------------------------------------------------------

class MoEMAEClassifier(nn.Module):
    """
    Wraps the encoder part of FrequencyAwareMoEMAE and adds:
      - sequence pooling (attention-weighted)
      - classification head (Linear -> BN -> LeakyReLU -> Dropout -> Linear)

    Interface is compatible with Model in model/model.py:
        forward(x, attention_mask=None) -> logits (B, num_classes)
    """

    def __init__(
        self,
        mae: FrequencyAwareMoEMAE,
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

        self.seq_pool_weight = nn.Linear(d_model, 1)

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

    def sequence_pooling(
        self, seq: torch.Tensor, attention_mask=None
    ) -> torch.Tensor:
        weights = self.seq_pool_weight(seq).permute(0, 2, 1)            # (B, 1, N)
        if attention_mask is not None:
            weights = weights.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)
        weights = weights.softmax(dim=-1)
        return (weights @ seq).squeeze(1)                               # (B, d_model)

    def forward(self, x: torch.Tensor, attention_mask=None) -> torch.Tensor:
        """
        Args:
            x:              (B, C, T, H, W) video frames in [-1, 1]
            attention_mask: (B, T) optional — kept for API compatibility

        Returns:
            logits: (B, num_classes)
        """
        patches, grids = patchify(x, patch_size=16)                     # (B, N, 768)
        hidden = self.patch_embedding(patches)                          # (B, N, d_model)
        hidden, cu_seqlens, position_embeddings = self.positional_encoding(hidden, grids)

        for layer in self.encoder_layers:
            hidden, _aux = layer(hidden, cu_seqlens, None, position_embeddings)
        hidden = self.encoder_norm(hidden)                              # (B, N, d_model)

        pooled = self.sequence_pooling(hidden)                          # (B, d_model)
        return self.classification_head(pooled)                         # (B, num_classes)

    @property
    def device(self):
        return next(self.parameters()).device


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------

def load_moe_mae_classifier(
    checkpoint_path: str,
    num_classes: int = 2,
    freeze_encoder: bool = False,
) -> MoEMAEClassifier:
    """
    Load a pretrained MoE MAE encoder checkpoint and wrap it in MoEMAEClassifier.

    The checkpoint is produced by pretrain_moe.py and contains:
        {
            'encoder_state_dict': {...},
            'hparams': {
                encoder_depth, d_model, num_heads, patch_size, max_frames, img_size,
                num_experts, top_k, capacity_factor,   # MoE-specific
            },
            'epoch': int,
            'loss': float,
        }

    Args:
        checkpoint_path: path to encoder checkpoint (.pth)
        num_classes:     number of output classes for the classification head
        freeze_encoder:  if True, encoder weights are frozen during finetuning

    Returns:
        MoEMAEClassifier ready for finetuning
    """
    ckpt = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    hparams = ckpt['hparams']

    mae = FrequencyAwareMoEMAE(
        encoder_depth=hparams['encoder_depth'],
        d_model=hparams['d_model'],
        num_heads=hparams['num_heads'],
        patch_size=hparams.get('patch_size', 16),
        max_frames=hparams.get('max_frames', 16),
        img_size=hparams.get('img_size', 224),
        # MoE params — fall back to defaults if checkpoint is older
        num_experts=hparams.get('num_experts', 8),
        top_k=hparams.get('top_k', 2),
        capacity_factor=hparams.get('capacity_factor', 1.5),
    )

    missing, unexpected = mae.load_state_dict(ckpt['encoder_state_dict'], strict=False)
    print(
        f'Loaded MoE encoder from {checkpoint_path} '
        f'(epoch {ckpt["epoch"]}, loss {ckpt["loss"]:.6f})'
    )
    if missing:
        print(f'  Missing keys (decoder — expected): {len(missing)}')
    if unexpected:
        print(f'  Unexpected keys: {unexpected}')

    return MoEMAEClassifier(mae, num_classes=num_classes, freeze_encoder=freeze_encoder)

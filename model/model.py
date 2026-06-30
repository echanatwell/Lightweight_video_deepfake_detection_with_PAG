import itertools

import numpy as np
from torchinfo import summary
import torch
from torch import nn
from torch.nn import functional as F
from transformers import Qwen3_5TextConfig
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_next.modeling_qwen3_next import apply_rotary_pos_emb
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5TextRotaryEmbedding, Qwen3_5VisionRotaryEmbedding, Qwen3_5VisionAttention


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb_vision(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()
    # cos, sin = cos.unsqueeze(-2).float(), sin.unsqueeze(-2).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)
    return q_embed, k_embed


class ResCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.main_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 2, 3, padding=1),
            nn.BatchNorm2d(out_channels // 2),
            nn.LeakyReLU(0.01),
            nn.Conv2d(out_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.res_branch = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1), 
            nn.BatchNorm2d(out_channels)
        )

        self.pooling = nn.MaxPool2d(2, 2)

    def forward(self, x):
        y1 = self.main_branch(x)
        y2 = self.res_branch(x)

        y = F.leaky_relu(y1+y2, 0.01)
        y = self.pooling(y)

        return y
    

class CustomTransformerEncoderLayer(nn.Module):
    def __init__(self, config: Qwen3_5VisionConfig):
        super().__init__()

        self.input_layernorm = nn.RMSNorm(config.hidden_size)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size)
        self.attn = Qwen3_5VisionAttention(config)
        self.attn_dropout = nn.Dropout(0.1)
        
        self.ffn = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(config.intermediate_size, config.hidden_size)
        )

    def forward(self, 
                hidden_states: torch.Tensor,
                cu_seqlens: torch.Tensor,
                rotary_pos_emb: torch.Tensor | None,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None = None):
        
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)

        hidden_states = self.attn(
                            hidden_states=hidden_states.reshape(-1, hidden_states.shape[-1]),
                            cu_seqlens=cu_seqlens,
                            rotary_pos_emb=rotary_pos_emb,
                            position_embeddings=position_embeddings,
                        )

        hidden_states = residual + hidden_states.reshape(*residual.shape)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.ffn(hidden_states)

        hidden_states = residual + hidden_states

        return hidden_states


class MRoPEInterleaveLikePositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        
        self.theta = 10000
        self.mrope_section = [12, 10, 10]
        self.rope_dim = config.hidden_size // config.num_heads

        inv_freq = 1.0 / (self.theta ** (torch.arange(0, self.rope_dim, 2) / self.rope_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def build_mrope_position_ids(
        self,
        grid_thw,            # (T, H, W) OR (B, 3)
        attention_mask=None, # (N,) optional
    ):
        """
        returns: (3, N)
        """

        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        device = grid_thw.device

        all_t, all_h, all_w = [], [], []

        offset = 0

        for b in range(grid_thw.shape[0]):
            t, h, w = grid_thw[b].tolist()
            h //= self.config.spatial_merge_size
            w //= self.config.spatial_merge_size

            N = t * h * w

            t_idx = torch.arange(t, device=device).repeat_interleave(h * w)
            h_idx = torch.arange(h, device=device).repeat(t).repeat_interleave(w)
            w_idx = torch.arange(w, device=device).repeat(t * h)

            all_t.append(t_idx)
            all_h.append(h_idx)
            all_w.append(w_idx)

            offset += N

        t = torch.cat(all_t)
        h = torch.cat(all_h)
        w = torch.cat(all_w)

        pos = torch.stack([t, h, w], dim=0)  # (3, N)

        # optional masking (IMPORTANT)
        if attention_mask is not None:
            pos = pos[:, attention_mask.bool()]

        return pos


    def build_cu_seqlens(self, grid_thw):
        lens = []

        for t, h, w in grid_thw.tolist():
            h //= self.config.spatial_merge_size
            w //= self.config.spatial_merge_size
            lens.append(t * h * w)

        cu = torch.tensor([0] + lens, dtype=torch.long).cumsum(0)
        return cu


    def apply_mrope_rope(self, inv_freq, position_ids):
        """
        position_ids: (3, N)
        returns freqs: (3, N, D/2)
        """

        pos = position_ids[:, :, None].float()   # (3, N, 1)
        inv = inv_freq[None, None, :]      # (1, 1, D/2, 1)

        freqs = (pos * inv)      # (3, N, D/2)

        return freqs


    def mrope_interleave(self, freqs):
        """
        freqs: (3, N, D/2)
        """

        f = freqs[0].clone()  # T base

        for dim, offset in enumerate((1, 2), start=1):
            length = self.mrope_section[dim] * 3
            idx = slice(offset, length, 3)
            f[..., idx] = freqs[dim, ..., idx]

        return f


    def get_cos_sin(self, freqs, scale=1.0):
        emb = torch.cat([freqs, freqs], dim=-1)

        cos = emb.cos() * scale
        sin = emb.sin() * scale

        return cos, sin


    def forward(self, grid_thw, mask=None):
        position_ids = self.build_mrope_position_ids(grid_thw, mask)
        cu_seqlens = self.build_cu_seqlens(grid_thw)
        freqs = self.apply_mrope_rope(self.inv_freq, position_ids)
        freqs = self.mrope_interleave(freqs)
        cos, sin = self.get_cos_sin(freqs)

        return (cos, sin), cu_seqlens


class Qwen3_5LikeVisualPositionEncoding(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        head_dim = config.hidden_size // config.num_heads

        self.spatial_merge_size = config.spatial_merge_size
        self.patch_size = config.patch_size
        self.spatial_merge_unit = self.spatial_merge_size * self.spatial_merge_size

        self.pos_embed = nn.Embedding(config.num_position_embeddings, config.hidden_size)
        self.num_grid_per_side = int(config.num_position_embeddings**0.5)

        self.rotary_pos_emb = Qwen3_5VisionRotaryEmbedding(head_dim // 2)

    
    def rot_pos_emb(self, grid_thw: torch.Tensor) -> torch.Tensor:
        merge_size = self.spatial_merge_size
        grid_thw_list = grid_thw.tolist()

        max_hw = max(max(h, w) for _, h, w in grid_thw_list)
        freq_table = self.rotary_pos_emb(max_hw)  # (max_hw, dim // 2)
        device = freq_table.device

        total_tokens = sum(t * h * w for t, h, w in grid_thw_list)
        pos_ids = torch.empty((total_tokens, 2), dtype=torch.long, device=device)

        offset = 0
        for num_frames, height, width in grid_thw_list:
            merged_h, merged_w = height // merge_size, width // merge_size

            block_rows = torch.arange(merged_h, device=device)  # block row indices
            block_cols = torch.arange(merged_w, device=device)  # block col indices
            intra_row = torch.arange(merge_size, device=device)  # intra-block row offsets
            intra_col = torch.arange(merge_size, device=device)  # intra-block col offsets

            # Compute full-resolution positions
            row_idx = block_rows[:, None, None, None] * merge_size + intra_row[None, None, :, None]
            col_idx = block_cols[None, :, None, None] * merge_size + intra_col[None, None, None, :]

            row_idx = row_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)
            col_idx = col_idx.expand(merged_h, merged_w, merge_size, merge_size).reshape(-1)

            coords = torch.stack((row_idx, col_idx), dim=-1)

            if num_frames > 1:
                coords = coords.repeat(num_frames, 1)

            num_tokens = coords.shape[0]
            pos_ids[offset : offset + num_tokens] = coords
            offset += num_tokens

        embeddings = freq_table[pos_ids]  # lookup rotary embeddings
        embeddings = embeddings.flatten(1)
        return embeddings


    def fast_pos_embed_interpolate(self, grid_thw):
        grid_thw_list = grid_thw.tolist()
        grid_ts = [row[0] for row in grid_thw_list]
        grid_hs = [row[1] for row in grid_thw_list]
        grid_ws = [row[2] for row in grid_thw_list]
        device = self.pos_embed.weight.device

        idx_list = [[] for _ in range(4)]
        weight_list = [[] for _ in range(4)]

        for t, h, w in grid_thw_list:
            h_idxs = torch.linspace(0, self.num_grid_per_side - 1, h, device=device)
            w_idxs = torch.linspace(0, self.num_grid_per_side - 1, w, device=device)

            h_idxs_floor = h_idxs.int()
            w_idxs_floor = w_idxs.int()
            h_idxs_ceil = (h_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)
            w_idxs_ceil = (w_idxs.int() + 1).clip(max=self.num_grid_per_side - 1)

            dh = h_idxs - h_idxs_floor
            dw = w_idxs - w_idxs_floor

            base_h = h_idxs_floor * self.num_grid_per_side
            base_h_ceil = h_idxs_ceil * self.num_grid_per_side

            indices = [
                (base_h[None].T + w_idxs_floor[None]).flatten(),
                (base_h[None].T + w_idxs_ceil[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_floor[None]).flatten(),
                (base_h_ceil[None].T + w_idxs_ceil[None]).flatten(),
            ]

            weights = [
                ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
                ((1 - dh)[None].T * dw[None]).flatten(),
                (dh[None].T * (1 - dw)[None]).flatten(),
                (dh[None].T * dw[None]).flatten(),
            ]

            for i in range(4):
                idx_list[i].extend(indices[i].tolist())
                weight_list[i].extend(weights[i].tolist())

        idx_tensor = torch.tensor(idx_list, dtype=torch.long, device=device)
        weight_tensor = torch.tensor(weight_list, dtype=self.pos_embed.weight.dtype, device=device)
        pos_embeds = self.pos_embed(idx_tensor).to(device) * weight_tensor[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]

        patch_pos_embeds = patch_pos_embeds.split([h * w for h, w in zip(grid_hs, grid_ws)])

        patch_pos_embeds_permute = []
        merge_size = self.config.spatial_merge_size
        for pos_embed, t, h, w in zip(patch_pos_embeds, grid_ts, grid_hs, grid_ws):
            pos_embed = pos_embed.repeat(t, 1)
            pos_embed = (
                pos_embed.view(t, h // merge_size, merge_size, w // merge_size, merge_size, -1)
                .permute(0, 1, 3, 2, 4, 5)
                .flatten(0, 4)
            )
            patch_pos_embeds_permute.append(pos_embed)
        patch_pos_embeds = torch.cat(patch_pos_embeds_permute)
        return patch_pos_embeds
        
    def forward(self, hidden_states, grid_thw):
        pos_embeds = self.fast_pos_embed_interpolate(grid_thw)
        hidden_states = hidden_states + pos_embeds.view(hidden_states.shape[0], -1, hidden_states.shape[-1])

        rotary_pos_emb = self.rot_pos_emb(grid_thw)

        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1) #.reshape(hidden_states.shape[0], hidden_states.shape[1], -1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = (grid_thw[:, 1] * grid_thw[:, 2] * grid_thw[:, 0]).cumsum(
            dim=0,
            # Select dtype based on the following factors:
            #  - FA2 requires that cu_seqlens_q must have dtype int32
            #  - torch.onnx.export requires that cu_seqlens_q must have same dtype as grid_thw
            # See https://github.com/huggingface/transformers/pull/34852 for more information
            dtype=grid_thw.dtype
        )
        cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

        return hidden_states, cu_seqlens, position_embeddings



class Qwen3_5LikeTextPositionalEncoding(nn.Module):
    def __init__(self, device='cpu', dtype=torch.float32):
        super().__init__()

        self.cfg = Qwen3_5TextConfig()
        self.cfg.standardize_rope_params()

        self.pos_enc = Qwen3_5TextRotaryEmbedding(self.cfg)

        self.register_buffer('dummy_buffer', torch.empty(0, device=device, dtype=dtype))


    @property
    def device(self):
        return self.dummy_buffer.device

    @property
    def dtype(self):
        return self.dummy_buffer.dtype


    def forward(self, input_shape, video_grid_thw, attention_mask):
        pos_idx, _ = self.get_rope_index(
            torch.zeros(input_shape, dtype=torch.long, device=self.device), 
            torch.full(input_shape, fill_value=2, dtype=torch.int, device=self.device),
            None, 
            video_grid_thw,
            attention_mask
        )

        dummy_x = torch.zeros((1,), device=self.device, dtype=self.dtype)

        cos, sin = self.pos_enc(dummy_x, pos_idx)

        return cos, sin


    def get_vision_position_ids(
        self,
        start_position: int,
        grid_thw: torch.Tensor,
        temp_merge_size: int = 1,
        spatial_merge_size: int = 1,
        time_interval: int = 1,
        device: str | torch.device | None = None,
    ):
        llm_grid_t, llm_grid_h, llm_grid_w = (
            int(grid_thw[0].item() // temp_merge_size),
            int(grid_thw[1].item() // spatial_merge_size),
            int(grid_thw[2].item() // spatial_merge_size),
        )

        image_seq_length = int(llm_grid_h * llm_grid_w * llm_grid_t)
        position_width = torch.arange(start_position, start_position + llm_grid_w, device=device).repeat(
            llm_grid_h * llm_grid_t
        )
        position_height = torch.arange(start_position, start_position + llm_grid_h, device=device).repeat_interleave(
            llm_grid_w * llm_grid_t
        )
        position_temporal = torch.full((image_seq_length,), start_position, device=device, dtype=torch.long)
        position_temporal = position_temporal * time_interval
        vision_position_ids = torch.stack([position_temporal, position_height, position_width], dim=0)

        return vision_position_ids


    
    def get_rope_index(
        self,
        input_ids: torch.LongTensor,
        mm_token_type_ids: torch.IntTensor,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:

        mrope_position_deltas = []
        position_ids = torch.zeros(
            3,
            input_ids.shape[0],
            input_ids.shape[1],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        grid_iters = {
            1: iter(image_grid_thw) if image_grid_thw is not None else None,
            2: iter(video_grid_thw) if video_grid_thw is not None else None,
        }

        for batch_idx, current_input_ids in enumerate(input_ids):
            input_token_type = mm_token_type_ids[batch_idx]
            if attention_mask is not None:
                current_input_ids = current_input_ids[attention_mask[batch_idx].bool()]
                input_token_type = input_token_type[attention_mask[batch_idx].bool()]

            input_type_group = []
            for key, group in itertools.groupby(enumerate(input_token_type.tolist()), lambda x: x[1]):
                group = list(group)
                start_index = group[0][0]
                end_index = group[-1][0] + 1
                input_type_group.append((key, start_index, end_index))

            current_pos = 0
            llm_pos_ids_list = []
            for modality_type, start_idx, end_idx in input_type_group:
                # text == 0
                if modality_type == 0:
                    text_len = end_idx - start_idx
                    llm_pos_ids_list.append(
                        torch.arange(text_len, device=input_ids.device).view(1, -1).expand(3, -1) + current_pos
                    )
                    current_pos += text_len
                # image == 1, video == 2
                elif grid_iters[modality_type] is None:
                    raise ValueError(f'modality of type {modality_type} is not initialized')
                else:
                    grid_thw = next(grid_iters[modality_type])
                    vision_position_ids = self.get_vision_position_ids(
                        current_pos, grid_thw, 1, 1, device=input_ids.device
                    )
                    llm_pos_ids_list.append(vision_position_ids)
                    current_pos += max(grid_thw[1], grid_thw[2])
            llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
            if attention_mask is not None:
                position_ids[:, batch_idx, attention_mask[batch_idx].bool()] = llm_positions.to(position_ids.device)
            else:
                position_ids[:, batch_idx] = llm_positions.to(position_ids.device)
            mrope_position_deltas.append(llm_positions.max() + 1 - len(current_input_ids))
        mrope_position_deltas = torch.tensor(mrope_position_deltas, device=input_ids.device).unsqueeze(1)
        return position_ids, mrope_position_deltas


class Model(nn.Module):
    def __init__(self, blocks=4, d_model=128, num_heads=8, max_frames=16, num_classes=3, img_size=224):
        super().__init__()

        self.blocks = blocks
        self.d_model = d_model
        self.num_heads = num_heads
        self.feedforward_dim = d_model * 4
        self.in_channels = 3

        self.max_frames = max_frames

        # self.backbone = nn.Sequential(
        #     ResCNNBlock(self.in_channels, 32),  # 224 -> 112
        #     ResCNNBlock(32, 64),  # 112 -> 56
        #     ResCNNBlock(64, 128),  # 56 -> 28
        #     ResCNNBlock(128, 256),  # 28 -> 14
        #     ResCNNBlock(256, self.d_model)  # 14 -> 7
        # )

        self.vision_config = Qwen3_5VisionConfig(
            depth=self.blocks,
            hidden_size=self.d_model,
            intermediate_size=self.feedforward_dim,
            num_heads=self.num_heads,
            in_channels=self.in_channels,
            patch_size=16,
            spatial_merge_size=1, 
            temporal_patch_size=1,
            out_hidden_size=self.d_model,
            # num_position_embeddings=(img_size // (2 ** len(self.backbone))) ** 2,
            num_position_embeddings = (img_size // 16) ** 2,
            _attn_implementation='sdpa'
        )
        self.positional_encoding = Qwen3_5LikeVisualPositionEncoding(self.vision_config)

        self.patch_embedding = nn.Linear(768, self.d_model)

        self.transformer_encoder_layers = nn.ModuleList([
            CustomTransformerEncoderLayer(self.vision_config) for _ in range(self.blocks)
        ])

        self.seq_pool_weight = nn.Linear(self.d_model, 1)
        
        self.classification_head = nn.Sequential(
            nn.Linear(self.d_model, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(p=0.1),
            nn.Linear(64, num_classes)
        )

    @property
    def device(self):
        return next(self.parameters()).device

    def sequence_pooling(self, seq, attention_mask=None):
        seq_element_weights = self.seq_pool_weight(seq).permute(0, 2, 1)

        if attention_mask is not None:
            seq_element_weights = seq_element_weights.masked_fill(attention_mask.unsqueeze(1) == 0, -1e9)

        seq_element_weights = seq_element_weights.softmax(dim=-1)
        
        return (seq_element_weights @ seq).squeeze(1)
    

    def encode_visual_features(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # visual_features_padded = self.backbone(x.view(-1, *x.shape[2:]))
        # X: B, F, C, H, W => B, F, C, h, p, w, p
        x = x.reshape(*x.shape[:3], x.shape[4] // 16, 16, x.shape[4] // 16, 16)
        # X: B, F, C, h, p, w, p => B, F, h, w, C, p, p => B, F, H, W, C*p*2
        x = x.permute(0, 1, 3, 5, 2, 4, 6).flatten(4)
        grids = torch.stack([torch.as_tensor(x_.shape[:3]) for x_ in x], dim=0)
        visual_features = self.patch_embedding(x.flatten(1, 3))

        # visual_features = visual_features.view(len(x), self.max_frames, *visual_features.shape[1:])
        # grids = torch.stack([torch.as_tensor(x_.shape[:3]) for i, x_ in enumerate(visual_features)], dim=0)
        # visual_features = visual_features.flatten(2, 3).permute(0, 1, 3, 2) # BFEhw -> BFEt -> BFtE

        #  BFtE -> BTE
        # visual_features = visual_features.flatten(1, 3)
        return visual_features, grids


    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None):
        visual_features, grids = self.encode_visual_features(x)
        t = int(visual_features.shape[1] / self.max_frames)
        assert visual_features.shape[1] % self.max_frames == 0
        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(-1).expand(-1, -1, t).flatten(1)
        hidden_states, cu_seqlens, position_embeddings = self.positional_encoding(visual_features, grids)

        for layer in self.transformer_encoder_layers:
            hidden_states = layer(hidden_states, cu_seqlens, None, position_embeddings, attention_mask)

        final_hidden_states = self.sequence_pooling(hidden_states, attention_mask)

        cls_logits = self.classification_head(final_hidden_states)

        return cls_logits
    
    def print_summary(self, batch_size=16, depth=3):
        summary(self, (batch_size, self.max_frames, 3, 224, 224), device=self.device, depth=depth)

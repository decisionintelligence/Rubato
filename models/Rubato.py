import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from utils.ExpConfigs import ExpConfigs



class SORF_RFSF_Features_TRP(nn.Module):
    """SORF-based Random Fourier Signature Features with Fused CUDA Kernel.

    Uses fast_hadamard_transform's fused SORF kernel: 3 rounds of diag×FWHT
    + cos/sin in a single kernel launch. O(D log d) instead of O(D·d).
    Falls back to dense matmul when CUDA is unavailable.
    """

    def __init__(self, M: int, in_dim: int, rff_dim: int,
                 final_feature_dim: int, sigma_init: float = 1.0,
                 sorf_seed: int = 42):
        super().__init__()
        self.M = M
        self.in_dim = in_dim
        self.rff_dim = rff_dim
        self.final_feature_dim = final_feature_dim
        # Pad to power-of-2, minimum 8 (CUDA kernel requires 8-alignment)
        self.padded_dim = max(2 ** math.ceil(math.log2(max(in_dim, 2))), 8)
        self.log_sigma = nn.Parameter(torch.log(torch.tensor(sigma_init)))
        self.num_sorf_groups = math.ceil(rff_dim / self.padded_dim)

        from fast_hadamard_transform import sorf_rff_transform  # noqa: F401

        sorf_rng = torch.Generator()
        sorf_rng.manual_seed(sorf_seed)

        n_rounds = 3
        for m in range(1, M + 1):
            group_diags = []
            for g in range(self.num_sorf_groups):
                round_signs = []
                for _ in range(n_rounds):
                    diag = (torch.randint(0, 2, (self.padded_dim,),
                                          generator=sorf_rng) * 2 - 1).float()
                    round_signs.append(diag)
                group_diags.append(torch.stack(round_signs, dim=0))

            self.register_buffer(f"sorf_diag_{m}",
                                 torch.stack(group_diags, dim=0))

            p = torch.randn(2 * rff_dim, final_feature_dim)
            self.register_buffer(f"P_{m}", p)

    def _sorf_project_and_rff(self, x: torch.Tensor, m: int) -> torch.Tensor:
        """Project x via SORF fused CUDA kernel and compute [cos, sin] RFF features.

        Uses the batched-diag fused kernel: a single CUDA launch over all
        ``num_sorf_groups`` SORF blocks. Sigma scaling is applied in PyTorch
        (not inside CUDA kernel) so autograd can compute gradients w.r.t.
        log_sigma. Returns: (B, L, 2 * rff_dim)
        """
        from fast_hadamard_transform import sorf_rff_transform_batched
        diag_all = getattr(self, f"sorf_diag_{m}")  # (G, 3, padded_dim)
        if x.shape[-1] < self.padded_dim:
            x = F.pad(x, (0, self.padded_dim - x.shape[-1]))
        x_scaled = x * torch.exp(-self.log_sigma)
        sorf_scale = (1.0 / math.sqrt(self.padded_dim)) ** 3
        rff_scale = 1.0 / math.sqrt(self.rff_dim)
        G = self.num_sorf_groups
        d_pad = self.padded_dim

        # Single launch over all groups. Output layout per batch element:
        # [cos_g0|sin_g0 | cos_g1|sin_g1 | ... | cos_g{G-1}|sin_g{G-1}]
        phi_all = sorf_rff_transform_batched(
            x_scaled, diag_all,
            input_scale=1.0,
            sorf_scale=sorf_scale,
            rff_scale=rff_scale,
            rff_dim=d_pad,
        )
        leading = list(x_scaled.shape[:-1])
        phi_per_group = phi_all.view(*leading, G, 2 * d_pad)
        cos_cat = phi_per_group[..., :d_pad].reshape(*leading, G * d_pad)[..., :self.rff_dim]
        sin_cat = phi_per_group[..., d_pad:].reshape(*leading, G * d_pad)[..., :self.rff_dim]
        return torch.cat([cos_cat, sin_cat], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, in_dim) path data (should be forward-filled before calling)
        """
        B, L, _ = x.shape
        U_per_level = []

        for m in range(1, self.M + 1):
            phi = self._sorf_project_and_rff(x, m)  # (B, L, 2*rff_dim)
            diffs = phi[:, 1:, :] - phi[:, :-1, :]
            u_m = torch.matmul(diffs, getattr(self, f"P_{m}"))
            U_per_level.append(u_m)

        inv_sqrt_d = 1.0 / math.sqrt(self.final_feature_dim)
        curr_level_seq = U_per_level[0]
        all_levels = [torch.sum(curr_level_seq, dim=1) * inv_sqrt_d]
        for m in range(1, self.M):
            integrated_prev = torch.cumsum(curr_level_seq, dim=1)
            shifted_prev = F.pad(integrated_prev[:, :-1, :], (0, 0, 1, 0), value=0.0)
            term = shifted_prev * U_per_level[m]
            curr_level_seq = term
            all_levels.append(torch.sum(term, dim=1) * inv_sqrt_d)

        features = torch.cat(all_levels, dim=-1)
        const = torch.ones(B, 1, device=x.device, dtype=x.dtype) * 1e-3
        return torch.cat([const, features], dim=-1)


# ─────────────────────────────────────────────────────────
# Stage 1: Per-Channel Tokenizer + Joint Signature Context
# ─────────────────────────────────────────────────────────

class PerChannelTokenizer(nn.Module):
    """Per-channel MLP tokenizer (fast).

    Each channel's (value, mask, time) patch is encoded via a lightweight MLP.
    No SORF here — cross-channel signatures are handled by JointSignatureExtractor.
    Returns per-channel content tokens.
    """

    def __init__(self, embed_dim: int, patch_len: int, dropout: float):
        super().__init__()
        self.patch_proj = nn.Sequential(
            nn.Linear(3 * patch_len, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.skip_proj = nn.Linear(3, embed_dim)
        self.token_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, obs_patches: torch.Tensor,
                patch_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs_patches: (B*C*N, P, 4) per-channel patches [value, mask, time, indicator]
            patch_mask:  (B*C*N, P, 1) observation mask per step
        Returns:
            content: (B*C*N, d_model)
        """
        m = patch_mask.squeeze(-1)
        obs_vti = torch.stack([
            obs_patches[..., 0] * m,  # value (gated by obs mask)
            obs_patches[..., 2],      # time (preserved regardless of obs)
            obs_patches[..., 3],      # target indicator (preserved regardless of obs)
        ], dim=-1)  # (BCN, P, 3)

        flat = obs_vti.reshape(obs_vti.shape[0], -1)  # (BCN, 3*P)
        content = self.patch_proj(flat)

        mask_sum = patch_mask.sum(dim=1).clamp(min=1.0)
        raw_mean = (obs_vti * patch_mask).sum(dim=1) / mask_sum
        content = content + self.skip_proj(raw_mean)

        content = self.token_norm(self.dropout(content))
        return content


class JointSignatureExtractor(nn.Module):
    """Joint R^{2C+1} signature for cross-channel context.

    Computes one signature per patch from the joint multivariate path.
    Used as attention bias (signature kernel) for cross-channel awareness.
    """

    def __init__(self, n_variables: int, M: int, rff_dim: int,
                 final_feature_dim_per_level: int, sigma_init: float,
                 sorf_seed: int = 42):
        super().__init__()
        self.n_variables = n_variables
        self.joint_dim = 2 * n_variables + 1

        self.sorf = SORF_RFSF_Features_TRP(
            M=M, in_dim=self.joint_dim, rff_dim=rff_dim,
            final_feature_dim=final_feature_dim_per_level,
            sigma_init=sigma_init,
            sorf_seed=sorf_seed,
        )

    def forward(self, x_patches: torch.Tensor, mask_patches: torch.Tensor,
                time_patches: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_patches:    (B, N, P, C)
            mask_patches: (B, N, P, C)
            time_patches: (B, N, P, 1)
        Returns:
            joint_sig: (B, N, sig_dim)
        """
        B, N, P, C = x_patches.shape

        joint_path = torch.cat([x_patches, mask_patches, time_patches], dim=-1)
        joint_flat = joint_path.reshape(B * N, P, -1)

        sig = self.sorf(joint_flat)
        return sig.reshape(B, N, -1)


# ─────────────────────────────────────────────────────────
# Stage 2: Signature Kernel Attention
# ─────────────────────────────────────────────────────────

class SignatureKernelAttention(nn.Module):
    """Multi-head adaptive signature kernel attention.

    Signature kernel K stays at (B, H, N, N) — shared across channels.
    Per-channel adaptivity via value gating: each channel gates what info
    it contributes, while the kernel determines which patches are related.

        K_h = cosine_norm( (W_h ψ)(W_h ψ)^T ) · τ_h
        attn_h = softmax( K_h )
        out = attn_h @ (v ⊙ σ(gate))   # per-channel value gating
    """

    def __init__(self, d_model: int, sig_dim: int,
                 n_heads: int = 4, dropout: float = 0.0,
                 n_variables: int = 1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.sig_proj = nn.Linear(sig_dim, n_heads * self.d_k)
        self.content_gate = nn.Linear(d_model, n_heads)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.temperature = nn.Parameter(torch.ones(n_heads))
        self.dp = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, joint_sig: torch.Tensor,
                patch_mask: torch.Tensor = None) -> torch.Tensor:
        BC, N, d = x.shape
        B = joint_sig.shape[0]
        C = BC // B
        H = self.n_heads

        # Per-head signature kernel: (B, H, N, N) — stays at batch level
        proj = self.sig_proj(joint_sig).reshape(B, N, H, self.d_k).permute(0, 2, 1, 3)
        K = torch.matmul(proj, proj.transpose(-2, -1))

        # Cosine normalization per head
        diag = torch.diagonal(K, dim1=-2, dim2=-1)
        norm = torch.sqrt(diag.unsqueeze(-1) * diag.unsqueeze(-2)).clamp(min=1e-8)
        K = K / norm * self.temperature.reshape(1, H, 1, 1)

        # Mask out empty patches at (B, H, N, N)
        if patch_mask is not None:
            K = K.masked_fill(~patch_mask.unsqueeze(1).unsqueeze(-2).bool(), -1e9)

        attn = self.dp(torch.softmax(K, dim=-1))  # (B, H, N, N)

        # Per-channel value gating
        s = torch.sigmoid(self.content_gate(x)).permute(0, 2, 1).unsqueeze(-1)  # (BC, H, N, 1)
        v = self.v_proj(x).reshape(BC, N, H, self.d_k).permute(0, 2, 1, 3)  # (BC, H, N, d_k)
        v = v * s

        # Broadcast matmul: (B, 1, H, N, N) @ (B, C, H, N, d_k) → (B, C, H, N, d_k)
        out = torch.matmul(
            attn.unsqueeze(1),
            v.reshape(B, C, H, N, self.d_k)
        ).reshape(BC, H, N, self.d_k)

        out = out.permute(0, 2, 1, 3).reshape(BC, N, d)
        return self.out_proj(out)


class RubatoBlock(nn.Module):
    def __init__(self, attn: SignatureKernelAttention, d_model: int, d_ff: int,
                 dropout: float, n_variables: int = 0):
        super().__init__()
        self.attn = attn
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_ff, d_model), nn.Dropout(dropout),
        )
        if n_variables > 1:
            self.channel_mix = nn.Linear(n_variables, n_variables, bias=False)
            self.norm_cm = nn.LayerNorm(d_model)
        else:
            self.channel_mix = None

    def forward(self, x: torch.Tensor, joint_sig: torch.Tensor,
                patch_mask: torch.Tensor = None,
                **kwargs) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), joint_sig, patch_mask)
        x = x + self.ffn(self.norm2(x))
        if self.channel_mix is not None:
            BC, N, D = x.shape
            B = joint_sig.shape[0]
            C = BC // B
            x_c = self.norm_cm(x).reshape(B, C, N, D).permute(0, 2, 3, 1)
            x = x + self.channel_mix(x_c).permute(0, 3, 1, 2).reshape(BC, N, D)
        return x


# ─────────────────────────────────────────────────────────
# Stage 3: Signature Kernel Decoder
# ─────────────────────────────────────────────────────────

class SignatureKernelDecoder(nn.Module):
    """Per-channel cross-attention decoder.

    Time-embedded + channel-conditioned queries attend to encoded history.
    Simple and fast — no horizon mixing or extra attention layers.
    """

    def __init__(self, d_model: int, te_dim: int, n_variables: int):
        super().__init__()
        self.d_model = d_model
        self.n_variables = n_variables

        self.te_scale = nn.Linear(1, 1)
        self.te_periodic = nn.Linear(1, te_dim - 1)
        self.channel_emb = nn.Embedding(n_variables, d_model)
        self.query_proj = nn.Linear(te_dim + d_model, d_model)
        self.query_norm = nn.LayerNorm(d_model)

        self.v_proj = nn.Linear(d_model, d_model)

        self.pred_head = nn.Linear(d_model, 1)

    def _time_embed(self, t):
        return torch.cat([self.te_scale(t), torch.sin(self.te_periodic(t))], dim=-1)

    def forward(self, h_history, y_mark, channel_emb_weight,
                patch_active=None):
        C = self.n_variables
        B = h_history.shape[0] // C
        pred_len = y_mark.shape[1]

        te_q = self._time_embed(y_mark)
        te_exp = te_q.unsqueeze(1).expand(-1, C, -1, -1)
        ch_exp = channel_emb_weight.unsqueeze(0).unsqueeze(2).expand(B, -1, pred_len, -1)
        q = self.query_norm(self.query_proj(torch.cat([te_exp, ch_exp], dim=-1)))
        q = q.reshape(B * C, pred_len, self.d_model)

        v = self.v_proj(h_history)
        scores = torch.bmm(q, v.transpose(-2, -1)) / math.sqrt(self.d_model)

        # Mask out empty patches in decoder cross-attention
        if patch_active is not None:
            pm = patch_active.unsqueeze(1).expand(-1, C, -1).reshape(B * C, 1, -1)
            scores = scores.masked_fill(~pm.bool(), -1e9)

        attn = torch.softmax(scores, dim=-1)
        output = torch.bmm(attn, v)

        pred = self.pred_head(output).squeeze(-1)

        return rearrange(pred, "(b c) l -> b l c", b=B, c=C)


# ─────────────────────────────────────────────────────────
# Main Model
# ─────────────────────────────────────────────────────────

class Model(nn.Module):
    """Rubato: Multi-Resolution Signature for Irregular MTS.

    Architecture:
    - Per-channel R³ SORF tokenizer: C×N tokens preserving per-variable dynamics
    - Joint R^{2C+1} SORF signature: cross-channel context (computed once per patch)
    - Signature kernel attention: per-channel self-attn + joint sig kernel bias
    - Per-channel decoder: signature kernel matching + channel-conditioned queries

    Per-channel tokens keep quality. Joint signature adds cross-channel awareness.
    Transformer processes C independent length-N sequences: O(C × N² × d).
    Joint signature kernel is O(N²) and shared across all channels.
    """

    def __init__(self, configs: ExpConfigs):
        super().__init__()
        self.task_name = configs.task_name
        self.c_out = configs.c_out
        C = configs.c_out

        d_model = configs.d_model
        d_ff = int(getattr(configs, "d_ff", 4 * d_model))
        dropout = configs.dropout
        e_layers = configs.e_layers

        M = int(getattr(configs, "rubato_M", 2))
        sigma_init = float(getattr(configs, "rubato_sigma_init", 0.5))
        sorf_seed = int(getattr(configs, "rubato_sorf_seed", 42))
        rff_dim_cfg = int(getattr(configs, "rubato_rff_dim", 0))
        final_feature_dim_cfg = int(
            getattr(configs, "rubato_final_feature_dim_per_level", 0))
        rff_dim = rff_dim_cfg if rff_dim_cfg > 0 else 2 * d_model
        final_feature_dim_per_level = (
            final_feature_dim_cfg if final_feature_dim_cfg > 0
            else (2 * d_model // max(M, 1)))

        base_seq_len = (configs.seq_len_max_irr
                        if configs.seq_len_max_irr is not None
                        else configs.seq_len)
        base_pred_len = (configs.pred_len_max_irr
                         if configs.pred_len_max_irr is not None
                         else configs.pred_len)
        base_total_len = base_seq_len + base_pred_len

        # Patching by fixed number of patches over joint (input + prediction) sequence.
        # patch_len derived so that num_patches · patch_len ≥ base_total_len.
        self.num_patches = int(configs.rubato_num_patches)
        assert self.num_patches > 0, "rubato_num_patches must be > 0"
        self.patch_len = max(1, math.ceil(base_total_len / self.num_patches))
        self.target_len = self.num_patches * self.patch_len
        self.num_patch_tokens = self.num_patches

        # ── Stage 1a: Per-Channel Tokenizer (MLP, fast) ──
        self.tokenizer = PerChannelTokenizer(
            embed_dim=d_model, patch_len=self.patch_len, dropout=dropout,
        )

        # ── Stage 1b: Joint Signature Context ──
        self.joint_sig = JointSignatureExtractor(
            n_variables=C, M=M, rff_dim=rff_dim,
            final_feature_dim_per_level=final_feature_dim_per_level,
            sigma_init=sigma_init,
            sorf_seed=sorf_seed,
        )

        self.channel_emb = nn.Embedding(C, d_model)
        self.patch_pos_emb = nn.Parameter(
            torch.zeros(1, self.num_patch_tokens, d_model))

        # ── Stage 2: Multi-Head Adaptive Signature Kernel Attention ──
        n_heads = int(getattr(configs, "n_heads", 4))
        sig_dim = M * final_feature_dim_per_level + 1
        self.layers = nn.ModuleList([
            RubatoBlock(
                attn=SignatureKernelAttention(
                    d_model=d_model, sig_dim=sig_dim,
                    n_heads=n_heads, dropout=dropout,
                    n_variables=C,
                ),
                d_model=d_model, d_ff=d_ff, dropout=dropout,
                n_variables=C,
            ) for _ in range(e_layers)
        ])
        self.final_norm = nn.LayerNorm(d_model)

        # ── Stage 3: Decoder ──
        te_dim = int(getattr(configs, "rubato_te_dim", 16))
        self.sig_decoder = SignatureKernelDecoder(
            d_model=d_model, te_dim=te_dim, n_variables=C,
        )


    def _repeat_tail(self, tensor, target_len):
        current_len = tensor.shape[1]
        if current_len == target_len:
            return tensor
        if current_len > target_len:
            return tensor[:, :target_len]
        pad = tensor[:, -1:].expand(-1, target_len - current_len, *tensor.shape[2:])
        return torch.cat([tensor, pad], dim=1)

    def _compact_dim1(self, tensor, sort_idx, Na):
        """Gather active patches along dim=1 using precomputed sort indices."""
        idx = sort_idx[:, :Na]
        for _ in range(tensor.dim() - 2):
            idx = idx.unsqueeze(-1)
        return torch.gather(tensor, 1, idx.expand(-1, -1, *tensor.shape[2:]))

    def _encode(self, x_enc, x_mark_enc, x_mask, indicator):
        """Returns (h, compact_mask): h is (B*C, Na, d_model), mask is (B, Na)."""
        B, L, C = x_enc.shape
        target_len = self.target_len
        if L != target_len:
            x_enc = self._repeat_tail(x_enc, target_len)
            x_mask = self._repeat_tail(x_mask, target_len)
            x_mark_enc = self._repeat_tail(x_mark_enc, target_len)
            indicator = self._repeat_tail(indicator, target_len)

        num_patches = self.num_patches
        pl = self.patch_len

        # ── Compute patch activity & compact ──
        patch_active = x_mask.reshape(B, num_patches, pl, C).sum(dim=(2, 3)) > 0  # (B, N)
        active_counts = patch_active.sum(dim=1)  # (B,)
        Na = max(active_counts.max().item(), 1)

        sort_idx = patch_active.float().sort(dim=1, descending=True, stable=True).indices
        compact_mask = torch.arange(Na, device=x_enc.device).unsqueeze(0) < active_counts.unsqueeze(1)

        # ── Compact all patch-level tensors: (B, N, ...) → (B, Na, ...) ──
        time_col = x_mark_enc if x_mark_enc.dim() == 3 else x_mark_enc.unsqueeze(-1)
        if time_col.shape[-1] != 1:
            time_col = time_col[..., :1]

        obs_by_patch = torch.stack([
            x_enc.reshape(B, num_patches, pl, C),
            x_mask.reshape(B, num_patches, pl, C),
            time_col.expand(-1, -1, C).reshape(B, num_patches, pl, C),
            indicator.reshape(B, num_patches, pl, C),
        ], dim=-1)  # (B, N, P, C, 4)
        obs_compact = self._compact_dim1(obs_by_patch, sort_idx, Na)  # (B, Na, P, C, 4)
        mask_compact = self._compact_dim1(
            x_mask.reshape(B, num_patches, pl, C), sort_idx, Na
        )

        # ── Stage 1a: Per-channel MLP tokenizer ──
        obs_patches = rearrange(obs_compact, "b n p c f -> (b c n) p f")  # (BCNa, P, 4)
        patch_mask = rearrange(mask_compact, "b n p c -> (b c n) p 1")

        h = self.tokenizer(obs_patches, patch_mask)
        h = rearrange(h, "(b c n) d -> (b c) n d", b=B, c=C, n=Na)

        # Add channel + position embeddings
        channel_bias = rearrange(
            self.channel_emb.weight.unsqueeze(0).expand(B, -1, -1),
            "b c d -> (b c) 1 d",
        )
        pos_emb = self._repeat_tail(self.patch_pos_emb, num_patches).expand(B, -1, -1)
        pos_compact = self._compact_dim1(pos_emb, sort_idx, Na)
        pos_compact = pos_compact.unsqueeze(1).expand(-1, C, -1, -1).reshape(B * C, Na, -1)
        h = h + channel_bias + pos_compact

        # ── Stage 1b: Joint signature context (on compacted patches) ──
        x_patches_c = obs_compact[..., 0]
        mask_patches_c = mask_compact
        time_patches_c = self._compact_dim1(
            time_col.reshape(B, num_patches, pl, 1), sort_idx, Na
        )
        joint_sig = self.joint_sig(x_patches_c, mask_patches_c, time_patches_c)

        # ── Stage 2: Adaptive Signature Kernel Attention ──
        for layer in self.layers:
            h = layer(h, joint_sig, compact_mask)

        h = self.final_norm(h)
        return h, compact_mask

    def forecast(self, x, x_mark, x_mask, y, y_mark, y_mask):
        B = x.shape[0]
        # Joint encoding: [input, prediction_targets]
        x_L = torch.cat([x, torch.zeros_like(y)], dim=1)
        mark_L = torch.cat([x_mark, y_mark], dim=1)
        mask_L = torch.cat([x_mask, y_mask], dim=1)
        indicator = torch.cat([torch.zeros_like(x_mask), y_mask], dim=1)

        h, patch_active = self._encode(x_L, mark_L, mask_L, indicator)
        return self.sig_decoder(h, y_mark, self.channel_emb.weight, patch_active)

    def forward(self, x, x_mark=None, x_mask=None, y=None,
                y_mark=None, y_mask=None, **kwargs):
        if self.task_name in ["long_term_forecast", "short_term_forecast"]:
            if x_mask is None:
                x_mask = torch.ones_like(x)
            if y is None:
                y = torch.zeros(x.shape[0], self.c_out, x.shape[2],
                                device=x.device, dtype=x.dtype)
            if y_mark is None:
                y_mark = x_mark
            if y_mask is None:
                y_mask = torch.ones_like(y)
            dec_out = self.forecast(x, x_mark, x_mask, y, y_mark, y_mask)
            return {"pred": dec_out, "true": y, "mask": y_mask}
        raise NotImplementedError("Only forecasting is supported.")

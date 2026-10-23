"""Readable Gemma 4 Unified text backbone, including a PyTorch NF4 path.

Equations and checkpoint names follow Transformers' Apache-2.0 reference at
02d8fb9784e8f14a1251e4c992cd82a5762417c6. Runtime does not import Transformers.
This first port implements full-sequence text prefill, without a KV cache.
"""

from __future__ import annotations

import json
import math
import hashlib
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn
from torch.nn import functional as F


CHECKPOINT = "google/gemma-4-12B-it"
REVISION = "707f0a3b8a3c7ad586ed01e27eafbad8a27dd0f7"
WEIGHT_SHA256 = "5a84cb313260ac447237b890387116dfa8682e49a6b44bc585ae8353abbff18d"


@dataclass(frozen=True)
class GemmaConfig:
    hidden_size: int = 3840
    intermediate_size: int = 15360
    num_hidden_layers: int = 48
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    vocab_size: int = 262144
    sliding_window: int = 1024
    rms_norm_eps: float = 1e-6
    final_logit_softcapping: float = 30.0
    layer_types: tuple[str, ...] = tuple(
        "full_attention" if i % 6 == 5 else "sliding_attention" for i in range(48)
    )

    @classmethod
    def from_json(cls, path: Path) -> "GemmaConfig":
        raw = json.loads(path.read_text())["text_config"]
        if (raw.get("num_kv_shared_layers", 0) or raw.get("enable_moe_block", False)
                or raw.get("use_double_wide_mlp", False) or raw.get("attention_bias", False)
                or not raw.get("attention_k_eq_v", False)
                or raw.get("hidden_activation") != "gelu_pytorch_tanh"):
            raise ValueError("This port supports the dense Unified 12B text architecture")
        expected_rope = {
            "full_attention": {"partial_rotary_factor": 0.25, "rope_theta": 1000000.0,
                               "rope_type": "proportional"},
            "sliding_attention": {"rope_theta": 10000.0, "rope_type": "default"},
        }
        if raw["rope_parameters"] != expected_rope:
            raise ValueError("Unsupported rotary-position configuration")
        values = {key: raw[key] for key in cls.__dataclass_fields__}
        values["layer_types"] = tuple(values["layer_types"])
        return cls(**values)


class RMSNorm(nn.Module):
    def __init__(self, width: int, eps: float, with_scale: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(width)) if with_scale else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * (y.square().mean(-1, keepdim=True) + self.eps).pow(-0.5)
        if self.weight is not None:
            y = y * self.weight.float()
        return y.to(x.dtype)


# NormalFloat4 levels, as used by the Apache-2.0 bitsandbytes reference.
# Scales remain FP32 here: no double quantization or fused CUDA dependency.
NF4_LEVELS = (-1.0, -0.6961928, -0.52507305, -0.39491749, -0.28444138,
              -0.18477343, -0.09105004, 0.0, 0.07958030, 0.16093020,
              0.24611230, 0.33791524, 0.44070983, 0.56261700, 0.72295684, 1.0)


class NF4Linear(nn.Module):
    """Packed 4-bit weights; materialize one BF16 matrix for each linear call.

    This intentionally transparent implementation measures fit and quality.
    It is not a fused quantized matmul performance claim.
    """

    def __init__(self, weight: torch.Tensor, block_size: int = 64):
        super().__init__()
        self.shape = tuple(weight.shape)
        self.block_size = block_size
        if weight.numel() % block_size or block_size % 2:
            raise ValueError("NF4 weight size must be divisible by an even block size")
        blocks = weight.float().reshape(-1, block_size)
        scales = blocks.abs().amax(-1).clamp_min(1e-12)
        levels = torch.tensor(NF4_LEVELS, device=weight.device)
        thresholds = (levels[1:] + levels[:-1]) / 2
        codes = torch.bucketize((blocks / scales[:, None]).contiguous(), thresholds).to(torch.uint8)
        packed = codes[:, 0::2] | (codes[:, 1::2] << 4)
        self.register_buffer("packed", packed)
        self.register_buffer("scales", scales)
        self.register_buffer("levels", levels)

    def dequantize(self, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
        codes = torch.stack((self.packed & 15, self.packed >> 4), dim=-1).reshape(-1, self.block_size)
        values = self.levels[codes.long()] * self.scales[:, None]
        return values.reshape(self.shape).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dequantize(x.dtype))


def rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    a, b = x.chunk(2, dim=-1)
    return x * cos[:, :, None] + torch.cat((-b, a), dim=-1) * sin[:, :, None]


class Attention(nn.Module):
    def __init__(self, c: GemmaConfig, full: bool):
        super().__init__()
        self.heads = c.num_attention_heads
        self.kv_heads = c.num_global_key_value_heads if full else c.num_key_value_heads
        self.head_dim = c.global_head_dim if full else c.head_dim
        self.q_proj = nn.Linear(c.hidden_size, self.heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(c.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.v_proj = None if full else nn.Linear(c.hidden_size, self.kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.heads * self.head_dim, c.hidden_size, bias=False)
        self.q_norm = RMSNorm(self.head_dim, c.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, c.rms_norm_eps)
        self.v_norm = RMSNorm(self.head_dim, c.rms_norm_eps, with_scale=False)

    def forward(self, x, positions, mask):
        batch, length, _ = x.shape
        cos, sin = positions
        q = self.q_proj(x).reshape(batch, length, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(batch, length, self.kv_heads, self.head_dim)
        v = self.v_proj(x).reshape_as(k) if self.v_proj is not None else k
        q = rotary(self.q_norm(q), cos, sin).transpose(1, 2)
        k = rotary(self.k_norm(k), cos, sin).transpose(1, 2)
        v = self.v_norm(v).transpose(1, 2)
        # Match the reference's explicit GQA expansion when a mask is present.
        # Masked implicit GQA selects a different math kernel in Torch 2.8.
        groups = self.heads // self.kv_heads
        # Preserve the broadcast stride of the one-head global K/V tensors.
        k = k[:, :, None].expand(-1, -1, groups, -1, -1).reshape(batch, self.heads, length, self.head_dim)
        v = v[:, :, None].expand(-1, -1, groups, -1, -1).reshape(batch, self.heads, length, self.head_dim)
        # Gemma's Q/K normalization makes its attention scale 1.0.
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, scale=1.0)
        return self.o_proj(y.transpose(1, 2).reshape(batch, length, -1))


class MLP(nn.Module):
    def __init__(self, c: GemmaConfig):
        super().__init__()
        self.gate_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.up_proj = nn.Linear(c.hidden_size, c.intermediate_size, bias=False)
        self.down_proj = nn.Linear(c.intermediate_size, c.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class Layer(nn.Module):
    def __init__(self, c: GemmaConfig, full: bool):
        super().__init__()
        self.self_attn = Attention(c, full)
        self.mlp = MLP(c)
        self.input_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.pre_feedforward_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.post_feedforward_layernorm = RMSNorm(c.hidden_size, c.rms_norm_eps)
        self.register_buffer("layer_scalar", torch.ones(1))

    def forward(self, x, positions, mask):
        x = x + self.post_attention_layernorm(self.self_attn(self.input_layernorm(x), positions, mask))
        x = x + self.post_feedforward_layernorm(self.mlp(self.pre_feedforward_layernorm(x)))
        return x * self.layer_scalar


class GemmaTextModel(nn.Module):
    def __init__(self, config: GemmaConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(Layer(config, kind == "full_attention") for kind in config.layer_types)
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def masks_and_positions(self, input_ids, attention_mask):
        c = self.config
        length = input_ids.shape[1]
        p = torch.arange(length, device=input_ids.device)
        allowed = p[:, None] >= p[None, :]
        masks = {
            "full_attention": allowed[None, None] & attention_mask[:, None, None, :].bool(),
            "sliding_attention": (allowed & ((p[:, None] - p[None, :]) < c.sliding_window))[None, None]
            & attention_mask[:, None, None, :].bool(),
        }
        positions = {}
        for kind, dim, base, proportion in (
            ("sliding_attention", c.head_dim, 10000.0, 1.0),
            ("full_attention", c.global_head_dim, 1000000.0, 0.25),
        ):
            count = int(dim * proportion // 2)
            inv = 1.0 / (base ** (torch.arange(0, 2 * count, 2, device=p.device).float() / dim))
            inv = F.pad(inv, (0, dim // 2 - count))
            angles = (inv[None, :, None] @ p[None, None, :].float()).transpose(1, 2)
            angles = torch.cat((angles, angles), dim=-1)
            positions[kind] = (angles.cos().to(self.embed_tokens.weight.dtype),
                               angles.sin().to(self.embed_tokens.weight.dtype))
        return masks, positions

    def forward(self, input_ids, attention_mask=None):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
            raise ValueError("IDs and attention mask must have shape [batch, tokens]")
        masks, positions = self.masks_and_positions(input_ids, attention_mask)
        x = self.embed_tokens(input_ids)
        # Match the BF16 rounding of the published embedding multiplier.
        x = x * torch.tensor(math.sqrt(self.config.hidden_size), dtype=x.dtype, device=x.device)
        for layer, kind in zip(self.layers, self.config.layer_types):
            x = layer(x, positions[kind], masks[kind])
        return self.norm(x)

    def logits(self, hidden: torch.Tensor, token_ids: torch.Tensor | None = None):
        weight = self.embed_tokens.weight if token_ids is None else self.embed_tokens.weight[token_ids]
        logits = F.linear(hidden, weight).float()
        cap = self.config.final_logit_softcapping
        return (logits / cap).tanh() * cap


def verify_weights(directory: Path) -> str:
    digest = hashlib.sha256()
    with (directory / "model.safetensors").open("rb") as source:
        for block in iter(lambda: source.read(16 * 1024 * 1024), b""):
            digest.update(block)
    actual = digest.hexdigest()
    if actual != WEIGHT_SHA256:
        raise ValueError(f"Gemma weight SHA-256 mismatch: {actual}")
    return actual


def load_model(directory: Path, *, device="cuda", quantization="nf4", verify=True) -> GemmaTextModel:
    """Stream official tensors and quantize only linear matrices.

    Embeddings and normalization stay BF16. At no point are all dense model
    weights resident on GPU. Quantization is frozen; head training is separate.
    """
    if quantization not in ("nf4", "bnb-nf4", "bf16"):
        raise ValueError("quantization must be nf4, bnb-nf4 or bf16")
    if verify:
        verify_weights(directory)
    config = GemmaConfig.from_json(directory / "config.json")
    with torch.device("meta"):
        model = GemmaTextModel(config)
    prefix = "model.language_model."
    with safe_open(directory / "model.safetensors", framework="pt", device="cpu") as source:
        keys = set(source.keys())
        wanted = {prefix + key for key in model.state_dict()}
        if not wanted <= keys:
            raise ValueError(f"Missing text weights: {sorted(wanted - keys)[:8]}")
        for name, module in list(model.named_modules()):
            if isinstance(module, nn.Linear) and quantization != "bf16":
                weight = source.get_tensor(prefix + name + ".weight")
                parent_name, attr = name.rsplit(".", 1)
                if quantization == "nf4":
                    replacement = NF4Linear(weight.to(device))
                else:
                    # Optional acceleration; model equations remain above.
                    from bitsandbytes.nn import Linear4bit, Params4bit
                    replacement = Linear4bit(module.in_features, module.out_features,
                                             bias=False, compute_dtype=torch.bfloat16,
                                             compress_statistics=False, quant_type="nf4")
                    replacement.weight = Params4bit(weight, requires_grad=False,
                                                   compress_statistics=False, quant_type="nf4")
                    replacement = replacement.to(device)
                setattr(model.get_submodule(parent_name), attr, replacement)
                del weight
            else:
                for attr, tensor in list(module.named_parameters(recurse=False)):
                    value = source.get_tensor(prefix + name + "." + attr).to(device=device, dtype=torch.bfloat16)
                    setattr(module, attr, nn.Parameter(value, requires_grad=False))
                for attr, tensor in list(module.named_buffers(recurse=False)):
                    value = source.get_tensor(prefix + name + "." + attr).to(device=device, dtype=torch.bfloat16)
                    setattr(module, attr, value)
    model.requires_grad_(False).eval()
    torch.cuda.empty_cache() if str(device).startswith("cuda") else None
    return model

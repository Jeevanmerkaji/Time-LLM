"""
Time-LLM reproduction (simplified) - GPT-2 backbone version
Based on: "Time-LLM: Time Series Forecasting by Reprogramming Large Language Models"
(Jin et al., ICLR 2024) - https://arxiv.org/abs/2310.01728

This is a simplified, single-file reimplementation for learning + reproduction
purposes. It captures the core idea of the paper:
  1. Patch the input time series into short windows
  2. Project patches into the LLM's embedding space ("reprogramming")
  3. Feed reprogrammed patches + a text prompt prefix into a FROZEN LLM
  4. Only the reprogramming layer + output head are trained

For a full-fidelity reproduction, use the official repo instead:
https://github.com/KimMeen/Time-LLM
This version trades some fidelity for clarity and ease of running on modest
hardware (CPU or a single small GPU).
"""

import math
import numpy as np
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Tokenizer, GPT2Config


class RevIN(nn.Module):
    """
    Reversible Instance Normalization (Kim et al., 2021), used by the paper
    to normalize each input window using its own mean/std, then reverse that
    same transform on the model's output before the loss is computed. This
    handles per-instance distribution shift far better than a single global
    train-set mean/std.
    """

    def __init__(self, eps: float = 1e-5, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.affine = affine
        if affine:
            self.affine_weight = nn.Parameter(torch.ones(1))
            self.affine_bias = nn.Parameter(torch.zeros(1))

    def forward(self, x, mode: str):
        # x: [batch, seq_len]
        if mode == "norm":
            self.mean = x.mean(dim=-1, keepdim=True).detach()
            self.std = (x.var(dim=-1, keepdim=True, unbiased=False) + self.eps).sqrt().detach()
            x = (x - self.mean) / self.std
            if self.affine:
                x = x * self.affine_weight + self.affine_bias
            return x
        elif mode == "denorm":
            if self.affine:
                x = (x - self.affine_bias) / (self.affine_weight + self.eps * self.eps)
            return x * self.std + self.mean
        raise ValueError(f"Unknown RevIN mode: {mode}")


class PatchEmbedding(nn.Module):
    """Splits a univariate time series into overlapping patches and embeds them."""

    def __init__(self, patch_len: int, stride: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.value_embedding = nn.Linear(patch_len, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len]  (univariate series, already normalized)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x: [batch, num_patches, patch_len]
        x = self.value_embedding(x)
        # x: [batch, num_patches, d_model]
        return self.dropout(x)


class ReprogrammingLayer(nn.Module):
    """
    Cross-attention layer that maps patch embeddings into the LLM's word-
    embedding space using a small vocabulary of learned "text prototypes"
    (a compressed subset of the LLM's own embedding matrix). This is the
    heart of the Time-LLM idea: patches attend over prototype word embeddings
    so the frozen LLM sees something that resembles its native input space.
    """

    def __init__(self, d_model: int, n_heads: int, llm_dim: int, num_prototypes: int = 1000):
        super().__init__()
        self.n_heads = n_heads
        self.d_keys = d_model // n_heads

        self.query_proj = nn.Linear(d_model, self.d_keys * n_heads)
        self.key_proj = nn.Linear(llm_dim, self.d_keys * n_heads)
        self.value_proj = nn.Linear(llm_dim, self.d_keys * n_heads)
        self.out_proj = nn.Linear(self.d_keys * n_heads, llm_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, patch_embeddings, prototype_embeddings):
        # patch_embeddings: [batch, num_patches, d_model]
        # prototype_embeddings: [num_prototypes, llm_dim]  (subset of LLM vocab embeddings)
        B, L, _ = patch_embeddings.shape
        S, _ = prototype_embeddings.shape
        H = self.n_heads

        Q = self.query_proj(patch_embeddings).view(B, L, H, self.d_keys)
        K = self.key_proj(prototype_embeddings).view(S, H, self.d_keys)
        V = self.value_proj(prototype_embeddings).view(S, H, self.d_keys)

        scale = 1.0 / math.sqrt(self.d_keys)
        # attention: each patch attends over all prototype tokens
        scores = torch.einsum("blhd,shd->bhls", Q, K) * scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhls,shd->blhd", attn, V)
        out = out.reshape(B, L, H * self.d_keys)
        return self.out_proj(out)  # [batch, num_patches, llm_dim]


class TimeLLM(nn.Module):
    def __init__(
        self,
        seq_len: int = 96,
        pred_len: int = 24,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 32,
        n_heads: int = 4,
        num_prototypes: int = 1000,
        llm_name: str = "gpt2",
        freeze_llm: bool = True,
        description: str = "Time series forecasting.",
        prompt_max_tokens: int = 48,
        use_prompt: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.description = description
        self.prompt_max_tokens = prompt_max_tokens
        self.use_prompt = use_prompt

        # --- Frozen LLM backbone ---
        self.tokenizer = GPT2Tokenizer.from_pretrained(llm_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Left-pad so every prompt's real tokens sit immediately before the
        # patch tokens that follow, regardless of per-sample prompt length.
        self.tokenizer.padding_side = "left"
        # If a prompt does get truncated, drop from the front (the static
        # instruction) rather than the back (the per-window stats, which
        # matter more and are cheap to compute so worth keeping).
        self.tokenizer.truncation_side = "left"
        self.llm = GPT2Model.from_pretrained(llm_name)
        llm_dim = self.llm.config.n_embd

        if freeze_llm:
            for p in self.llm.parameters():
                p.requires_grad = False

        # Subsample the LLM's vocab embedding matrix down to `num_prototypes`
        # rows to act as "text prototypes" (paper does this to keep the
        # reprogramming layer's attention over a manageable, informative set)
        full_vocab_emb = self.llm.get_input_embeddings().weight  # [vocab_size, llm_dim]
        idx = torch.linspace(0, full_vocab_emb.shape[0] - 1, num_prototypes).long()
        self.register_buffer("prototype_idx", idx)
        self.prototype_source = full_vocab_emb  # kept frozen, indexed at forward time

        # --- Trainable components ---
        num_patches = (seq_len - patch_len) // stride + 1
        self.revin = RevIN()
        self.patch_embed = PatchEmbedding(patch_len, stride, d_model)
        self.reprogram = ReprogrammingLayer(d_model, n_heads, llm_dim, num_prototypes)
        self.output_proj = nn.Linear(num_patches * llm_dim, pred_len)

    @staticmethod
    def _top_lags(series: np.ndarray, k: int = 5) -> list:
        """Top-k autocorrelation lags (FFT-based), excluding lag 0."""
        n = len(series)
        centered = series - series.mean()
        f = np.fft.rfft(centered)
        acf = np.fft.irfft(f * np.conjugate(f), n=n)[: n // 2]
        order = np.argsort(-acf[1:]) + 1  # skip lag 0 (always maximal)
        return order[:k].tolist()

    def _build_prompts(self, x_raw: torch.Tensor) -> list:
        """Builds one Prompt-as-Prefix text string per sample in the batch,
        using that sample's own raw (pre-RevIN) input statistics."""
        x_np = x_raw.detach().cpu().numpy()
        prompts = []
        for series in x_np:
            trend = "up" if series[-1] > series[0] else "down"
            lags = ",".join(str(v) for v in self._top_lags(series))
            # Kept deliberately compact (vs. the paper's full natural-language
            # prompt) since every extra token here is extra frozen-LLM
            # sequence length on every forward pass — expensive on CPU.
            prompts.append(
                f"{self.description} Predict next {self.pred_len} given "
                f"previous {self.seq_len}. min={series.min():.2f} "
                f"max={series.max():.2f} median={np.median(series):.2f} "
                f"trend={trend} lags={lags}"
            )
        return prompts

    def forward(self, x):
        # x: [batch, seq_len]  -- raw (unnormalized) univariate series
        if self.use_prompt:
            prompts = self._build_prompts(x)
            tok = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=True,
                max_length=self.prompt_max_tokens,
            ).to(x.device)
            prompt_embeds = self.llm.get_input_embeddings()(tok.input_ids)  # [batch, prompt_len, llm_dim]

        x = self.revin(x, "norm")  # instance-normalized using this window's own mean/std

        prototypes = self.prototype_source[self.prototype_idx].detach()  # [num_prototypes, llm_dim]

        patches = self.patch_embed(x)  # [batch, num_patches, d_model]
        reprogrammed = self.reprogram(patches, prototypes)  # [batch, num_patches, llm_dim]
        num_patches = reprogrammed.shape[1]

        if self.use_prompt:
            combined = torch.cat([prompt_embeds, reprogrammed], dim=1)  # [batch, prompt_len + num_patches, llm_dim]
            patch_mask = torch.ones(
                x.shape[0], num_patches, device=x.device, dtype=tok.attention_mask.dtype
            )
            attention_mask = torch.cat([tok.attention_mask, patch_mask], dim=1)
            # Left-padding shifts real token positions; recompute position_ids so
            # padded slots don't consume position indices meant for real tokens.
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 0)

            llm_out = self.llm(
                inputs_embeds=combined, attention_mask=attention_mask, position_ids=position_ids
            ).last_hidden_state  # [batch, prompt_len + num_patches, llm_dim]
            llm_out = llm_out[:, -num_patches:, :]  # discard prefix positions, keep only patch outputs
        else:
            llm_out = self.llm(inputs_embeds=reprogrammed).last_hidden_state  # [batch, num_patches, llm_dim]

        flat = llm_out.reshape(llm_out.shape[0], -1)
        forecast = self.output_proj(flat)  # [batch, pred_len], still in normalized space

        return self.revin(forecast, "denorm")  # reversed back to original units

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

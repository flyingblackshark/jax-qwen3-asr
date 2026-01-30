import logging
import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

import jax
import jax.numpy as jnp
from flax import nnx
from transformers import PretrainedConfig

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.layers.embeddings import Embed, ParallelLMHead
from sgl_jax.srt.layers.layernorm import RMSNorm
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessor
from sgl_jax.srt.mem_cache.memory_pool import KVCache
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.models.qwen3 import QWen3DecoderLayer
from sgl_jax.srt.precision_tracer import precision_tracer
from sgl_jax.srt.utils.weight_utils import WeightLoader, WeightMapping

logger = logging.getLogger(__name__)


def _conv_out_length(length: int) -> int:
    for _ in range(3):
        length = (length + 1) // 2
    return int(length)


def _feat_extract_output_lengths(input_lengths: jax.Array) -> jax.Array:
    input_lengths = input_lengths.astype(jnp.int32)
    input_lengths_leave = input_lengths % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
    return output_lengths.astype(jnp.int32)


@dataclass(frozen=True)
class Qwen3ASRAudioEncoderConfig:
    num_mel_bins: int = 128
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    d_model: int = 1024
    dropout: float = 0.0
    attention_dropout: float = 0.0
    activation_function: str = "gelu"
    activation_dropout: float = 0.0
    scale_embedding: bool = False
    initializer_range: float = 0.02
    max_source_positions: int = 1500
    n_window: int = 50
    output_dim: int = 2048
    n_window_infer: int = 800
    conv_chunksize: int = 500
    downsample_hidden_size: int = 480

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen3ASRAudioEncoderConfig":
        return cls(
            num_mel_bins=int(data.get("num_mel_bins", cls.num_mel_bins)),
            encoder_layers=int(data.get("encoder_layers", cls.encoder_layers)),
            encoder_attention_heads=int(
                data.get("encoder_attention_heads", cls.encoder_attention_heads)
            ),
            encoder_ffn_dim=int(data.get("encoder_ffn_dim", cls.encoder_ffn_dim)),
            d_model=int(data.get("d_model", cls.d_model)),
            dropout=float(data.get("dropout", cls.dropout)),
            attention_dropout=float(data.get("attention_dropout", cls.attention_dropout)),
            activation_function=str(data.get("activation_function", cls.activation_function)),
            activation_dropout=float(data.get("activation_dropout", cls.activation_dropout)),
            scale_embedding=bool(data.get("scale_embedding", cls.scale_embedding)),
            initializer_range=float(data.get("initializer_range", cls.initializer_range)),
            max_source_positions=int(data.get("max_source_positions", cls.max_source_positions)),
            n_window=int(data.get("n_window", cls.n_window)),
            output_dim=int(data.get("output_dim", cls.output_dim)),
            n_window_infer=int(data.get("n_window_infer", cls.n_window_infer)),
            conv_chunksize=int(data.get("conv_chunksize", cls.conv_chunksize)),
            downsample_hidden_size=int(data.get("downsample_hidden_size", cls.downsample_hidden_size)),
        )


@dataclass(frozen=True)
class Qwen3ASRThinkerConfig:
    audio_config: Qwen3ASRAudioEncoderConfig
    audio_token_id: int = 151676

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen3ASRThinkerConfig":
        audio_cfg = Qwen3ASRAudioEncoderConfig.from_dict(data.get("audio_config", {}))
        return cls(
            audio_config=audio_cfg,
            audio_token_id=int(data.get("audio_token_id", cls.audio_token_id)),
        )


@dataclass(frozen=True)
class Qwen3ASRConfig:
    thinker_config: Qwen3ASRThinkerConfig

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Qwen3ASRConfig":
        thinker = Qwen3ASRThinkerConfig.from_dict(data.get("thinker_config", {}))
        return cls(thinker_config=thinker)


def _parse_asr_config(config: PretrainedConfig) -> Qwen3ASRConfig:
    if isinstance(config, Mapping):
        return Qwen3ASRConfig.from_dict(dict(config))

    thinker_cfg = getattr(config, "thinker_config", None)
    if thinker_cfg is None:
        return Qwen3ASRConfig.from_dict({})

    if isinstance(thinker_cfg, Mapping):
        thinker_dict = dict(thinker_cfg)
    elif hasattr(thinker_cfg, "to_dict"):
        try:
            thinker_dict = thinker_cfg.to_dict()
        except Exception:
            thinker_dict = {}
    else:
        thinker_dict = {}

    audio_token_id = getattr(config, "audio_token_id", None)
    if audio_token_id is not None:
        thinker_dict.setdefault("audio_token_id", audio_token_id)

    return Qwen3ASRConfig.from_dict({"thinker_config": thinker_dict})


class Qwen3ASRAudioAttention(nnx.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig, *, rngs: nnx.Rngs):
        self.embed_dim = int(config.d_model)
        self.num_heads = int(config.encoder_attention_heads)
        if self.embed_dim % self.num_heads != 0:
            raise ValueError(
                f"d_model={self.embed_dim} must be divisible by encoder_attention_heads={self.num_heads}."
            )
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nnx.Linear(self.embed_dim, self.embed_dim, use_bias=True, rngs=rngs)
        self.k_proj = nnx.Linear(self.embed_dim, self.embed_dim, use_bias=True, rngs=rngs)
        self.v_proj = nnx.Linear(self.embed_dim, self.embed_dim, use_bias=True, rngs=rngs)
        self.out_proj = nnx.Linear(self.embed_dim, self.embed_dim, use_bias=True, rngs=rngs)

    def __call__(self, hidden_states: jax.Array, *, token_mask: Optional[jax.Array] = None):
        bsz, seq_len, _ = hidden_states.shape
        q = (
            self.q_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        k = (
            self.k_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )
        v = (
            self.v_proj(hidden_states)
            .reshape(bsz, seq_len, self.num_heads, self.head_dim)
            .transpose(0, 2, 1, 3)
        )

        attn_weights = jnp.matmul(q, k.transpose(0, 1, 3, 2)) * (self.head_dim**-0.5)
        if token_mask is not None:
            key_bias = jnp.where(token_mask, 0.0, -1e9).astype(attn_weights.dtype)
            attn_weights = attn_weights + key_bias[:, None, None, :]

        attn_probs = jax.nn.softmax(attn_weights.astype(jnp.float32), axis=-1).astype(
            attn_weights.dtype
        )
        attn_output = jnp.matmul(attn_probs, v)
        if token_mask is not None:
            attn_output = attn_output * token_mask[:, None, :, None]

        attn_output = (
            attn_output.transpose(0, 2, 1, 3)
            .reshape(bsz, seq_len, self.embed_dim)
        )
        return self.out_proj(attn_output)


class Qwen3ASRAudioEncoderLayer(nnx.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig, *, rngs: nnx.Rngs):
        self.self_attn = Qwen3ASRAudioAttention(config, rngs=rngs)
        self.self_attn_layer_norm = nnx.LayerNorm(int(config.d_model), rngs=rngs)
        self.final_layer_norm = nnx.LayerNorm(int(config.d_model), rngs=rngs)

        self.fc1 = nnx.Linear(
            int(config.d_model), int(config.encoder_ffn_dim), use_bias=True, rngs=rngs
        )
        self.fc2 = nnx.Linear(
            int(config.encoder_ffn_dim), int(config.d_model), use_bias=True, rngs=rngs
        )
        self.activation = str(config.activation_function)

    def __call__(self, hidden_states: jax.Array, *, token_mask: Optional[jax.Array] = None):
        residual = hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)
        hidden_states = self.self_attn(hidden_states, token_mask=token_mask)
        hidden_states = residual + hidden_states
        if token_mask is not None:
            hidden_states = hidden_states * token_mask[..., None]

        residual = hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        hidden_states = self.fc1(hidden_states)
        if self.activation == "gelu":
            hidden_states = jax.nn.gelu(hidden_states, approximate=False)
        elif self.activation == "silu":
            hidden_states = jax.nn.silu(hidden_states)
        elif self.activation == "relu":
            hidden_states = jax.nn.relu(hidden_states)
        else:
            raise ValueError(f"Unsupported activation_function={self.activation!r}")
        hidden_states = self.fc2(hidden_states)
        hidden_states = residual + hidden_states
        if token_mask is not None:
            hidden_states = hidden_states * token_mask[..., None]
        return hidden_states


class SinusoidsPositionEmbedding(nnx.Module):
    def __init__(self, length: int, channels: int, *, max_timescale: float = 10000.0):
        if channels % 2 != 0:
            raise ValueError("SinusoidsPositionEmbedding requires even `channels`.")
        self.length = int(length)
        self.channels = int(channels)
        self.max_timescale = float(max_timescale)

        self.positional_embedding = nnx.Cache(self._build_embedding())

    def _build_embedding(self) -> jax.Array:
        log_timescale_increment = math.log(self.max_timescale) / (self.channels // 2 - 1)
        inv_timescales = jnp.exp(
            -log_timescale_increment * jnp.arange(self.channels // 2, dtype=jnp.float32)
        )
        scaled_time = jnp.arange(self.length, dtype=jnp.float32)[:, None] * inv_timescales[None, :]
        return jnp.concatenate([jnp.sin(scaled_time), jnp.cos(scaled_time)], axis=1)

    def materialize(self):
        emb = self.positional_embedding.value
        if isinstance(emb, jax.ShapeDtypeStruct):
            self.positional_embedding = nnx.Cache(self._build_embedding())

    def __call__(self, seqlen: int) -> jax.Array:
        emb = self.positional_embedding.value
        if isinstance(emb, jax.ShapeDtypeStruct):
            emb = self._build_embedding()
        return emb[:seqlen, :]


class Qwen3ASRAudioEncoder(nnx.Module):
    def __init__(self, config: Qwen3ASRAudioEncoderConfig, *, rngs: nnx.Rngs):
        self.config = config
        chunk_size = int(config.n_window) * 2
        if chunk_size != 100:
            raise ValueError(
                f"Expected n_window*2 == 100, got {chunk_size} for Qwen3-ASR audio encoder."
            )
        self.chunk_size = chunk_size
        self.embed_dim = int(config.d_model)
        self.out_dim = int(config.output_dim)

        self.conv2d1 = nnx.Conv(
            1,
            int(config.downsample_hidden_size),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        self.conv2d2 = nnx.Conv(
            int(config.downsample_hidden_size),
            int(config.downsample_hidden_size),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )
        self.conv2d3 = nnx.Conv(
            int(config.downsample_hidden_size),
            int(config.downsample_hidden_size),
            kernel_size=(3, 3),
            strides=(2, 2),
            padding=((1, 1), (1, 1)),
            rngs=rngs,
        )

        freq_out = _conv_out_length(int(config.num_mel_bins))
        self.conv_time_out = _conv_out_length(chunk_size)
        self.conv_out = nnx.Linear(
            int(config.downsample_hidden_size) * freq_out,
            self.embed_dim,
            use_bias=False,
            rngs=rngs,
        )

        self.positional_embedding = SinusoidsPositionEmbedding(
            int(config.max_source_positions), self.embed_dim
        )

        self.layers = nnx.List(
            [Qwen3ASRAudioEncoderLayer(config, rngs=rngs) for _ in range(int(config.encoder_layers))]
        )
        self.ln_post = nnx.LayerNorm(self.embed_dim, rngs=rngs)
        self.proj1 = nnx.Linear(self.embed_dim, self.embed_dim, use_bias=True, rngs=rngs)
        self.proj2 = nnx.Linear(self.embed_dim, self.out_dim, use_bias=True, rngs=rngs)
        self.activation = str(config.activation_function)

        ratio = int(config.n_window_infer) // chunk_size
        if ratio < 1 or (int(config.n_window_infer) % chunk_size) != 0:
            raise ValueError(
                f"Expected n_window_infer ({config.n_window_infer}) to be a multiple of chunk_size ({chunk_size})."
            )
        self.window_len = self.conv_time_out * ratio

    def __call__(
        self,
        input_features: jax.Array,
        *,
        feature_attention_mask: Optional[jax.Array] = None,
    ) -> tuple[jax.Array, jax.Array]:
        bsz, n_mels, n_frames = input_features.shape
        if feature_attention_mask is None:
            feature_attention_mask = jnp.ones((bsz, n_frames), dtype=jnp.int32)

        feature_lens = feature_attention_mask.sum(axis=-1).astype(jnp.int32)

        pad_frames = (-n_frames) % self.chunk_size
        if pad_frames:
            input_features = jnp.pad(input_features, ((0, 0), (0, 0), (0, pad_frames)))
            feature_attention_mask = jnp.pad(feature_attention_mask, ((0, 0), (0, pad_frames)))

        n_frames_padded = input_features.shape[-1]
        num_chunks = n_frames_padded // self.chunk_size

        chunks = (
            input_features.reshape(bsz, n_mels, num_chunks, self.chunk_size)
            .transpose(0, 2, 1, 3)
        )
        x = chunks.reshape(bsz * num_chunks, n_mels, self.chunk_size)[..., None]

        x = nnx.gelu(self.conv2d1(x))
        x = nnx.gelu(self.conv2d2(x))
        x = nnx.gelu(self.conv2d3(x))

        x = x.transpose(0, 2, 3, 1).reshape(bsz * num_chunks, self.conv_time_out, -1)
        x = self.conv_out(x)

        pos = self.positional_embedding(self.conv_time_out)[None, :, :].astype(x.dtype)
        x = x + pos
        x = x.reshape(bsz, num_chunks, self.conv_time_out, self.embed_dim)

        chunk_starts = (jnp.arange(num_chunks, dtype=jnp.int32) * self.chunk_size)[None, :]
        remaining = feature_lens[:, None] - chunk_starts
        chunk_lens = jnp.clip(remaining, 0, self.chunk_size)
        out_lens = _feat_extract_output_lengths(chunk_lens)
        t_idx = jnp.arange(self.conv_time_out, dtype=jnp.int32)[None, None, :]
        chunk_token_mask = t_idx < out_lens[..., None]
        chunk_token_mask = chunk_token_mask & (chunk_lens[..., None] > 0)

        token_states = x.reshape(bsz, num_chunks * self.conv_time_out, self.embed_dim)
        token_mask = chunk_token_mask.reshape(bsz, num_chunks * self.conv_time_out)

        pad_tokens = (-token_states.shape[1]) % self.window_len
        if pad_tokens:
            token_states = jnp.pad(token_states, ((0, 0), (0, pad_tokens), (0, 0)))
            token_mask = jnp.pad(token_mask, ((0, 0), (0, pad_tokens)))

        num_windows = token_states.shape[1] // self.window_len
        win_states = token_states.reshape(bsz * num_windows, self.window_len, self.embed_dim)
        win_mask = token_mask.reshape(bsz * num_windows, self.window_len).astype(bool)

        for layer in self.layers:
            win_states = layer(win_states, token_mask=win_mask)

        win_states = self.ln_post(win_states)
        win_states = self.proj1(win_states)
        if self.activation == "gelu":
            win_states = jax.nn.gelu(win_states, approximate=False)
        elif self.activation == "silu":
            win_states = jax.nn.silu(win_states)
        elif self.activation == "relu":
            win_states = jax.nn.relu(win_states)
        else:
            raise ValueError(f"Unsupported activation_function={self.activation!r}")

        win_states = self.proj2(win_states)
        win_states = win_states * win_mask[..., None]

        token_states = win_states.reshape(bsz, num_windows * self.window_len, self.out_dim)
        token_states = token_states[:, : num_chunks * self.conv_time_out, :]
        token_mask = token_mask[:, : num_chunks * self.conv_time_out].astype(bool)
        return token_states, token_mask


class Qwen3ASRTextModel(nnx.Module):
    def __init__(self, config: PretrainedConfig, mesh: jax.sharding.Mesh, dtype: jnp.dtype):
        self.layers = nnx.data(
            [
                QWen3DecoderLayer(config=config, layer_id=i, dtype=dtype, mesh=mesh)
                for i in range(config.num_hidden_layers)
            ]
        )
        self.norm = RMSNorm(
            config.hidden_size,
            epsilon=config.rms_norm_eps,
            param_dtype=dtype,
            scope_name="norm",
        )
        self.layers_to_capture = []

    def __call__(self, forward_batch: ForwardBatch, token_to_kv_pool: KVCache, hidden_states):
        residual = None
        layers_kv_fused = []
        layers_callback_flag = []
        aux_hidden_states = []
        for layer_id, layer in enumerate(self.layers):
            if layer_id in self.layers_to_capture:
                aux_hidden_states.append(
                    hidden_states + residual if residual is not None else hidden_states
                )
            hidden_states, residual, kv_fused, callback_flag = layer(
                forward_batch.positions,
                hidden_states,
                forward_batch,
                token_to_kv_pool,
                residual,
            )
            layers_kv_fused.append(kv_fused)
            layers_callback_flag.extend(callback_flag)

        if residual is not None:
            hidden_states += residual
        hidden_states = self.norm(hidden_states)

        callback_flag = precision_tracer.jit_pure_callback_record(
            hidden_states, "transformer_output", "TRANSFORMER"
        )
        layers_callback_flag.append(callback_flag)
        return hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag


class Qwen3ASRForConditionalGeneration(nnx.Module):
    def __init__(self, config: PretrainedConfig, dtype: jnp.dtype, mesh: jax.sharding.Mesh):
        self.mesh = mesh
        self.config = config
        self.dtype = dtype

        asr_cfg = _parse_asr_config(config)
        self.audio_token_id = int(asr_cfg.thinker_config.audio_token_id)

        self.audio_tower = Qwen3ASRAudioEncoder(asr_cfg.thinker_config.audio_config, rngs=nnx.Rngs(0))
        self.embed_tokens = Embed(
            num_embeddings=config.vocab_size,
            features=config.hidden_size,
            dtype=dtype,
            kernel_axes=("tensor", None),
            param_dtype=dtype,
            mesh=mesh,
        )
        self.model = Qwen3ASRTextModel(config, mesh=mesh, dtype=dtype)
        if not getattr(self.config, "tie_word_embeddings", False):
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                dtype=dtype,
                param_dtype=dtype,
                kernel_axes=("tensor", None),
            )
        self.logits_processor = LogitsProcessor(config.vocab_size, mesh=self.mesh)

        # For EAGLE3 support (unused for ASR)
        self.capture_aux_hidden_states = False

    def _merge_audio(
        self,
        hidden_states: jax.Array,
        forward_batch: ForwardBatch,
        audio_embeds: jax.Array,
    ) -> jax.Array:
        input_ids = forward_batch.input_ids
        positions = forward_batch.positions
        start_locs = forward_batch.extend_start_loc

        token_idx = jnp.arange(input_ids.shape[0], dtype=jnp.int32)
        seq_ids = jnp.sum(token_idx[None, :] >= start_locs[:, None], axis=0) - 1
        seq_ids = jnp.clip(seq_ids, 0, start_locs.shape[0] - 1)

        audio_start = forward_batch.audio_token_start[seq_ids]
        audio_len = forward_batch.audio_token_len[seq_ids]
        audio_index = positions - audio_start

        audio_mask = (
            (input_ids == self.audio_token_id)
            & (audio_start >= 0)
            & (audio_index >= 0)
            & (audio_index < audio_len)
        )

        max_audio_len = audio_embeds.shape[1]
        audio_index = jnp.clip(audio_index, 0, max_audio_len - 1)
        gathered = audio_embeds[seq_ids, audio_index]
        return jnp.where(audio_mask[:, None], gathered, hidden_states)

    def load_weights(self, model_config: ModelConfig):
        loader = WeightLoader(
            model=self,
            model_config=model_config,
            mesh=self.mesh,
            dtype=self.dtype,
        )
        weight_mappings = {}
        weight_mappings.update(self._create_audio_weight_mappings())
        weight_mappings.update(self._create_text_weight_mappings())
        loader.load_weights_from_safetensors(weight_mappings)
        # Ensure positional embedding cache is materialized after eval_shape.
        try:
            self.audio_tower.positional_embedding.materialize()
        except Exception:
            pass
        logger.info("Qwen3-ASR weights loaded successfully!")

    def _create_audio_weight_mappings(self) -> dict:
        mappings = {
            "thinker.audio_tower.conv2d1.weight": WeightMapping(
                target_path="audio_tower.conv2d1.kernel",
                sharding=(None, None, None, None),
                transpose_axes=(2, 3, 1, 0),
            ),
            "thinker.audio_tower.conv2d1.bias": WeightMapping(
                target_path="audio_tower.conv2d1.bias", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.conv2d2.weight": WeightMapping(
                target_path="audio_tower.conv2d2.kernel",
                sharding=(None, None, None, None),
                transpose_axes=(2, 3, 1, 0),
            ),
            "thinker.audio_tower.conv2d2.bias": WeightMapping(
                target_path="audio_tower.conv2d2.bias", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.conv2d3.weight": WeightMapping(
                target_path="audio_tower.conv2d3.kernel",
                sharding=(None, None, None, None),
                transpose_axes=(2, 3, 1, 0),
            ),
            "thinker.audio_tower.conv2d3.bias": WeightMapping(
                target_path="audio_tower.conv2d3.bias", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.conv_out.weight": WeightMapping(
                target_path="audio_tower.conv_out.kernel",
                sharding=(None, None),
                transpose=True,
            ),
            "thinker.audio_tower.ln_post.weight": WeightMapping(
                target_path="audio_tower.ln_post.scale", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.ln_post.bias": WeightMapping(
                target_path="audio_tower.ln_post.bias", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.proj1.weight": WeightMapping(
                target_path="audio_tower.proj1.kernel", sharding=(None, None), transpose=True
            ),
            "thinker.audio_tower.proj1.bias": WeightMapping(
                target_path="audio_tower.proj1.bias", sharding=(None,), transpose=False
            ),
            "thinker.audio_tower.proj2.weight": WeightMapping(
                target_path="audio_tower.proj2.kernel", sharding=(None, None), transpose=True
            ),
            "thinker.audio_tower.proj2.bias": WeightMapping(
                target_path="audio_tower.proj2.bias", sharding=(None,), transpose=False
            ),
        }

        num_layers = int(self.audio_tower.config.encoder_layers)
        for layer_idx in range(num_layers):
            prefix = f"thinker.audio_tower.layers.{layer_idx}"
            target_prefix = f"audio_tower.layers.{layer_idx}"
            mappings.update(
                {
                    f"{prefix}.self_attn_layer_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn_layer_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn_layer_norm.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn_layer_norm.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.final_layer_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.final_layer_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.final_layer_norm.bias": WeightMapping(
                        target_path=f"{target_prefix}.final_layer_norm.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.out_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.out_proj.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.self_attn.out_proj.bias": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.out_proj.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.fc1.weight": WeightMapping(
                        target_path=f"{target_prefix}.fc1.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.fc1.bias": WeightMapping(
                        target_path=f"{target_prefix}.fc1.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.fc2.weight": WeightMapping(
                        target_path=f"{target_prefix}.fc2.kernel",
                        sharding=(None, None),
                        transpose=True,
                    ),
                    f"{prefix}.fc2.bias": WeightMapping(
                        target_path=f"{target_prefix}.fc2.bias",
                        sharding=(None,),
                        transpose=False,
                    ),
                }
            )
        return mappings

    def _create_text_weight_mappings(self) -> dict:
        mappings = {
            "thinker.model.embed_tokens.weight": WeightMapping(
                target_path="embed_tokens.embedding",
                sharding=("tensor", None),
                transpose=False,
            ),
            "thinker.model.norm.weight": WeightMapping(
                target_path="model.norm.scale", sharding=(None,), transpose=False
            ),
        }

        if not getattr(self.config, "tie_word_embeddings", False):
            mappings["thinker.lm_head.weight"] = WeightMapping(
                target_path="lm_head.embedding", sharding=("tensor", None), transpose=False
            )

        num_layers = self.config.num_hidden_layers
        for layer_idx in range(num_layers):
            prefix = f"thinker.model.layers.{layer_idx}"
            target_prefix = f"model.layers.{layer_idx}"
            mappings.update(
                {
                    f"{prefix}.input_layernorm.weight": WeightMapping(
                        target_path=f"{target_prefix}.input_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.post_attention_layernorm.weight": WeightMapping(
                        target_path=f"{target_prefix}.post_attention_layernorm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.q_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.k_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.v_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.v_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                        kv_head_padding=True,
                    ),
                    f"{prefix}.self_attn.o_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.o_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                        kv_head_padding=False,
                    ),
                    f"{prefix}.self_attn.q_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.q_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.self_attn.k_norm.weight": WeightMapping(
                        target_path=f"{target_prefix}.self_attn.k_norm.scale",
                        sharding=(None,),
                        transpose=False,
                    ),
                    f"{prefix}.mlp.gate_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.mlp.gate_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.up_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.mlp.up_proj.weight",
                        sharding=(None, "tensor"),
                        transpose=True,
                    ),
                    f"{prefix}.mlp.down_proj.weight": WeightMapping(
                        target_path=f"{target_prefix}.mlp.down_proj.weight",
                        sharding=("tensor", None),
                        transpose=True,
                    ),
                }
            )

            if getattr(self.config, "attention_bias", False):
                mappings.update(
                    {
                        f"{prefix}.self_attn.q_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.q_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.k_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.k_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.v_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.v_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                        f"{prefix}.self_attn.o_proj.bias": WeightMapping(
                            target_path=f"{target_prefix}.self_attn.o_proj.bias",
                            sharding=(None,),
                            transpose=False,
                        ),
                    }
                )

        return mappings

    def __call__(
        self,
        forward_batch: ForwardBatch,
        token_to_kv_pool: KVCache,
        logits_metadata: LogitsMetadata,
    ):
        hidden_states = self.embed_tokens(forward_batch.input_ids)

        if forward_batch.audio_features is not None and forward_batch.forward_mode.is_extend():
            audio_embeds, _ = self.audio_tower(
                forward_batch.audio_features,
                feature_attention_mask=forward_batch.audio_attention_mask,
            )
            audio_embeds = audio_embeds.astype(hidden_states.dtype)
            hidden_states = self._merge_audio(hidden_states, forward_batch, audio_embeds)

        hidden_states, aux_hidden_states, layers_kv_fused, layers_callback_flag = self.model(
            forward_batch, token_to_kv_pool, hidden_states
        )

        if not getattr(self.config, "tie_word_embeddings", False):
            output = self.logits_processor(
                hidden_states, self.lm_head, logits_metadata, aux_hidden_states=aux_hidden_states
            )
        else:
            output = self.logits_processor(
                hidden_states,
                self.embed_tokens,
                logits_metadata,
                aux_hidden_states=aux_hidden_states,
            )

        return output, layers_kv_fused, layers_callback_flag, None


EntryClass = Qwen3ASRForConditionalGeneration

from __future__ import annotations

from transformers import AutoConfig, PretrainedConfig


class Qwen3ASRAudioEncoderConfig(PretrainedConfig):
    model_type = "qwen3_asr_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        encoder_layers: int = 32,
        encoder_attention_heads: int = 20,
        encoder_ffn_dim: int = 5120,
        d_model: int = 1280,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        scale_embedding: bool = False,
        initializer_range: float = 0.02,
        max_source_positions: int = 1500,
        n_window: int = 100,
        output_dim: int = 3584,
        n_window_infer: int = 400,
        conv_chunksize: int = 500,
        downsample_hidden_size: int = 480,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.d_model = d_model
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.scale_embedding = scale_embedding
        self.initializer_range = initializer_range
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3ASRThinkerConfig(PretrainedConfig):
    model_type = "qwen3_asr_thinker"

    def __init__(
        self,
        audio_config=None,
        text_config=None,
        audio_token_id: int = 151646,
        audio_start_token_id: int = 151647,
        audio_end_token_id: int | None = None,
        user_token_id: int | None = None,
        initializer_range: float = 0.02,
        dtype: str | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.audio_token_id = audio_token_id
        self.audio_start_token_id = audio_start_token_id
        self.audio_end_token_id = audio_end_token_id
        self.user_token_id = user_token_id
        self.initializer_range = initializer_range
        self.dtype = dtype

        if isinstance(audio_config, dict):
            audio_config = Qwen3ASRAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3ASRAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            model_type = text_config.get("model_type", "qwen3")
            try:
                text_config = AutoConfig.for_model(model_type, **text_config)
            except Exception:
                text_config = PretrainedConfig(**text_config)
        elif text_config is None:
            text_config = AutoConfig.for_model("qwen3")
        self.text_config = text_config

    def get_text_config(self):
        return self.text_config


class Qwen3ASRConfig(PretrainedConfig):
    model_type = "qwen3_asr"
    sub_configs = {"thinker_config": Qwen3ASRThinkerConfig}

    def __init__(self, thinker_config=None, support_languages=None, **kwargs) -> None:
        super().__init__(**kwargs)
        if thinker_config is None:
            thinker_config = {}
        if isinstance(thinker_config, dict):
            thinker_config = Qwen3ASRThinkerConfig(**thinker_config)
        self.thinker_config = thinker_config
        self.support_languages = support_languages
        self._sync_text_config_attrs()

    def get_text_config(self, decoder: bool = False):
        return self.thinker_config.get_text_config()

    def _sync_text_config_attrs(self) -> None:
        text_config = getattr(self.thinker_config, "text_config", None)
        if text_config is None:
            return
        for key, val in text_config.__dict__.items():
            if not hasattr(self, key) and val is not None:
                setattr(self, key, val)


__all__ = ["Qwen3ASRConfig", "Qwen3ASRThinkerConfig", "Qwen3ASRAudioEncoderConfig"]

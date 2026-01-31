"""A tensor parallel worker."""

import itertools
import logging
import os
import signal
import threading
import time
from queue import Queue

import jax
import jax.numpy as jnp
import numpy as np
import psutil
from flax import nnx
from jax.experimental.multihost_utils import broadcast_one_to_all
from tqdm import tqdm

from sgl_jax.srt.configs.model_config import ModelConfig
from sgl_jax.srt.constrained.bitmask_ops import allocate_token_bitmask
from sgl_jax.srt.layers.logits_processor import LogitsMetadata, LogitsProcessorOutput
from sgl_jax.srt.layers.routed_experts_capturer import get_global_experts_capturer
from sgl_jax.srt.managers.schedule_batch import (
    ModelWorkerBatch,
    global_server_args_dict,
    _ASR_AUDIO_FRAME_PADDINGS,
    _ASR_AUDIO_HOP_LENGTH,
)
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.mem_cache.memory_pool import ReqToTokenPool
from sgl_jax.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sgl_jax.srt.model_executor.model_runner import MockModelRunner, ModelRunner
from sgl_jax.srt.sampling.sampling_batch_info import SamplingBatchInfo, SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import (
    PRECOMPILE_DEFAULT_BS_PADDINGS,
    PRECOMPILE_DEFAULT_TOKEN_PADDINGS,
    get_bool_env_var,
)
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)

PERF_BREAKDOWN = get_bool_env_var("SGLANG_PERF_BREAKDOWN")
PERF_BLOCK_UNTIL_READY = get_bool_env_var("SGLANG_PERF_BLOCK_UNTIL_READY")
PERF_LOG_EVERY = int(os.environ.get("SGLANG_PERF_LOG_EVERY", "1") or "1")
PERF_SLOW_MS = float(os.environ.get("SGLANG_PERF_SLOW_MS", "0") or "0")
LOG_CACHE_MISS = get_bool_env_var("SGLANG_LOG_CACHE_MISS")


def _asr_feat_extract_output_length(input_frames: int) -> int:
    """Match TokenizerManager._feat_extract_output_length (for warmup token lengths)."""
    input_len = int(input_frames)
    input_lengths_leave = input_len % 100
    feat_lengths = (input_lengths_leave - 1) // 2 + 1
    output_lengths = ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_len // 100) * 13
    return int(output_lengths)


class ModelWorker:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
        req_to_token_pool: ReqToTokenPool | None = None,
        is_draft_worker: bool = False,
    ):
        # Parse args
        self.tp_size = server_args.tp_size
        from sgl_jax.srt.speculative.spec_info import SpeculativeAlgorithm

        self.speculative_algorithm = SpeculativeAlgorithm.from_string(
            server_args.speculative_algorithm
        )
        self.server_args = server_args

        # LoRA configurations
        self.lora_paths = server_args.lora_paths
        self.max_loras_per_batch = server_args.max_loras_per_batch

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=(
                server_args.model_path
                if not is_draft_worker
                else server_args.speculative_draft_model_path
            ),
            model_revision=(
                server_args.revision
                if not is_draft_worker
                else server_args.speculative_draft_model_revision
            ),
            is_draft_model=is_draft_worker,
        )

        self.mesh = mesh
        self.page_size = server_args.page_size

        # need_prepare_lora_batch is False in overlap mode, default is True
        self.need_prepare_lora_batch = True

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        if server_args.random_seed is None:
            with jax.default_device(jax.local_devices()[0]):
                seed_to_broadcast = server_args.random_seed if jax.process_index() == 0 else 0
                self.random_seed = broadcast_one_to_all(seed_to_broadcast).item()
        else:
            self.random_seed = server_args.random_seed

        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.chunked_prefill_size = server_args.chunked_prefill_size

        # init model runner
        self.model_runner = ModelRunner(
            model_config=self.model_config,
            mem_fraction_static=server_args.mem_fraction_static,
            tp_size=server_args.tp_size,
            server_args=server_args,
            mesh=self.mesh,
            is_draft_worker=is_draft_worker,
            req_to_token_pool=req_to_token_pool,
            rngs=nnx.Rngs(self.random_seed),
            max_padding=max(self.max_prefill_tokens, self.chunked_prefill_size),
        )

        # set infer devices
        self.device = server_args.device

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens

        # Calculate max_running_requests from different constraints
        attn_backend_limit = self.model_runner.attn_backend.get_max_running_reqests(
            self.model_config.context_len, self.page_size
        )
        server_limit = (
            self.max_total_num_tokens // 2
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )
        pool_limit = self.model_runner.req_to_token_pool.size
        constraints = [server_limit, pool_limit, attn_backend_limit]
        self.max_running_requests = min(constraints)
        # Log each constraint for debugging
        logger.info("Max running requests constraints:")
        logger.info(
            "  - Server limit: %s %s",
            server_limit,
            (
                "(max_total_tokens//2)"
                if server_args.max_running_requests is None
                else "(configured)"
            ),
        )
        logger.info("  - Token pool size: %s", pool_limit)
        logger.info(
            "  - Attention backend: %s (context_len=%s, page_size=%s)",
            attn_backend_limit,
            self.model_config.context_len,
            self.page_size,
        )
        logger.info("  → Final max_running_requests: %s", self.max_running_requests)
        assert self.max_running_requests > 0, "max_running_request is zero"

        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert self.max_req_len > 0 and self.max_req_input_len > 0, "Memory pool size is too small"

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        # self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

        self.max_padded_batch_size, self.max_padded_num_tokens = self.get_max_padded_size()

        # precompile
        self.precompile_token_paddings = server_args.precompile_token_paddings

        # normalize server_args.precompile_token_paddings
        # ensure every token padding value is not less than max_runnig_requests
        self.normalize_token_paddings()

        bs_padding_list = (
            server_args.precompile_bs_paddings
            if server_args.precompile_bs_paddings is not None
            else PRECOMPILE_DEFAULT_BS_PADDINGS
        )
        self.precompile_bs_paddings = []
        for bs in bs_padding_list:
            if bs <= self.max_padded_batch_size and (
                server_args.moe_backend != "fused" or bs >= self.tp_size * 2
            ):
                self.precompile_bs_paddings.append(bs)
        self.precompile_bs_paddings.sort()
        if (
            len(self.precompile_bs_paddings) == 0
            or self.precompile_bs_paddings[-1] < self.max_padded_batch_size
        ):
            self.precompile_bs_paddings.append(self.max_padded_batch_size)

        # padding cache_loc_paddings
        # note: the length of following two cache_loc_paddings must keep the same to length of separate bs_paddings.
        self.precompile_cache_loc_paddings = [
            (item * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
            for item in self.precompile_bs_paddings
        ]

        self.parent_process = psutil.Process().parent()
        self.sync_queue = Queue()
        self.sync_expert_ids_d2h_thread = threading.Thread(
            target=self._sync_expert_ids_d2h_thread_func,
            daemon=True,
        )
        self.sync_expert_ids_d2h_thread.start()

    def _sync_expert_ids_d2h_thread_func(self):
        try:
            self._sync_experts_ids_d2h()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorker sync thread hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def _sync_experts_ids_d2h(self):
        while True:
            layers_topk_ids, model_worker_batch = self.sync_queue.get()
            get_global_experts_capturer().on_forward_end(layers_topk_ids, model_worker_batch)

    def normalize_token_paddings(self):
        normalized_token_paddings = []

        if self.precompile_token_paddings is None:
            self.precompile_token_paddings = PRECOMPILE_DEFAULT_TOKEN_PADDINGS
        for item in self.precompile_token_paddings:
            if item >= self.max_padded_batch_size and item <= self.max_padded_num_tokens:
                normalized_token_paddings.append(item)

        normalized_token_paddings.sort()
        if (
            len(normalized_token_paddings) == 0
            or normalized_token_paddings[-1] < self.max_padded_num_tokens
        ):
            normalized_token_paddings.append(self.max_padded_num_tokens)

        self.precompile_token_paddings = normalized_token_paddings

    def run_precompile(self, future_token_ids_map=None):
        self.precompile_extend(future_token_ids_map)
        self.precompile_asr_audio(future_token_ids_map)
        self.precompile_decode(future_token_ids_map)

    def precompile_extend(self, future_token_ids_map=None):
        start_time = time.perf_counter()
        logger.info(
            "[EXTEND] Begin to precompile bs_paddings=%s token_paddings=%s",
            self.precompile_bs_paddings[-1:],
            self.precompile_token_paddings,
        )

        bs, _ = self.get_max_padded_size()
        pairs = list(itertools.product([bs], self.precompile_token_paddings))

        with tqdm(pairs, desc="[EXTEND] PRECOMPILE", leave=False) as pbar:
            for pair in pbar:
                pair = list(pair)
                bs, num_tokens = pair[0], pair[1]
                pbar.set_postfix(bs=bs, tokens=num_tokens)
                if bs > num_tokens:
                    logger.warning("bs=%s > num_tokens=%s, skip this pair", bs, num_tokens)
                    continue
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    num_tokens,
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    enable_static_lora=self.server_args.enable_static_lora,
                )
                # Prepare LoRA batch if LoRA is enabled
                if self.server_args.enable_lora:
                    self.prepare_lora_batch(model_worker_batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch,
                    0,
                    self.mesh,
                    self.model_config.vocab_size,
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                        model_worker_batch.forward_batch.input_ids,
                        future_token_ids_map,
                    )

                self.forward_batch_generation(model_worker_batch, None, False, sampling_metadata)
        end_time = time.perf_counter()
        logger.info("[EXTEND] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_asr_audio(self, future_token_ids_map=None):
        """Precompile ASR audio path (Whisper log-mel + audio tower) if enabled.

        This avoids first-request compilation latency when using model-side feature extraction.
        """
        if not getattr(self.server_args, "asr_warmup_audio", False):
            return

        token_paddings = getattr(self.server_args, "asr_warmup_audio_token_paddings", None)
        if not token_paddings:
            # Default is a single, common bucket for the ASR prompt.
            token_paddings = [256]
        token_paddings = sorted(set(int(x) for x in token_paddings if int(x) > 0))

        frame_paddings = getattr(self.server_args, "asr_warmup_audio_frame_paddings", None)
        if not frame_paddings:
            # Match runtime bucketing by default.
            frame_paddings = list(_ASR_AUDIO_FRAME_PADDINGS)
        frame_paddings = sorted(set(int(x) for x in frame_paddings if int(x) > 0))

        hop = max(int(_ASR_AUDIO_HOP_LENGTH), 1)

        bs, _ = self.get_max_padded_size()
        start_time = time.perf_counter()
        logger.info(
            "[ASR_WARMUP] Begin to precompile audio path. bs=%d token_paddings=%s frame_paddings=%s",
            bs,
            token_paddings,
            frame_paddings,
        )

        pairs = list(itertools.product(token_paddings, frame_paddings))
        with tqdm(pairs, desc="[ASR_WARMUP] PRECOMPILE", leave=False) as pbar:
            for num_tokens, frames in pbar:
                pbar.set_postfix(tokens=num_tokens, frames=frames)
                if bs > num_tokens:
                    logger.warning(
                        "[ASR_WARMUP] bs=%s > num_tokens=%s, skip this pair", bs, num_tokens
                    )
                    continue

                audio_samples = int(frames * hop)
                # Keep a minimum length to avoid invalid reflect padding on very short clips.
                audio_samples = max(audio_samples, 400)

                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    int(num_tokens),
                    ForwardMode.EXTEND,
                    self.precompile_cache_loc_paddings[-1],
                    enable_static_lora=self.server_args.enable_static_lora,
                )

                # Provide dummy waveforms so model-side feature extraction executes.
                model_worker_batch.audio_waveforms = np.zeros(
                    (bs, audio_samples), dtype=np.float32
                )
                model_worker_batch.audio_waveform_lens = np.full(
                    (bs,), audio_samples, dtype=np.int32
                )
                model_worker_batch.audio_sampling_rates = np.full(
                    (bs,), 16000, dtype=np.int32
                )
                model_worker_batch.audio_token_start = np.zeros((bs,), dtype=np.int32)
                model_worker_batch.audio_token_len = np.full(
                    (bs,),
                    max(_asr_feat_extract_output_length(int(frames)), 1),
                    dtype=np.int32,
                )

                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch,
                    0,
                    self.mesh,
                    self.model_config.vocab_size,
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                        model_worker_batch.forward_batch.input_ids,
                        future_token_ids_map,
                    )

                # Only compile/execute the forward; sampling is already covered by the normal precompile.
                self.forward_batch_generation(model_worker_batch, None, True, sampling_metadata)

        end_time = time.perf_counter()
        logger.info("[ASR_WARMUP] Precompile finished in %.0f secs", end_time - start_time)

    def precompile_decode(self, future_token_ids_map=None):
        start_time = time.perf_counter()
        logger.info(
            "[DECODE] Begin to precompile bs_paddings=%s",
            self.precompile_bs_paddings,
        )

        with tqdm(self.precompile_bs_paddings, desc="[DECODE] PRECOMPILE", leave=False) as pbar:
            for bs in pbar:
                pbar.set_postfix(bs=bs)
                # use same page aligned with precompile cache_loc_paddings
                aligned_cache_loc_size = (
                    (bs * self.max_req_len + self.page_size - 1) // self.page_size * self.page_size
                )
                model_worker_batch = self.generate_model_worker_batch(
                    bs,
                    bs,
                    ForwardMode.DECODE,
                    aligned_cache_loc_size,
                    enable_static_lora=self.server_args.enable_static_lora,
                )
                # Prepare LoRA batch if LoRA is enabled
                if self.server_args.enable_lora:
                    self.prepare_lora_batch(model_worker_batch)
                sampling_metadata = SamplingMetadata.from_model_worker_batch(
                    model_worker_batch, 0, self.mesh, self.model_config.vocab_size
                )
                model_worker_batch.forward_batch = ForwardBatch.init_new(
                    model_worker_batch, self.model_runner
                )
                if future_token_ids_map is not None:
                    model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                        model_worker_batch.forward_batch.input_ids,
                        future_token_ids_map,
                    )
                _, next_token_ids, _ = self.forward_batch_generation(
                    model_worker_batch, None, False, sampling_metadata
                )
                if future_token_ids_map is not None:
                    set_future_token_ids(future_token_ids_map, 0, next_token_ids)

        end_time = time.perf_counter()
        logger.info("[DECODE] Precompile finished in %.0f secs", end_time - start_time)

    def set_forward_metadata(self, model_worker_batch: ModelWorkerBatch):
        self.model_runner.attn_backend.forward_metadata = (
            self.worker.model_runner.attn_backend.get_forward_metadata(model_worker_batch)
        )

    def get_max_padded_size(self):
        """Calculate the max padded batch size and token nums.

        Returns:
            tuple: (max_padded_batch_size, max_padded_num_tokens)
                - max_padded_batch_size: Maximum batch size, constrained by max_running_requests
                - max_padded_num_tokens: Maximum tokens, using chunked_prefill_size if enabled
        """
        # Use chunked prefill size if enabled (> 0), otherwise use max prefill tokens
        # Take minimum with max_prefill_tokens as upper bound
        max_padded_num_tokens = self.max_prefill_tokens
        if self.chunked_prefill_size > 0 and max_padded_num_tokens > self.chunked_prefill_size:
            max_padded_num_tokens = self.chunked_prefill_size

        # Batch size is constrained by both max_running_requests and available tokens divide by page_size
        max_padded_batch_size = min(self.max_running_requests, max_padded_num_tokens)

        return max_padded_batch_size, max_padded_num_tokens

    def get_precompile_paddings(self):
        return (
            self.precompile_token_paddings,
            self.precompile_bs_paddings,
            self.precompile_cache_loc_paddings,
        )

    def generate_model_worker_batch(
        self,
        bs: int,
        num_tokens: int,
        mode: ForwardMode,
        max_cache_loc_size: int,
        do_penalties: bool = False,
        speculative_algotithm=None,
        enable_static_lora: bool = None,
    ) -> ModelWorkerBatch:
        valid_input_ids = np.array([1] * bs, dtype=jnp.int32)
        invalid_input_ids = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        valid_out_cache_loc = np.arange(bs, dtype=jnp.int32)
        invalid_out_cache_loc = np.array([-1] * (num_tokens - bs), dtype=jnp.int32)
        valid_positions = np.array([0] * bs, dtype=jnp.int32)
        invalid_positions = np.array([0] * (num_tokens - bs), dtype=jnp.int32)
        invalid_cache_loc_size = max_cache_loc_size - bs
        if invalid_cache_loc_size < 0:
            raise ValueError(f"padding cache_loc_size {invalid_cache_loc_size} < 0!")

        valid_cache_loc = np.arange(bs, dtype=jnp.int32)
        invalid_cache_loc = np.array([0] * (invalid_cache_loc_size), dtype=jnp.int32)
        lora_ids = ["0"] * bs

        return ModelWorkerBatch(
            bid=1,
            forward_mode=mode,
            input_ids=np.concat([valid_input_ids, invalid_input_ids], axis=0),
            real_input_ids_len=len(valid_input_ids),
            real_bs=bs,
            req_pool_indices=np.arange(bs, dtype=np.int32),
            seq_lens=np.array([1] * bs, dtype=np.int32),
            out_cache_loc=np.concat([valid_out_cache_loc, invalid_out_cache_loc], axis=0),
            return_logprob=False,
            return_output_logprob_only=True,
            sampling_info=(
                SamplingBatchInfo.generate_for_precompile(bs, self.model_config.vocab_size)
                if speculative_algotithm is None
                else SamplingBatchInfo.generate_for_precompile_all_greedy(
                    bs, self.model_config.vocab_size
                )
            ),
            extend_input_logprob_token_ids=None,
            positions=np.concat([valid_positions, invalid_positions], axis=0),
            extend_start_loc=np.arange(bs, dtype=np.int32),
            cache_loc=np.concat([valid_cache_loc, invalid_cache_loc], axis=0),
            extend_prefix_lens=(
                np.array([0] * bs, dtype=np.int32) if mode == ForwardMode.EXTEND else None
            ),
            extend_seq_lens=(
                np.array([1] * bs, dtype=np.int32) if mode == ForwardMode.EXTEND else None
            ),
            top_logprobs_nums=None,
            token_ids_logprobs=None,
            extend_logprob_start_lens=None,
            capture_hidden_mode=(
                CaptureHiddenMode.FULL if self.server_args.multimodal else CaptureHiddenMode.NULL
            ),
            spec_algorithm=speculative_algotithm,
            lora_ids=lora_ids,  # Already set to [None] * bs above
            audio_features=None,
            audio_attention_mask=None,
            audio_token_start=None,
            audio_token_len=None,
        )

    def get_model_runner(self):
        return self.model_runner

    def prepare_lora_batch(self, model_worker_batch: ModelWorkerBatch):
        self.model_runner.lora_manager.prepare_lora_batch(model_worker_batch)
        if self.model_runner.lora_manager.has_new_weights:
            _, model_state = nnx.split(self.model_runner.model)
            self.model_runner.model_state_leaves, _ = jax.tree_util.tree_flatten(model_state)

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            self.device,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    @property
    def sliding_window_size(self) -> int | None:
        return self.model_runner.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.model_runner.is_hybrid

    def get_tokens_per_layer_info(self):
        return (
            self.model_runner.full_max_total_num_tokens,
            self.model_runner.swa_max_total_num_tokens,
        )

    def get_pad_input_ids_func(self):
        return getattr(self.model_runner.model, "pad_input_ids", None)

    def get_memory_pool(self):
        return (
            self.model_runner.req_to_token_pool,
            self.model_runner.token_to_kv_pool_allocator,
        )

    def _update_grammar_vocab_mask(
        self, batch: ModelWorkerBatch, sampling_metadata: SamplingMetadata
    ):
        if batch.sampling_info.grammars:
            # Overlap mode: wait for the mask prepared in set_next_batch_sampling_info_done
            if batch.sampling_info.sampling_info_done:
                batch.sampling_info.sampling_info_done.wait()
            else:
                batch.sampling_info.update_grammar_vocab_mask()
        if batch.sampling_info.vocab_mask is None:
            sampling_metadata.apply_vocab_mask = False
            sampling_metadata.vocab_mask = allocate_token_bitmask(
                len(batch.sampling_info.temperatures), batch.sampling_info.vocab_size
            )
        else:
            sampling_metadata.apply_vocab_mask = True
            sampling_metadata.vocab_mask = batch.sampling_info.vocab_mask

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        launch_done: threading.Event | None = None,
        skip_sample: bool = False,
        sampling_metadata: SamplingMetadata = None,
        forward_metadata=None,
    ) -> tuple[LogitsProcessorOutput | jax.Array | int, jax.Array | None]:
        perf_enabled = PERF_BREAKDOWN and (
            PERF_LOG_EVERY <= 1 or (model_worker_batch.bid % PERF_LOG_EVERY == 0)
        )
        perf_t0 = time.perf_counter() if perf_enabled else 0.0
        perf = {} if perf_enabled else None

        def _pmark(name: str, t0: float) -> None:
            if perf_enabled:
                perf[name] = time.perf_counter() - t0

        def _block_until_ready(x) -> None:
            if not PERF_BLOCK_UNTIL_READY:
                return
            if x is None:
                return
            try:
                x.block_until_ready()
            except Exception:
                return

        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora and self.need_prepare_lora_batch:
            t = time.perf_counter() if perf_enabled else 0.0
            self.prepare_lora_batch(model_worker_batch)
            _pmark("lora_prepare_s", t)

        # Use pre-initialized ForwardBatch if available (for overlap scheduling optimization)
        if model_worker_batch.forward_batch is not None:
            forward_batch = model_worker_batch.forward_batch
        else:
            t = time.perf_counter() if perf_enabled else 0.0
            forward_batch = ForwardBatch.init_new(model_worker_batch, self.model_runner)
            _pmark("forward_batch_init_s", t)

        if forward_metadata is None:
            t = time.perf_counter() if perf_enabled else 0.0
            forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
                model_worker_batch
            )
            _pmark("forward_metadata_s", t)

        if sampling_metadata is None:
            t = time.perf_counter() if perf_enabled else 0.0
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                self.model_config.vocab_size,
            )
            _pmark("sampling_metadata_s", t)

        self.model_runner.attn_backend.forward_metadata = forward_metadata
        t_forward = time.perf_counter() if perf_enabled else 0.0
        logits_output, forward_cache_miss, layers_topk_ids = self.model_runner.forward(
            forward_batch,
            logits_metadata=LogitsMetadata.from_model_worker_batch(model_worker_batch, self.mesh),
        )
        _pmark("model_forward_dispatch_s", t_forward)
        if perf_enabled:
            t = time.perf_counter()
            _block_until_ready(getattr(logits_output, "next_token_logits", None))
            _pmark("model_forward_block_s", t)

        self.dump_topk_ids(layers_topk_ids, model_worker_batch)

        if launch_done is not None:
            launch_done.set()

        self.sync_queue.put(
            (
                layers_topk_ids,
                model_worker_batch,
            )
        )

        # SAVE last layer logits
        save_logits_file_info = os.getenv("DUMP_LAST_LAYER_LOGITS_FILENAMES", None)
        if save_logits_file_info:
            save_logits_with_txt(
                logits_output.next_token_logits[: model_worker_batch.real_bs, :],
                save_logits_file_info,
                forward_batch.forward_mode,
            )

        if skip_sample:
            next_token_ids_device = None
            new_logits_output = None
            sample_cache_miss = 0
        else:
            import jax._src.test_util as jtu

            # Preprocess logits: update grammar vocab masks if needed
            if model_worker_batch.sampling_info:
                self._update_grammar_vocab_mask(model_worker_batch, sampling_metadata)

            with jtu.count_pjit_cpp_cache_miss() as count:
                t_sample = time.perf_counter() if perf_enabled else 0.0
                next_token_ids_device, token_logprobs, new_logits_output = self.model_runner.sample(
                    logits_output,
                    sampling_metadata,
                )
                _pmark("sample_dispatch_s", t_sample)
                if perf_enabled:
                    t = time.perf_counter()
                    _block_until_ready(next_token_ids_device)
                    _pmark("sample_block_s", t)
                sample_cache_miss = int(count() or 0)
            if model_worker_batch.return_output_logprob_only:
                t_logprob = time.perf_counter() if perf_enabled else 0.0
                logprobs = self.model_runner.compute_logprobs(token_logprobs, next_token_ids_device)
                logits_output.next_token_logprobs = logprobs[: model_worker_batch.real_bs]
                _pmark("compute_logprobs_s", t_logprob)

        cache_miss_count = int(forward_cache_miss or 0) + int(sample_cache_miss or 0)

        if LOG_CACHE_MISS and cache_miss_count > 0:
            try:
                dist = None
                if forward_metadata is not None and getattr(forward_metadata, "distribution", None) is not None:
                    dist = jax.device_get(forward_metadata.distribution).tolist()
                logger.info(
                    "[CACHE_MISS] mode=%s bid=%s real_bs=%d padded_bs=%d real_tokens=%d padded_tokens=%d "
                    "forward_miss=%d sample_miss=%d dist=%s",
                    str(model_worker_batch.forward_mode),
                    int(getattr(model_worker_batch, "bid", -1)),
                    int(model_worker_batch.real_bs or 0),
                    int(len(model_worker_batch.seq_lens) if model_worker_batch.seq_lens is not None else 0),
                    int(getattr(model_worker_batch, "real_input_ids_len", 0) or 0),
                    int(len(model_worker_batch.input_ids) if model_worker_batch.input_ids is not None else 0),
                    int(forward_cache_miss or 0),
                    int(sample_cache_miss or 0),
                    dist,
                )
            except Exception:
                logger.exception("Failed to log cache-miss breakdown")
        if new_logits_output is not None:
            logits_output = new_logits_output
            if logits_output.next_token_top_logprobs_val is not None:
                logits_output.next_token_top_logprobs_val = (
                    logits_output.next_token_top_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.next_token_top_logprobs_idx = (
                    logits_output.next_token_top_logprobs_idx.tolist()
                )
            if logits_output.next_token_token_ids_logprobs_val is not None:
                logits_output.next_token_token_ids_logprobs_val = (
                    logits_output.next_token_token_ids_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.next_token_token_ids_logprobs_idx = (
                    logits_output.next_token_token_ids_logprobs_idx.tolist()
                )
            if logits_output.input_token_ids_logprobs_val is not None:
                logits_output.input_token_ids_logprobs_val = (
                    logits_output.input_token_ids_logprobs_val.astype(jnp.float32).tolist()
                )
                logits_output.input_token_ids_logprobs_idx = (
                    logits_output.input_token_ids_logprobs_idx.tolist()
                )
            if logits_output.input_top_logprobs_val is not None:
                logits_output.input_top_logprobs_val = logits_output.input_top_logprobs_val.astype(
                    jnp.float32
                ).tolist()
                logits_output.input_top_logprobs_idx = logits_output.input_top_logprobs_idx.tolist()

        if perf_enabled:
            total_s = time.perf_counter() - perf_t0
            max_stage = ("", 0.0)
            if perf:
                max_stage = max(perf.items(), key=lambda kv: kv[1])

            def _shape(x):
                try:
                    return tuple(x.shape)
                except Exception:
                    return None

            audio_shape = _shape(getattr(forward_batch, "audio_waveforms", None))
            feat_shape = _shape(getattr(forward_batch, "audio_features", None))
            tok_shape = _shape(getattr(forward_batch, "input_ids", None))
            bs = int(getattr(model_worker_batch, "real_bs", 0) or 0)
            mode = getattr(model_worker_batch, "forward_mode", None)
            mode_str = str(mode) if mode is not None else "?"

            if PERF_SLOW_MS <= 0 or (total_s * 1000.0) >= PERF_SLOW_MS:
                # Keep this as a single line so it's easy to grep.
                breakdown = ", ".join(
                    f"{k}={v:.4f}s" for k, v in sorted(perf.items(), key=lambda kv: -kv[1])
                )
                logger.info(
                    "[PERF][worker] mode=%s bid=%s bs=%d tok=%s audio=%s feat=%s miss=%d total=%.4fs max=%s=%.4fs breakdown={%s}",
                    mode_str,
                    getattr(model_worker_batch, "bid", None),
                    bs,
                    tok_shape,
                    audio_shape,
                    feat_shape,
                    int(cache_miss_count or 0),
                    total_s,
                    max_stage[0],
                    max_stage[1],
                    breakdown,
                )

        return (
            logits_output,
            next_token_ids_device,
            cache_miss_count,
        )

    def dump_topk_ids(self, layers_topk_ids: list[jax.Array], model_worker_batch: ModelWorkerBatch):
        enable = self.server_args.enable_return_routed_experts
        dump_topk_ids_file_info = os.getenv("DUMP_TOPK_IDS_FILEINFO", None)
        if not enable or dump_topk_ids_file_info is None:
            return

        # format: {prefill_file_name},{decode_file_name}
        file_slice = dump_topk_ids_file_info.split(",")
        if model_worker_batch.forward_mode.is_extend():
            file_name = file_slice[0]
        elif model_worker_batch.forward_mode.is_decode():
            file_name = file_slice[1]
        else:
            raise ValueError(
                f"Unsupported {model_worker_batch.forward_mode} to save topk_ids with txt"
            )
        import datetime

        unpadded_input_len = model_worker_batch.get_original_input_len()
        layers_topk_ids_cpu = jax.device_get(layers_topk_ids)

        file_name = (
            f"{datetime.datetime.now().strftime('%Y-%m-%d_%H_%M_%S')}_{unpadded_input_len}"
            + file_name
        )

        valid_topk_ids = []
        for ids_cpu in layers_topk_ids_cpu:
            valid_ids = ids_cpu[
                :unpadded_input_len, : self.model_config.hf_text_config.num_experts_per_tok
            ]
            valid_topk_ids.append(valid_ids)

        # Stack to create (num_layers, seq_len, num_experts_per_tok)
        valid_topk_ids_stacked = np.stack(valid_topk_ids, axis=0)

        # Transpose to (seq_len, num_layers, num_experts_per_tok)
        seq_layer_topk_cpu = np.transpose(valid_topk_ids_stacked, (1, 0, 2))

        # os.makedirs(os.path.dirname(file_name), exist_ok=True)
        np.savetxt(file_name, np.asarray(seq_layer_topk_cpu).flatten(), fmt="%d")


class MockModelWorker:
    """A mock tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
    ):
        # Parse args
        self.tp_size = server_args.tp_size

        # Init model and tokenizer
        self.model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
        )

        # Sync random seed across TP workers
        # Each process may have different random_seed. After broadcast, all processes will have the same random_seed.
        self.random_seed = broadcast_one_to_all(server_args.random_seed).item()

        # init model runner
        self.model_runner = MockModelRunner(
            model_config=self.model_config,
            rngs=jax.random.PRNGKey(self.random_seed),
            server_args=server_args,
        )

        # Profile number of tokens
        self.max_total_num_tokens = self.model_runner.max_total_num_tokens
        self.max_prefill_tokens = server_args.max_prefill_tokens
        self.max_running_requests = min(
            (
                self.max_total_num_tokens // 2
                if server_args.max_running_requests is None
                else server_args.max_running_requests
            ),
            self.model_runner.req_to_token_pool.size,
        )
        assert self.max_running_requests > 0, "max_running_request is zero"
        self.max_req_len = min(
            self.model_config.context_len - 1,
            self.max_total_num_tokens - 1,
        )
        self.max_req_input_len = self.max_req_len - 5
        assert self.max_req_len > 0 and self.max_req_input_len > 0, "Memory pool size is too small"

        # A reference make this class has the same member as TpModelWorkerClient
        self.worker = self

    def get_worker_info(self):
        return (
            self.max_total_num_tokens,
            self.max_prefill_tokens,
            self.max_running_requests,
            self.max_req_len,
            self.max_req_input_len,
            self.random_seed,
            global_server_args_dict,
            self.model_runner.req_to_token_pool.size,
            self.model_runner.req_to_token_pool.max_context_len,
            self.model_runner.token_to_kv_pool.size,
        )

    def get_memory_pool(self):
        return (self.model_runner.req_to_token_pool, self.model_runner.token_to_kv_pool)

    def forward_batch_generation(
        self,
        _model_worker_batch: ModelWorkerBatch,
        _launch_done: threading.Event | None = None,
        _skip_sample: bool = False,
        _sampling_metadata: SamplingMetadata | None = None,
    ) -> tuple[LogitsProcessorOutput | jax.Array, jax.Array | None]:
        return (
            LogitsProcessorOutput(
                next_token_logits=jnp.array([0, 1]),
            ),
            None,
        )


def save_logits_with_txt(
    arr: jax.Array,
    file_info: str,
    forward_mode: ForwardMode,
):
    # format: {prefill_file_name},{decode_file_name}
    file_slice = file_info.split(",")
    if forward_mode.is_extend():
        file_name = file_slice[0]
    elif forward_mode.is_decode():
        file_name = file_slice[1]
    else:
        raise ValueError(f"Unsupported {forward_mode} to save logits with txt")

    os.makedirs(os.path.dirname(file_name), exist_ok=True)
    np.savetxt(file_name, np.asarray(jax.device_get(arr)).flatten(), fmt="%.15f")

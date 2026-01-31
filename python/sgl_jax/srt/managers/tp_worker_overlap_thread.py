"""A tensor parallel worker."""

import dataclasses
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
from jax.sharding import NamedSharding, PartitionSpec

from sgl_jax.srt.managers.schedule_batch import ModelWorkerBatch
from sgl_jax.srt.managers.tp_worker import ModelWorker
from sgl_jax.srt.managers.utils import resolve_future_token_ids, set_future_token_ids
from sgl_jax.srt.model_executor.forward_batch_info import ForwardBatch
from sgl_jax.srt.sampling.sampling_batch_info import SamplingMetadata
from sgl_jax.srt.server_args import ServerArgs
from sgl_jax.srt.utils.common_utils import get_bool_env_var
from sgl_jax.utils import get_exception_traceback

logger = logging.getLogger(__name__)

PERF_BREAKDOWN = get_bool_env_var("SGLANG_PERF_BREAKDOWN")
PERF_BLOCK_UNTIL_READY = get_bool_env_var("SGLANG_PERF_BLOCK_UNTIL_READY")
PERF_LOG_EVERY = int(os.environ.get("SGLANG_PERF_LOG_EVERY", "1") or "1")
PERF_SLOW_MS = float(os.environ.get("SGLANG_PERF_SLOW_MS", "0") or "0")


class ModelWorkerClient:
    """A tensor parallel model worker."""

    def __init__(
        self,
        server_args: ServerArgs,
        mesh: jax.sharding.Mesh,
    ):
        # Load the model
        self.worker = ModelWorker(server_args, mesh=mesh)
        # overlap mode set worker need_prepare_lora_batch to False
        self.worker.need_prepare_lora_batch = False

        self.max_running_requests = self.worker.max_running_requests
        self.device = self.worker.device

        # Init future mappings
        self.future_token_ids_ct = 0
        self.future_token_ids_limit = self.max_running_requests * 3
        self.future_token_ids_map = jnp.zeros((self.max_running_requests * 5,), dtype=jnp.int32)
        self.mesh = mesh
        sharding = NamedSharding(mesh, PartitionSpec(None))
        self.future_token_ids_map = jax.device_put(self.future_token_ids_map, sharding)
        # Launch threads
        self.input_queue = Queue()
        self.output_queue = Queue()
        # Exposed for scheduler-side metrics/debugging.
        self.last_exec_time_s = 0.0
        self.last_forward_mode = None
        self.last_real_bs = 0
        self.last_bid = -1
        self.last_output_queue_wait_s = 0.0
        self.last_device_get_s = 0.0
        self.last_launch_wait_s = 0.0
        # JAX handles device execution automatically, no need for explicit streams
        self.forward_thread = threading.Thread(
            target=self.forward_thread_func,
            daemon=bool(server_args.enable_single_process),
        )
        self.forward_thread.start()
        self.parent_process = psutil.Process().parent()

    def get_model_runner(self):
        return self.worker.get_model_runner()

    def get_worker_info(self):
        return self.worker.get_worker_info()

    def get_pad_input_ids_func(self):
        return self.worker.get_pad_input_ids_func()

    def get_memory_pool(self):
        return (
            self.worker.model_runner.req_to_token_pool,
            self.worker.model_runner.token_to_kv_pool_allocator,
        )

    def get_kv_cache(self):
        return self.worker.model_runner.token_to_kv_pool

    def get_max_padded_size(self):
        return self.worker.get_max_padded_size()

    def get_precompile_paddings(self):
        return self.worker.get_precompile_paddings()

    def forward_thread_func(self):
        try:
            self.forward_thread_func_()
        except Exception:
            traceback = get_exception_traceback()
            logger.error("ModelWorkerClient hit an exception: %s", traceback)
            self.parent_process.send_signal(signal.SIGQUIT)

    def forward_thread_func_(self):
        while True:
            (
                model_worker_batch,
                future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            ) = self.input_queue.get()
            if not model_worker_batch:
                break

            # Resolve future tokens in the input
            input_ids = model_worker_batch.forward_batch.input_ids
            model_worker_batch.forward_batch.input_ids = resolve_future_token_ids(
                input_ids, self.future_token_ids_map
            )

            # Run forward
            t_exec0 = time.perf_counter()
            with jax.profiler.TraceAnnotation(f"forward_batch_generation {model_worker_batch.bid}"):
                logits_output, next_token_ids, cache_miss_count = (
                    self.worker.forward_batch_generation(
                        model_worker_batch,
                        model_worker_batch.launch_done,
                        sampling_metadata=sampling_metadata,
                        forward_metadata=forward_metadata,
                    )
                )
            if PERF_BLOCK_UNTIL_READY:
                # In JAX, dispatch is async. This makes exec_time_s reflect true device time,
                # at the cost of overlap (debug/perf profiling only).
                jax.block_until_ready(next_token_ids)
            exec_time_s = time.perf_counter() - t_exec0

            # Update the future token ids map
            self.future_token_ids_map = set_future_token_ids(
                self.future_token_ids_map,
                future_token_ids_ct,
                next_token_ids,
            )
            self.output_queue.put(
                (
                    model_worker_batch.bid,
                    model_worker_batch.forward_mode,
                    int(model_worker_batch.real_bs or 0),
                    float(exec_time_s),
                    logits_output,
                    next_token_ids,
                    cache_miss_count,
                )
            )

    def resolve_last_batch_result(self, launch_done: threading.Event | None = None):
        """
        This function is called to resolve the last batch result and
        wait for the current batch to be launched. Used in overlap mode.
        """
        t0 = time.perf_counter()
        (
            bid,
            forward_mode,
            real_bs,
            exec_time_s,
            logits_output,
            next_token_ids,
            cache_miss_count,
        ) = self.output_queue.get()
        output_queue_wait_s = time.perf_counter() - t0
        perf_enabled = PERF_BREAKDOWN and (PERF_LOG_EVERY <= 1 or (bid % PERF_LOG_EVERY == 0))
        perf_t0 = t0 if perf_enabled else 0.0
        perf = {} if perf_enabled else None

        def _pmark(name: str, t0: float) -> None:
            if perf_enabled:
                perf[name] = time.perf_counter() - t0

        if perf_enabled:
            perf["output_queue_wait_s"] = output_queue_wait_s

        # Expose last batch timing for scheduler-side metrics (decode-only throughput).
        self.last_exec_time_s = float(exec_time_s)
        self.last_forward_mode = forward_mode
        self.last_real_bs = int(real_bs)
        self.last_bid = bid
        self.last_output_queue_wait_s = float(output_queue_wait_s)

        t_device_get0 = time.perf_counter()
        if logits_output.next_token_logprobs is not None:
            logits_output.next_token_logprobs = jax.device_get(
                logits_output.next_token_logprobs
            ).tolist()
        if logits_output.input_token_logprobs is not None:
            logits_output.input_token_logprobs = jax.device_get(
                logits_output.input_token_logprobs
            ).tolist()
        if logits_output.hidden_states is not None:
            logits_output.hidden_states = jax.device_get(logits_output.hidden_states)
        next_token_ids = jax.device_get(next_token_ids).tolist()
        device_get_s = time.perf_counter() - t_device_get0
        self.last_device_get_s = float(device_get_s)
        if perf_enabled:
            perf["device_get_s"] = device_get_s

        if launch_done is not None:
            t_launch0 = time.perf_counter()
            launch_done.wait()
            launch_wait_s = time.perf_counter() - t_launch0
            self.last_launch_wait_s = float(launch_wait_s)
            if perf_enabled:
                perf["launch_wait_s"] = launch_wait_s

        if perf_enabled:
            total_s = time.perf_counter() - perf_t0
            max_stage = ("", 0.0)
            if perf:
                max_stage = max(perf.items(), key=lambda kv: kv[1])
            if PERF_SLOW_MS <= 0 or (total_s * 1000.0) >= PERF_SLOW_MS:
                breakdown = ", ".join(
                    f"{k}={v:.4f}s" for k, v in sorted(perf.items(), key=lambda kv: -kv[1])
                )
                logger.info(
                    "[PERF][overlap_result] bid=%s miss=%d total=%.4fs max=%s=%.4fs breakdown={%s}",
                    bid,
                    int(cache_miss_count or 0),
                    total_s,
                    max_stage[0],
                    max_stage[1],
                    breakdown,
                )

        return logits_output, next_token_ids, cache_miss_count

    def forward_batch_generation(
        self,
        model_worker_batch: ModelWorkerBatch,
        sampling_metadata: SamplingMetadata = None,
    ) -> tuple[None, jax.Array, int]:
        perf_enabled = PERF_BREAKDOWN and (
            PERF_LOG_EVERY <= 1 or (model_worker_batch.bid % PERF_LOG_EVERY == 0)
        )
        perf_t0 = time.perf_counter() if perf_enabled else 0.0
        perf = {} if perf_enabled else None

        def _pmark(name: str, t0: float) -> None:
            if perf_enabled:
                perf[name] = time.perf_counter() - t0

        # Create a new copy of sampling_info because it will be updated in-place by the scheduler for the next batch.
        sampling_info = model_worker_batch.sampling_info
        t = time.perf_counter() if perf_enabled else 0.0
        sampling_info.update_penalties()
        model_worker_batch.sampling_info = self.cur_sampling_info = dataclasses.replace(
            sampling_info,
            sampling_info_done=threading.Event(),
            penalizer_orchestrator=None,
        )
        _pmark("sampling_info_copy_s", t)

        if sampling_metadata is None:
            t = time.perf_counter() if perf_enabled else 0.0
            sampling_metadata = SamplingMetadata.from_model_worker_batch(
                model_worker_batch,
                len(model_worker_batch.seq_lens) - model_worker_batch.real_bs,
                self.mesh,
                self.worker.model_config.vocab_size,
            )
            _pmark("sampling_metadata_s", t)

        t = time.perf_counter() if perf_enabled else 0.0
        forward_metadata = self.worker.model_runner.attn_backend.get_forward_metadata(
            model_worker_batch
        )
        _pmark("forward_metadata_s", t)

        # Prepare LoRA batch if LoRA is enabled
        if self.worker.server_args.enable_lora:
            t = time.perf_counter() if perf_enabled else 0.0
            self.worker.prepare_lora_batch(model_worker_batch)
            _pmark("lora_prepare_s", t)

        t = time.perf_counter() if perf_enabled else 0.0
        model_worker_batch.forward_batch = ForwardBatch.init_new(
            model_worker_batch, self.worker.get_model_runner()
        )
        _pmark("forward_batch_init_s", t)

        # Push a new batch to the queue (JAX handles synchronization automatically)
        t = time.perf_counter() if perf_enabled else 0.0
        self.input_queue.put(
            (
                model_worker_batch,
                self.future_token_ids_ct,
                sampling_metadata,
                forward_metadata,
            )
        )
        _pmark("queue_put_s", t)

        # Allocate output future objects
        bs = len([seq_len for seq_len in model_worker_batch.seq_lens if seq_len > 0])

        future_next_token_ids = np.arange(
            -(self.future_token_ids_ct + 1),
            -(self.future_token_ids_ct + 1 + bs),
            -1,
            dtype=np.int32,
        )
        self.future_token_ids_ct = (self.future_token_ids_ct + bs) % self.future_token_ids_limit

        if perf_enabled:
            total_s = time.perf_counter() - perf_t0
            max_stage = ("", 0.0)
            if perf:
                max_stage = max(perf.items(), key=lambda kv: kv[1])
            if PERF_SLOW_MS <= 0 or (total_s * 1000.0) >= PERF_SLOW_MS:
                breakdown = ", ".join(
                    f"{k}={v:.4f}s" for k, v in sorted(perf.items(), key=lambda kv: -kv[1])
                )
                logger.info(
                    "[PERF][overlap_submit] mode=%s bid=%s bs=%d total=%.4fs max=%s=%.4fs breakdown={%s}",
                    str(model_worker_batch.forward_mode),
                    model_worker_batch.bid,
                    int(model_worker_batch.real_bs or 0),
                    total_s,
                    max_stage[0],
                    max_stage[1],
                    breakdown,
                )

        return None, future_next_token_ids, 0

    def run_precompile(self):
        self.worker.run_precompile(self.future_token_ids_map)

    @property
    def sliding_window_size(self) -> int | None:
        return self.worker.sliding_window_size

    @property
    def is_hybrid(self) -> bool:
        return self.worker.is_hybrid

    def get_tokens_per_layer_info(self):
        return self.worker.get_tokens_per_layer_info()

    def __delete__(self):
        self.input_queue.put((None, None, None, None))

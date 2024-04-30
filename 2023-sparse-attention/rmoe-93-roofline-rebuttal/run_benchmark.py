import json
import time
from dataclasses import asdict, dataclass
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM

import llminference as L
from llminference.eval_adapter import Adapter


@dataclass
class Config:
    model_name: str
    batch_size: int
    seq_len: int
    bytes_per_param: float
    bytes_per_kv: float = 2


# Utility functions for measuring the attention op execution time
def measure_time(times: List[float]):
    def decorator(func: Callable):
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            torch.cuda.synchronize()
            t0 = time.time()
            out = func(*args, **kwargs)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
            return out

        return wrapper

    return decorator


def convert_llama(model: LlamaForCausalLM, times: List[float]) -> LlamaForCausalLM:
    def _replace(m: nn.Module) -> Optional[nn.Module]:
        if isinstance(m, LlamaAttention):
            module = L.models.llama_attention.LlamaAttention(model.config)
            module._attn = measure_time(times)(module._attn)
            return module

    model = L.utility.convert_module(model, _replace)
    return model


def get_model_config(model_name: str) -> Dict[str, Any]:
    config = AutoConfig.from_pretrained(model_name)
    try:
        kv_group_size = config.num_attention_heads // config.num_key_value_heads
    except AttributeError:
        kv_group_size = 1

    return dict(
        hidden_dim=config.hidden_size,
        kv_group_size=kv_group_size,
        n_layers=config.num_hidden_layers,
        vocab_size=config.vocab_size,
    )


def measure_generation_step(config: Config) -> Tuple[float, float]:
    assert torch.cuda.is_available(), "No CUDA device available"

    if config.bytes_per_param == 2:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, torch_dtype=torch.float16
        )
        model.to(device="cuda")
    elif config.bytes_per_param == 1:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, load_in_8bit=True
        )
    elif config.bytes_per_param == 0.5:
        model = AutoModelForCausalLM.from_pretrained(
            config.model_name, load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16
        )
    else:
        print(f"{config.bytes_per_param} is invalid")

    # Add wrapper for measuring attn execution time
    attn_times = []
    model = convert_llama(model, attn_times)

    # Generate random prefill cache
    hidden_dim = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    try:
        n_kv_heads = model.config.num_key_value_heads
    except AttributeError:
        n_kv_heads = n_heads
    head_dim = hidden_dim // n_heads

    past_key_values = Adapter._kv_to_tuple(
        torch.randn(
            (n_layers, 2, config.batch_size, n_kv_heads, config.seq_len, head_dim),
            dtype=torch.float16,
            device=model.device,
        )
    )

    torch.cuda.synchronize()
    t0 = time.time()
    _ = model(
        input_ids=torch.zeros(
            config.batch_size, 1, dtype=torch.int, device=model.device
        ),
        past_key_values=past_key_values,
    )
    torch.cuda.synchronize()
    t = time.time() - t0
    attn_t = sum(attn_times)
    return (t, attn_t)


if __name__ == "__main__":
    # Number of times to repeat each experiment
    n_reps = 1

    models = ["meta-llama/Llama-2-7b-hf"]
    batch_sizes = [1]
    seq_lens = [1]
    bytes_per_params = [2]
    configs = [
        Config(model_name, batch_size, seq_len, bytes_per_param)
        for model_name in models
        for batch_size in batch_sizes
        for seq_len in seq_lens
        for bytes_per_param in bytes_per_params
    ]

    with Path("measurements.jsonl").open("w") as f:
        for config in configs:
            # Check for out of memory error
            error: Optional[str] = None

            try:
                # Ignore first measurement (takes longer)
                times = [measure_generation_step(config) for _ in range(n_reps + 1)][1:]
            except torch.cuda.OutOfMemoryError as e:
                times = []
                error = repr(e)

            out = dict(
                **asdict(config),
                **get_model_config(config.model_name),
                times=times,
                error=error,
                device_name=torch.cuda.get_device_name(),
            )
            f.write(json.dumps(out) + "\n")
            f.flush()

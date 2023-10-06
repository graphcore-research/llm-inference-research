"""High-level experiment interface, supporting multiple tasks, models & logging."""

import dataclasses
import datetime
import logging
import multiprocessing
import multiprocessing.pool
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, cast

import torch
import tqdm
import wandb
from transformers import PreTrainedModel
from transformers.models.gpt_neox.modeling_gpt_neox import GPTNeoXForCausalLM

from . import (
    eval_adapter,
    eviction_attention,
    qa,
    sparse_attention,
    summarisation,
    utility,
)

# A dictionary of code changes that may affect the numbers, which is always
# logged alongside experiment results.
# E.g. {"evaluate-regex-permissive-newlines": True}
CODE_CHANGES: Dict[str, Any] = dict()

WANDB_PROJECT = "sparse-attention"
WANDB_URL = "https://wandb.sourcevertex.net"
logger = logging.getLogger(__name__)


# Configuration


TASKS = ["triviaqa", "squad", "cnn_dailymail"]


@dataclass
class Task:
    name: str  # TASKS
    shots: int
    samples: int


@dataclass
class Sparsity:
    name: str  # method name from SparsityMethods
    # additional keys are passed to the sparsity method

    def __init__(self, name: str, **kwargs: Any):
        self.name = name
        self.__dict__.update(kwargs)


@dataclass
class Execution:
    device: str
    dtype: str
    batch_size: int
    wandb: Union[bool, str]  # False | True | "offline"

    @classmethod
    def auto(cls, batch_size: Optional[int] = None) -> "Execution":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(
            device=device,
            dtype=dict(cpu="float32", cuda="float16")[device],
            batch_size=batch_size or dict(cpu=10, cuda=5)[device],
            wandb=True,
        )


@dataclass
class Experiment:
    name: str
    task: Task
    model: str  # e.g. EleutherAI/pythia-1b
    sparsity: Sparsity
    execution: Execution
    # This shouldn't be specified directly, and is set via `CODE_CHANGES`
    code_changes: Dict[str, Any] = dataclasses.field(
        default_factory=lambda: CODE_CHANGES.copy()
    )

    def to_dict(self) -> Dict[str, Any]:
        def convert(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            if hasattr(obj, "__dict__"):
                return convert(obj.__dict__)
            return obj

        return cast(Dict[str, Any], convert(self))


Outcome = Dict[str, Any]

# Method


class SparsityMethods:
    @classmethod
    def apply(cls, sparsity: Sparsity, model: PreTrainedModel) -> PreTrainedModel:
        method = getattr(cls, sparsity.name)
        return method(
            model, **{k: v for k, v in sparsity.__dict__.items() if k != "name"}
        )

    @staticmethod
    def dense(model: PreTrainedModel) -> PreTrainedModel:
        return model

    @staticmethod
    def sparse_v(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        model.generation_context = eval_adapter.patch_for_model(
            "torch.nn.functional.softmax",
            sparse_attention.sparse_softmax_fixed_k,
            **settings,
        )
        return model

    @staticmethod
    def eviction(model: PreTrainedModel, **settings: Any) -> PreTrainedModel:
        assert isinstance(model, GPTNeoXForCausalLM)
        return eviction_attention.convert_gptneox(
            model, eviction_attention.Settings(**settings)
        )


# Running


def _evaluate(
    task: Task, adapter: eval_adapter.Adapter, batch_size: int
) -> Dict[str, Any]:
    if task.name in ["triviaqa", "squad"]:
        data = (
            qa.TriviaQA.data(context="wiki")
            if task.name == "triviaqa"
            else qa.SQuAD.data()
        )
        examples = [
            qa.add_few_shot_prompt(
                data[i],
                k=task.shots,
                prompt_template=qa.get_default_prompt_template(
                    adapter.model.config._name_or_path, task.shots
                ),
            )
            for i in range(task.samples)
        ]
        evaluate_fn: Any = qa.evaluate
    if task.name == "cnn_dailymail":
        assert task.shots == 0
        data = summarisation.CnnDailymail.data()
        examples = [data[i] for i in range(task.samples)]
        evaluate_fn = summarisation.evaluate

    results = list(
        evaluate_fn(adapter=adapter, examples=examples, batch_size=batch_size)
    )
    return dict(
        results=results,
        count=len(results),
        **{
            k: sum(x[k] for x in results) / len(results)
            for k in ["prefill_length", "reference_length", "match", "rougeL"]
            if k in results[0]
        },
    )


def run_one(xp: Experiment) -> Outcome:
    """Run a single experiment, optionally logging to wandb."""
    if xp.execution.wandb:
        mode = "offline" if xp.execution.wandb == "offline" else "online"
        if mode == "online" and wandb.api.api_url != WANDB_URL:
            mode = "offline"
            logger.warning(
                f"Wandb not logged in to {WANDB_URL}; running in offline mode"
            )
        wandb.init(
            config=xp.to_dict(),
            mode=mode,
            entity="research",
            project=WANDB_PROJECT,
            reinit=True,
        )
    adapter = eval_adapter.Adapter.from_pretrained(
        xp.model, dtype=getattr(torch, xp.execution.dtype)
    )
    adapter.model = SparsityMethods.apply(xp.sparsity, adapter.model)
    adapter.model.to(torch.device(xp.execution.device))
    out = {}
    out["parameters"] = sum(p.nelement() for p in adapter.model.parameters())
    out["model_config"] = adapter.model.config.to_diff_dict()
    t0 = time.time()
    try:
        out.update(_evaluate(xp.task, adapter, xp.execution.batch_size))
    except Exception as error:
        out["error"] = repr(error)
        out["backtrace"] = traceback.format_exc()
    out["duration"] = time.time() - t0
    if xp.execution.wandb:
        wandb.summary.update(out)
        assert wandb.run is not None
        out["wandb"] = dict(
            id=wandb.run.id, name=wandb.run.name, url=wandb.run.get_url()
        )
        wandb.finish(exit_code=1 if "error" in out else 0)
    return dict(**xp.to_dict(), **out)


def _run_many_task(xp: Experiment) -> Dict[str, Any]:
    """Run an experiment in a non-daemonic subprocess (for sake of wandb)."""
    queue: "multiprocessing.Queue[Dict[str, Any]]" = multiprocessing.Queue(1)

    def _run() -> None:
        torch.set_num_threads(32)
        queue.put(run_one(xp))

    p = multiprocessing.Process(target=_run, daemon=False)
    p.start()
    p.join()
    return queue.get()


def run_many(
    xps: List[Experiment], n_workers: int = 1, out: Optional[Path] = None
) -> None:
    """Run multiple experiments, optionally as a multiprocess sweep."""
    if out is None:
        out = (
            Path("out") / datetime.datetime.now().isoformat(sep="/", timespec="seconds")
        ).with_suffix(".jsonl")

    with utility.jsonlines_writer(out) as writer:
        if n_workers >= 2:
            with multiprocessing.pool.ThreadPool(n_workers) as pool:
                for result in tqdm.tqdm(
                    pool.map(_run_many_task, xps), total=len(xps), desc="experiments"
                ):
                    writer(result)
        else:
            for xp in tqdm.tqdm(xps, desc="experiments"):
                writer(run_one(xp))

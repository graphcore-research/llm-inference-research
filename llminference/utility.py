"""Generic utilities"""

import datetime
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, TypeVar, cast

import torch
import tqdm
from torch import multiprocessing

T = TypeVar("T")


def batches(iterable: Iterable[T], batch_size: int) -> Iterable[List[T]]:
    """Chunks `iterable` into batches of consecutive values."""
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


AnyDict = Dict[str, Any]


def _sweep_runner(
    task: Callable[..., AnyDict], task_args: AnyDict, n_threads: int
) -> Dict[str, Any]:
    sys.stdout = open(os.devnull, "w")
    sys.stderr = open(os.devnull, "w")
    torch.set_num_threads(n_threads)
    t0 = time.time()
    try:
        result = task(**task_args)
        result["_duration"] = time.time() - t0
        return result
    except Exception as error:
        return dict(
            _args=task_args,
            _error=repr(error),
            _error_tb=traceback.format_exception(
                type(error), error, error.__traceback__
            ),
            _duration=time.time() - t0,
        )


def run_multiprocess_sweep(
    task: Callable[..., AnyDict],
    settings: List[Dict[str, Any]],
    dest: Path,
    n_workers: int,
    max_threads_per_worker: int = 32,
) -> None:
    """Run a sweep in worker processes, saving the results to a .jsonl file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        time = datetime.datetime.now().isoformat(timespec="seconds")
        dest = dest.parent / f"{dest.stem}-{time}{dest.suffix}"

    print(f"Sweeping {len(settings)} settings -> {dest}", file=sys.stderr)

    n_error = 0
    n_threads = min(max_threads_per_worker, cast(int, os.cpu_count()) // n_workers)
    with multiprocessing.Pool(n_workers, maxtasksperchild=1) as pool, dest.open(
        "w"
    ) as destfile:
        results = tqdm.tqdm(
            [
                pool.apply_async(
                    _sweep_runner,
                    kwds=dict(task=task, task_args=s, n_threads=n_threads),
                )
                for s in settings
            ],
            ncols=120,
            miniters=1,
        )
        for n, result in enumerate(results):
            out = result.get()
            print(json.dumps(out), file=destfile, flush=True)
            n_error += "_error" in out
            if n_error:
                results.set_description(f"{n_error}/{n+1} failed")
    print(
        f"Finished sweep ({n_error}/{n+1} failed) -> {dest}",
        file=sys.stderr,
    )

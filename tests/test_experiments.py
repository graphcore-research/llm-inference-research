# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import dataclasses
import json
import os
import unittest.mock as um
from pathlib import Path

from llminference import experiments


def test_run_one(tmp_path: Path) -> None:
    # A basic integration sanity-check
    with um.patch.dict(os.environ, dict(WANDB_DIR=str(tmp_path))):
        out = experiments.run_one(
            experiments.Experiment(
                "test",
                task=experiments.Task(
                    "squad", shots=1, samples=5, confusion_contexts=7
                ),
                model="EleutherAI/pythia-70m",
                sparsity=experiments.Sparsity("sparse_v", k=32),
                execution=dataclasses.replace(
                    experiments.Execution.auto(), wandb="offline"
                ),
            )
        )
        # Non-exhaustive check of the output fields
        assert out["name"] == "test"
        assert out["sparsity"] == dict(name="sparse_v", k=32)
        assert out["model_config"]["hidden_size"] == 512
        assert len(out["results"]) == 5
        assert 0 <= out["match"] <= 1
        assert out["wandb"]["id"]


def test_run_many(tmp_path: Path) -> None:
    with um.patch(
        "llminference.experiments.run_one",
        lambda xp, progress: dict(
            **xp.to_dict(), outcome="fake", pid=os.getpid(), progress=progress
        ),
    ):
        xps = [
            experiments.Experiment(
                "test",
                task=experiments.Task(
                    "squad", shots=1, samples=5, confusion_contexts=7
                ),
                model="EleutherAI/pythia-70m",
                sparsity=experiments.Sparsity("sparse_v", k=32),
                execution=dataclasses.replace(
                    experiments.Execution.auto(), wandb="offline"
                ),
            )
        ]
        for n_workers, filename in [(1, "one.jsonl"), (2, "many.jsonl")]:
            experiments.run_many(xps, n_workers=n_workers, out=tmp_path / filename)

    out_one = json.loads((tmp_path / "one.jsonl").read_text())
    out_many = json.loads((tmp_path / "many.jsonl").read_text())
    # `out_one` ran in this process, `out_many` ran in a subprocess
    assert out_one.pop("pid") != out_many.pop("pid")
    assert out_one == out_many
    assert out_one["progress"] is False

# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from pathlib import Path

import torch

import llminference as L

if __name__ == "__main__":
    torch.set_num_threads(32)
    out_path = Path("data/gen")
    out_path.mkdir(exist_ok=True, parents=True)
    for name in [
        *[
            f"EleutherAI/pythia-{size}"
            for size in [
                "70m",
                "160m",
                "410m",
                "1b",
                "1.4b",
                "2.8b",
                "6.9b",
                "12b",
            ]
        ],
        *[
            f"facebook/opt-{size}"
            for size in [
                "125m",
                "1.3b",
                "2.7b",
                "6.7b",
                "13b",
            ]
        ],
        "mosaicml/mpt-7b",
    ]:
        data = L.tasks.outcompare.generate_dataset(
            name,
            prompt_length=128,
            completion_length=64,
            batch_size=64,
        )
        data.save(out_path / f"{name.split('/')[-1]}.json")

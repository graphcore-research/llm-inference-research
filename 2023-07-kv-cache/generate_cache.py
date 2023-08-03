"""Iterate through the dataset and save KV caches"""

from math import ceil
from pathlib import Path
from typing import Iterator, List

import datasets
from datasets import Dataset
from tqdm import tqdm

import llminference as L

DATASET_PATH = "/nethome/lukar/datasets/triviaqa_2k"
CACHE_DIR = "/net/group/research/lukar/cache"
BATCH_SIZE = 16


def batch_data(dataset: Dataset, batch_size: int) -> Iterator[List[str]]:
    l = len(dataset)
    for i in range(0, l, batch_size):
        items = dataset[i : min(i + batch_size, l)]
        yield [entity["wiki_context"][0] for entity in items["entity_pages"]]


def generate_cache(
    model: str,
    dataset: Dataset,
    batch_size: int = BATCH_SIZE,
    cache_path: str = "cache",
) -> dict:
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    for batch in tqdm(
        batch_data(dataset, batch_size=adapter.batch_size),
        total=ceil(len(dataset) / adapter.batch_size),
    ):
        adapter.generate_kv_cache(batch, cache_path)
    return dict(model=model, num_examples=len(dataset))


models = [
    "pythia-70m",
    "pythia-160m",
    "pythia-410m",
    "pythia-1b",
    "pythia-1.4b",
    "pythia-2.8b",
    # "pythia-6.9b",
    # "pythia-12b",
]

triviaqa_2k = datasets.load_from_disk(DATASET_PATH)["validation"]
# Select a subset
# triviaqa_2k = datasets.load_from_disk(DATASET_PATH)["validation"].select(range(10))

# Single-process generation
for model in tqdm(models):
    cache_path = Path(CACHE_DIR, model)
    generate_cache(model, triviaqa_2k, cache_path=str(cache_path))


# Multi-process generation
# settings = [
#     dict(model=model, dataset=triviaqa_2k, cache_path=str(Path(CACHE_DIR, model)))
#     for model in models
# ]
# dest = Path(__file__).parent / "../../out/cache_generation.jsonl"
# L.utility.run_multiprocess_sweep(generate_cache, settings, dest, n_workers=4)

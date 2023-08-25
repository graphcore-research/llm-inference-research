import sys
from itertools import islice
from typing import *

import datasets
import torch
import tqdm

import llminference as L

torch.set_num_threads(16)


def create_example(d: Dict[str, Any], prompt: str) -> Dict[str, Any]:
    if 4000 <= len(d["article"]) <= 8000:
        return dict(
            id=d["id"],
            context="Article: " + d["article"],
            prompt=prompt,
            reference=d["highlights"],
        )


data = datasets.load_dataset("cnn_dailymail", name="3.0.0")["validation"].shuffle(
    2353669
)
prompt = "\nSummary:"
examples = list(
    islice(
        filter(None, (create_example(data[i], prompt) for i in range(len(data)))),
        200,
    )
)
model_scale = ["410m", "1b"][int(sys.argv[1])]
batch_size = 10
generation_context = L.eval_adapter.null_model_context

with L.utility.jsonlines_writer(f"sweep_{model_scale}.jsonl") as write:
    adapter = L.Adapter.from_pretrained(f"EleutherAI/pythia-{model_scale}")
    for batch in tqdm.tqdm(list(L.utility.batches(examples, batch_size))):
        outputs = adapter.greedy_sample(
            [b["context"] for b in batch],
            [b["prompt"] for b in batch],
            num_generated_tokens=256,
            max_prompt_and_generated_tokens=256 + 16,
            use_cache=False,
            generation_context=generation_context,
        )
        for b, out in zip(batch, list(outputs)):
            write(
                dict(
                    id=b["id"],
                    reference=b["reference"],
                    output=adapter.tok_decode(out),
                )
            )

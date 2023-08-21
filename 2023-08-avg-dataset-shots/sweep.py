import sys
import unittest.mock as um
from functools import partial

import torch

import llminference as L

CACHE = "/net/group/research/douglaso/llminference-cache"
torch.set_num_threads(16)

dataset, shots = [("TriviaQA", 0), ("TriviaQA", 1), ("SQuAD", 0), ("SQuAD", 1)][
    int(sys.argv[1])
]
data = getattr(L.qa, dataset).data()
examples = [L.qa.add_few_shot_prompt(data[i], k=shots) for i in range(400)]
adapter = L.Adapter.from_pretrained("EleutherAI/pythia-2.8b")
batch_size = 5

with L.utility.jsonlines_writer(f"results-2.8b/{dataset}_{shots}.jsonl") as write:
    for k in [None] + [2**n for n in range(4, 9)]:
        for average in [False, True]:
            with um.patch(
                "torch.nn.functional.softmax",
                partial(
                    L.sparse_attention.sparse_softmax_fixed_k,
                    k=k,
                    add_avg=average,
                ),
            ):
                results = L.qa.evaluate(
                    adapter,
                    examples,
                    batch_size=batch_size,
                    open_book=True,
                    cache_dir=CACHE,
                    desc=f"Evaluating k={k} average={average}",
                )
                for result in results:
                    write(
                        dict(
                            model_scale="2.8b",
                            dataset=dataset,
                            shots=shots,
                            sparse_softmax_k=k,
                            sparse_softmax_average=average,
                            **result,
                        )
                    )

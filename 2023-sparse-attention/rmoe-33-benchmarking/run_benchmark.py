import time

import torch

import llminference as L

with L.utility.jsonlines_writer("data/benchmark.jsonl") as writer:
    for model_scale in ["410m", "1b", "2.8b"]:
        adapter = L.Adapter.from_pretrained(f"EleutherAI/pythia-{model_scale}")
        for num_threads in [8, 16, 32, 64]:
            torch.set_num_threads(num_threads)
            for batch_size in [1, 4, 16]:
                for context_length in [16, 128, 1024]:
                    for num_tokens in [16, 32, 64]:
                        s = dict(
                            model_scale=model_scale,
                            num_threads=num_threads,
                            batch_size=batch_size,
                            context_length=context_length,
                            num_tokens=num_tokens,
                        )
                        print(s)
                        for rep in range(5):
                            t0 = time.time()
                            adapter.greedy_sample(
                                [
                                    " ".join("." for _ in range(context_length))
                                    for _ in range(batch_size)
                                ],
                                ["\n" for _ in range(batch_size)],
                                num_generated_tokens=num_tokens,
                                use_cache=False,
                            )
                            elapsed = time.time() - t0
                            writer(dict(**s, rep=rep, elapsed=elapsed))

from pathlib import Path
from typing import *

import llminference as L

CACHE = "/net/group/research/douglaso/llminference-cache/cnn_dailymail"


def model_context(
    sparse_softmax_k: Optional[int], sparse_softmax_avg: Optional[bool]
) -> L.eval_adapter.ModelContext:
    if sparse_softmax_k is None:
        return L.eval_adapter.null_model_context
    return L.eval_adapter.patch_for_model(
        "torch.nn.functional.softmax",
        L.sparse_attention.sparse_softmax_fixed_k,
        k=sparse_softmax_k,
        add_avg=sparse_softmax_avg,
    )


def run(model_scale: str, batch_size: int, **model_args: Any) -> Dict[str, Any]:
    data = L.summarisation.CnnDailymail.data()
    adapter = L.Adapter.from_pretrained(f"EleutherAI/pythia-{model_scale}")
    results = list(
        L.summarisation.evaluate(
            adapter,
            [data[i] for i in range(200)],
            batch_size=batch_size,
            generation_context=model_context(**model_args),
            cache_dir=CACHE,
        )
    )
    return dict(
        model_scale=model_scale,
        **model_args,
        rougeL=sum(r["rougeL"] for r in results) / len(results),
        results=results,
    )


if __name__ == "__main__":
    model_scale_and_batch_size = [
        # ("160m", 32),
        # ("410m", 16),
        # ("1b", 16),
        # ("1.4b", 8),
        ("2.8b", 8),
    ]
    sweeps = {}
    # sweeps["baselines"] = [
    #     dict(
    #         model_scale=m, batch_size=b, sparse_softmax_k=None, sparse_softmax_avg=None
    #     )
    #     for m, b in model_scale_and_batch_size
    # ]
    sweeps["sparse"] = [
        dict(model_scale=m, batch_size=b, sparse_softmax_k=k, sparse_softmax_avg=avg)
        for m, b in model_scale_and_batch_size
        for k in [16, 32, 64, 128]
        for avg in [False, True]
    ]
    for name, sweep in sweeps.items():
        print(name, len(sweep), sweep)
        L.utility.run_multiprocess_sweep(
            run, sweep, Path(f"data/{name}.jsonl"), n_workers=2
        )

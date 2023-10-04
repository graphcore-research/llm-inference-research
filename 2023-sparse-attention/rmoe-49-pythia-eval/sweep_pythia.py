from pathlib import Path

import llminference as L


def run_experiment(model, prompt, open_book, pe, n_examples=400, batch_size=4):
    examples = [
        L.qa.add_zero_shot_prompt(data[i], prompt) 
        for i in range(n_examples)
    ]
    adapter = L.Adapter.from_pretrained("EleutherAI/" + model, batch_size=batch_size)
    results = list(
        L.qa.evaluate(adapter, examples, batch_size=batch_size, use_cache=False, open_book=open_book, permissive_eval=pe)
    )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, prompt=prompt, acc=acc, pe=pe, open_book=open_book, n_examples=n_examples)


if __name__ == "__main__":
    data = L.qa.TriviaQA.data()
    prompts =  ["\nQuestion: {question}\nAnswer:", "\nQuestion: {question}\nSingle-word answer:"]
    models = [
        "pythia-70m",
        "pythia-160m",
        "pythia-410m",
        "pythia-1b",
        "pythia-1.4b",
        "pythia-2.8b",
    ]

    settings = [
        dict(
            model=model,
            prompt=prompt,
            open_book=open_book,
            pe=pe,
        )
        for model in models
        for prompt in prompts
        for open_book in [False, True]
        for pe in [True, False]
    ]
    dest = Path(__file__).parent / "results/llama.jsonl"

    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)

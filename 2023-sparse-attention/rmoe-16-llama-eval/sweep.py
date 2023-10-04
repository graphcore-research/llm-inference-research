from pathlib import Path

import llminference as L


def run_experiment(model, examples, open_book, batch_size=4):
    adapter = L.Adapter.from_pretrained("meta-llama/" + model, batch_size=batch_size)
    results = list(
        L.qa.evaluate(adapter, examples, batch_size=batch_size, open_book=open_book, use_cache=False)
    )
    acc = sum(r["match"] for r in results) / len(results)
    return dict(model=model, acc=acc, open_book=open_book)


if __name__ == "__main__":
    data = L.qa.TriviaQA.data()
    n_examples = 400
    examples_1 = [
        L.qa.add_zero_shot_prompt(data[i], "\nQuestion: {question}\nAnswer:") 
        for i in range(n_examples)
    ]

    examples_3 = [
        L.qa.add_zero_shot_prompt(data[i], "Question: {question} \nAnswer:")
        for i in range(n_examples)
    ]

    models = ["Llama-2-7b-hf"]

    settings = [
        dict(
            model=model,
            examples=examples,
            open_book=open_book,
        )
        for model in models
        for examples in [examples_1,examples_3]
        for open_book in [False, True]
    ]
    dest = Path(__file__).parent / "results/llama.jsonl"

    L.utility.run_multiprocess_sweep(run_experiment, settings, dest, n_workers=1)

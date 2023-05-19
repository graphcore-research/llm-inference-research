import lm_eval.evaluator

from .. import eval_adapter


def test_eval_adapter() -> None:
    adapter = eval_adapter.Adapter.from_pretrained("EleutherAI/pythia-70m")
    out = lm_eval.evaluator.evaluate(
        adapter, lm_eval.tasks.get_task_dict(["wikitext"]), limit=1
    )
    assert 1 < out["results"]["wikitext"]["word_perplexity"] < 200

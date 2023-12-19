# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import transformers

from llminference.tasks import outcompare


def test_outcompare() -> None:
    dataset = outcompare.generate_dataset(
        "EleutherAI/pythia-70m",
        prompt_length=16,
        completion_length=20,
        batch_size=4,
        limit=4,
        progress=False,
    )

    # The original model should have perfect metrics
    model = transformers.AutoModelForCausalLM.from_pretrained(dataset.model)
    out = outcompare.evaluate(model, dataset, batch_size=4)
    assert out == dict(
        entropy_rmse=0,
        entropy_rmse_stderr=0,
        exact_match_length=20,
        exact_match_length_stderr=0,
        edit_distance_L16=0,
        edit_distance_L16_stderr=0,
    )

    # Break the model a bit, and re-evaluate
    model.gpt_neox.layers[4].attention.dense.weight.data.fill_(0)
    out = outcompare.evaluate(model, dataset, batch_size=4)
    assert 0 < out["entropy_rmse"] < 10
    assert 0 < out["entropy_rmse_stderr"] < 10
    assert 0 < out["exact_match_length"] <= 20
    assert 0 < out["exact_match_length_stderr"] <= 20
    assert 0 < out["edit_distance_L16"] <= 16
    assert 0 < out["edit_distance_L16_stderr"] <= 16

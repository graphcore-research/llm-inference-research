from functools import partial

import torch
import transformers.models.llama.modeling_llama

import llminference as L

if __name__ == "__main__":
    adapter = L.Adapter.from_pretrained("meta-llama/Llama-2-7b-hf")
    adapter.model = L.utility.convert_module(
        adapter.model,
        lambda m: L.models.llama_attention.LlamaAttention(adapter.model.config)
        if isinstance(m, transformers.models.llama.modeling_llama.LlamaAttention)
        else None,
    )
    data = L.tasks.qa.SQuAD.data()
    examples = [
        L.tasks.qa.add_few_shot_prompt(
            data[i],
            k=1,
            prompt_template=L.tasks.qa.get_default_prompt_template(
                adapter.model.config._name_or_path, shots=1
            ),
        )
        for i in range(200)
    ]
    max_tokens = 2048
    examples = [
        x
        for x in examples
        if len(adapter.tok_encode(x["context"] + x["prompt"])) + 16 <= max_tokens
    ][:40]

    log = dict(q={}, m={}, k={}, v={})

    def logging_attn(layer_idx, self, query, key, value, attention_mask=None):
        assert not (attention_mask[0, 0, -1, :] < 0).any()
        assert key.shape[2] <= max_tokens
        data = dict(
            q=query[:, :, -1:, :],
            m=torch.where(
                torch.arange(max_tokens) >= max_tokens - key.shape[2], 0, -1e9
            )[None, None, None],
            k=torch.nn.functional.pad(
                key, [0, 0, max_tokens - key.shape[2], 0, 0, 0, 0, 0]
            ),
            v=torch.nn.functional.pad(
                value, [0, 0, max_tokens - value.shape[2], 0, 0, 0, 0, 0]
            ),
        )
        data = {key: value.half() for key, value in data.items()}
        for name in log:
            log[name][layer_idx] = (
                torch.concat([log[name][layer_idx], data[name]], axis=0)
                if layer_idx in log[name]
                else data[name]
            )
        return L.models.llama_attention.LlamaAttention._attn(
            self, query, key, value, attention_mask=attention_mask
        )

    for layer_idx, layer in enumerate(adapter.model.model.layers):
        layer.self_attn._attn = partial(logging_attn, layer_idx, layer.self_attn)

    out = list(
        L.tasks.qa.evaluate(adapter, examples, batch_size=1, output_token_limit=1)
    )
    torch.save(
        {
            name: torch.stack(list(layers.values())).swapaxes(0, 1)
            for name, layers in log.items()
        },
        f"data/llama7b_x{len(examples)}.pt",
    )

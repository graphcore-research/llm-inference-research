import copy
from typing import Any, Optional, Tuple, cast

import torch
from torch import nn
from transformers.models.gpt_neox.modeling_gpt_neox import (
    GPTNeoXForCausalLM,
    GPTNeoXLayer,
)
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaForCausalLM


class PipelinedLlamaDecoderLayer(LlamaDecoderLayer):  # type:ignore[misc]
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
        use_cache: Optional[bool] = False,
        **kwargs: Any,
    ) -> Tuple[
        torch.FloatTensor, Optional[Tuple[torch.FloatTensor, torch.FloatTensor]]
    ]:
        layer_device = self.input_layernorm.weight.device
        if hidden_states.device != layer_device:
            hidden_states = hidden_states.to(layer_device)
        if attention_mask.device != layer_device:
            attention_mask = attention_mask.to(layer_device)
        if position_ids.device != layer_device:
            position_ids = position_ids.to(layer_device)
        return super().forward(  # type:ignore[no-any-return]
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            **kwargs,
        )


class PipelinedGPTNeoXLayer(GPTNeoXLayer):  # type:ignore[misc]
    def forward(
        self,
        hidden_states: Optional[torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = False,
        layer_past: Optional[Tuple[torch.Tensor]] = None,
        output_attentions: Optional[bool] = False,
    ) -> Any:
        layer_device = self.input_layernorm.weight.device
        if hidden_states is not None and hidden_states.device != layer_device:
            hidden_states = hidden_states.to(layer_device)
        if attention_mask is not None and attention_mask.device != layer_device:
            attention_mask = attention_mask.to(layer_device)
        if position_ids is not None and position_ids.device != layer_device:
            position_ids = cast(torch.LongTensor, position_ids.to(layer_device))
        if head_mask is not None and head_mask.device != layer_device:
            head_mask = head_mask.to(layer_device)
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            head_mask=head_mask,
            use_cache=use_cache,
            layer_past=layer_past,
            output_attentions=output_attentions,
        )


def pipeline_llama(model: LlamaForCausalLM) -> LlamaForCausalLM:
    """Convert a Llama model to use pipelined layers."""

    def _convert(m: torch.nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, LlamaDecoderLayer):
                replacement = PipelinedLlamaDecoderLayer(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config)
    return model


def pipeline_gptneox(model: GPTNeoXForCausalLM) -> GPTNeoXForCausalLM:
    """Convert a GPTNeoX model to use pipelined layers."""

    def _convert(m: torch.nn.Module, **args: Any) -> None:
        for name, child in m.named_children():
            if isinstance(child, GPTNeoXLayer):
                replacement = PipelinedGPTNeoXLayer(**args)
                replacement.to(next(child.parameters()).dtype)
                replacement.load_state_dict(child.state_dict(), strict=False)
                setattr(m, name, replacement)
            _convert(child, **args)

    model = copy.deepcopy(model)
    _convert(model, config=model.config)
    return model


def pipeline_model(model: nn.Module) -> nn.Module:
    model_name = cast(str, model.config._name_or_path)  # type:ignore[union-attr]
    num_gpus = torch.cuda.device_count()
    num_hidden_layers = cast(
        int, model.config.num_hidden_layers  # type:ignore[union-attr]
    )
    partition_len = ((num_hidden_layers - 1) // num_gpus) + 1
    gpu_allocation = [i // partition_len for i in range(num_hidden_layers)]
    if "llama" in model_name:
        model = pipeline_llama(model)
        model.model.embed_tokens.cuda(0)  # type:ignore[union-attr]
        trunk = cast(nn.Module, model.model.layers)  # type:ignore[union-attr]
        for idx, (_, layer) in enumerate(trunk.named_children()):
            layer.cuda(gpu_allocation[idx])
        model.model.norm.cuda(num_gpus - 1)  # type:ignore[union-attr]
        model.lm_head.cuda(num_gpus - 1)
    elif "pythia" in model_name:
        model = pipeline_gptneox(model)
        model.gpt_neox.embed_in.cuda(0)  # type:ignore[union-attr]
        model.gpt_neox.emb_dropout.cuda(0)  # type:ignore[union-attr]
        trunk = cast(nn.Module, model.gpt_neox.layers)  # type:ignore[union-attr]
        for idx, (_, layer) in enumerate(trunk.named_children()):
            layer.cuda(gpu_allocation[idx])
        model.gpt_neox.final_layer_norm.cuda(num_gpus - 1)  # type:ignore[union-attr]
        model.embed_out.cuda(num_gpus - 1)
    else:
        raise ValueError
    return model

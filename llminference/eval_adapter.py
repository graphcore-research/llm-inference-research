from typing import Iterable, List, cast

import lm_eval.base
import torch
import transformers
from torch import Tensor


class Adapter(lm_eval.base.BaseLM):  # type:ignore[misc]
    """A simplified adapter for lm_eval <-> HuggingFace."""

    DEFAULT_BATCH_SIZE = 32

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        tokenizer: transformers.PreTrainedTokenizerBase,
        batch_size: int,
    ):
        super().__init__()
        self.model = model
        self.tokenizer = tokenizer
        self._batch_size = batch_size

    @classmethod
    def from_pretrained(
        cls, pretrained_model_name_or_path: str, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> "Adapter":
        return cls.from_model(
            transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path
            ),
            batch_size=batch_size,
        )

    @classmethod
    def from_model(
        cls, model: transformers.PreTrainedModel, batch_size: int = DEFAULT_BATCH_SIZE
    ) -> "Adapter":
        return cls(
            model=model,
            tokenizer=transformers.AutoTokenizer.from_pretrained(
                model.config._name_or_path
            ),
            batch_size=batch_size,
        )

    # lm_eval API

    @property
    def eot_token_id(self) -> int:
        return cast(int, self.tokenizer.eos_token_id)

    @property
    def max_length(self) -> int:
        return cast(int, self.model.config.max_position_embeddings)

    @property
    def max_gen_toks(self) -> int:
        return 128  # arbitary?

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return cast(torch.device, self.model.device)

    def tok_encode(self, string: str) -> List[int]:
        return cast(List[int], self.tokenizer.encode(string, add_special_tokens=False))

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return cast(str, self.tokenizer.decode(tokens))

    def _model_call(self, inps: Tensor) -> Tensor:
        with torch.no_grad():
            logits: Tensor = self.model(inps).logits
            return logits

    def _model_generate(
        self, context: Tensor, max_length: int, eos_token_id: int
    ) -> Tensor:
        generation: Tensor = self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )
        return generation

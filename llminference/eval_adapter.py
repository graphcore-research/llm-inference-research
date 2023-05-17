from typing import Iterable, List

import lm_eval.base
import torch
import transformers


class Adapter(lm_eval.base.BaseLM):
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
        cls, pretrained_model_name_or_path: str, batch_size: int = 16
    ) -> "Adapter":
        return cls(
            model=transformers.AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path
            ),
            tokenizer=transformers.AutoTokenizer.from_pretrained(
                pretrained_model_name_or_path
            ),
            batch_size=batch_size,
        )

    # lm_eval API

    @property
    def eot_token_id(self) -> int:
        return self.tokenizer.eos_token_id

    @property
    def max_length(self) -> int:
        return self.model.config.max_position_embeddings

    @property
    def max_gen_toks(self) -> int:
        return 128  # arbitary?

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def device(self) -> torch.device:
        return self.model.device

    def tok_encode(self, string: str) -> List[str]:
        return self.tokenizer.encode(string, add_special_tokens=False)

    def tok_decode(self, tokens: Iterable[int]) -> str:
        return self.tokenizer.decode(tokens)

    def _model_call(self, inps: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            return self.model(inps).logits

    def _model_generate(
        self, context: torch.Tensor, max_length: int, eos_token_id: int
    ):
        return self.model.generate(
            context, max_length=max_length, eos_token_id=eos_token_id, do_sample=False
        )

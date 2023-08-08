import hashlib
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, cast

import lm_eval.base
import torch
import transformers
from torch import Tensor
from torch.nn.functional import pad

KVCache = Tuple[Tuple[Tensor, Tensor], ...]


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

        # Set padding token
        self.tokenizer.pad_token = self.tokenizer.eos_token

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

    @contextmanager
    def tokenizer_padding(self, padding_side: str) -> Iterator[None]:
        try:
            default_pad_side = self.tokenizer.padding_side
            self.tokenizer.padding_side = padding_side
            yield
        finally:
            self.tokenizer.padding_side = default_pad_side

    def _get_cache_str(self, s: str) -> str:
        """Return a hash identifier for the KV cache
        based on the input string and the model config.

        Args:
            s (str): Context string that will be cached

        Returns:
            str: Filename hash used for caching
        """
        model_config = self.model.config.to_json_string(use_diff=False)
        hash_fn = hashlib.md5()
        for item in [model_config, s]:
            hash_fn.update(item.encode())

        return f"{hash_fn.hexdigest()}"

    @staticmethod
    def _kv_to_tuple(kv_cache: torch.Tensor) -> KVCache:
        return tuple((layer[0], layer[1]) for layer in kv_cache)

    @staticmethod
    def _kv_to_tensor(kv_cache: KVCache) -> torch.Tensor:
        return torch.stack([torch.stack(kv) for kv in kv_cache])

    def _save_kv_cache(
        self,
        past_key_values: KVCache,
        filepaths: List[Path],
        sequence_lens: List[Optional[int]],
    ) -> None:
        """
        Given KV cache for a batch of sequences, save the cache for each sequence i
        separately into a Tensor of shape
        (num_layers, 2, 1, num_heads, sequence_lens[i], embed_size_per_head)

        Args:
            past_key_values (KVCache): per-layer cache of (K, V) matrices, each
            of shape (batch_size, num_heads, max_sequence_length, embed_size_per_head)
            filepaths (List[Path]): per-sequence filepath
            sequence_lens (List[Optional[int]]): per-sequence sequence_length
        """
        batch_size = past_key_values[0][0].shape[0]
        assert (
            len(filepaths) == batch_size
        ), "Number of filepaths needs to match the batch size"
        assert (
            len(sequence_lens) == batch_size
        ), "Number of sequence lengths needs to match the batch size"

        past_key_values_t = self._kv_to_tensor(past_key_values)
        for i, (sequence_len, filepath) in enumerate(zip(sequence_lens, filepaths)):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Expect right-padded batch
            torch.save(
                past_key_values_t[:, :, i : i + 1, :, :sequence_len, :].clone(),
                filepath,
            )

    def generate_kv_cache(self, text: List[str], dir_path: str) -> None:
        """Given a batch of input strings, run the model and save
        the KV cache for each sequence individually.
        Note: uses right padding when batching.

        Args:
            text (List[str]): Batch of text inputs
            dir_path (str): Directory for storing the KV cache
        """
        with self.tokenizer_padding("right"):
            enc = self.tokenizer(text, return_tensors="pt", padding=True)

        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            past_key_values = self.model(
                input_ids=input_ids, attention_mask=attention_mask
            ).past_key_values

        # Get the actual length of each sequence in the batch
        sequence_lens = attention_mask.sum(dim=-1).tolist()

        # Get the filepath hash for each sequence
        filepaths = [
            Path(dir_path) / Path(self._get_cache_str(s) + ".pt") for s in text
        ]

        self._save_kv_cache(past_key_values, filepaths, sequence_lens)

    def greedy_sample(
        self,
        ctxs: List[str],
        questions: List[str],
        num_generated_tokens: int,
        use_cache: bool = True,
        cache_dir: str = "cache/",
    ) -> Tensor:
        """
        Sample greedily from the model assuming the prompts are ctx + question,
        where KV cache for the ctx can be loaded from memory.

        Args:
            ctxs (List[str]): List of context strings prepended to questions
            questions (List[str]): List of question strings
            num_generated_tokens (int): Number of generated tokens
            use_cache (bool, optional): Load ctx KV cache from disk if it exits.
            If not, first generate and save the cache. Defaults to True.
            cache_dir (str, optional): Directory for saving/loading KV cache.
            Defaults to "cache/".

        Returns:
            Tensor: Generated tokens. Shape - (batch_size, num_generated_tokens)
        """
        batch_size = len(ctxs)
        assert len(ctxs) == len(
            questions
        ), "Number of context and question strings should be the same"

        if not use_cache:
            with self.tokenizer_padding("left"):
                inp = self.tokenizer(
                    [ctx + question for ctx, question in zip(ctxs, questions)],
                    return_tensors="pt",
                    padding=True,
                )
            input_ids = inp["input_ids"]
            attention_mask = inp["attention_mask"]
            past_key_values = None
            position_ids = None
        else:
            filepaths = [
                Path(cache_dir, self._get_cache_str(ctx) + ".pt") for ctx in ctxs
            ]
            cache_exists = [filepath.exists() for filepath in filepaths]

            # Generate KV cache where missing
            if not all(cache_exists):
                ctxs_to_cache = [
                    ctx for ctx, exist in zip(ctxs, cache_exists) if not exist
                ]
                self.generate_kv_cache(ctxs_to_cache, dir_path=cache_dir)

            # All KV caches should be available now
            cache_exists = [filepath.exists() for filepath in filepaths]
            assert all(cache_exists), "Issue generating KV cache"

            # Left-pad the KV caches to maximum sequence length in batch
            cache = [torch.load(filepath) for filepath in filepaths]
            seq_lens = [pkv.shape[-2] for pkv in cache]
            max_len = max(seq_lens)
            past_key_values = torch.cat(
                [
                    pad(pkv, (0, 0, max_len - seq_len, 0))
                    for pkv, seq_len in zip(cache, seq_lens)
                ],
                dim=2,
            )
            attention_mask_left = torch.tensor(
                [[i >= max_len - len for i in range(max_len)] for len in seq_lens]
            ).long()

            # Tokenize questions with left padding (easier generation)
            with self.tokenizer_padding("left"):
                questions_enc = self.tokenizer(
                    questions, return_tensors="pt", padding=True
                )

            input_ids = questions_enc["input_ids"]
            attention_mask_right = questions_enc["attention_mask"]

            # Apply correct position ids given padding tokens
            q_len = input_ids.shape[-1]
            q_no_pad_lens = attention_mask_right.sum(dim=-1)
            position_ids = torch.stack(
                [
                    torch.arange(
                        seq_len - (q_len - q_no_pad_len), seq_len + q_no_pad_len
                    )
                    for seq_len, q_no_pad_len in zip(seq_lens, q_no_pad_lens)
                ]
            )

            # Concatanate cache and questions attention masks
            attention_mask = torch.cat(
                [attention_mask_left, attention_mask_right], dim=1
            )

        # Generate tokens one by one
        generated_tokens = []
        with torch.no_grad():
            for _ in range(num_generated_tokens):
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    position_ids=position_ids,
                )
                logits = out.logits[:, -1, :]
                next_token = logits.argmax(dim=-1, keepdim=True)
                generated_tokens.append(next_token)

                # Prepare inputs for next iteration
                input_ids = next_token
                attention_mask = torch.cat(
                    [attention_mask, torch.ones(batch_size, 1)], dim=1
                )
                past_key_values = out.past_key_values
                if position_ids is not None:
                    position_ids = position_ids[:, -1:] + 1

        return torch.cat(generated_tokens, dim=1)


def evaluate_prediction(out: Tensor, targets: List[List[int]]) -> bool:
    # Assume out is a single sequence
    assert len(out.shape) == 1
    return any(
        torch.equal(out[: len(target)], torch.tensor(target)) for target in targets
    )

import hashlib
import logging
import struct
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, List, Optional, Tuple, cast

import lm_eval.base
import torch
import transformers
from torch import Tensor
from torch.nn.functional import pad

KVCache = Tuple[Tuple[Tensor, Tensor], ...]

logger = logging.getLogger(__name__)


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
    def tokenizer_override(self, padding: str, truncation: str) -> Iterator[None]:
        try:
            defaults = self.tokenizer.padding_side, self.tokenizer.truncation_side
            self.tokenizer.padding_side = padding
            self.tokenizer.truncation_side = truncation
            yield
        finally:
            self.tokenizer.padding_side, self.tokenizer.truncation_side = defaults

    def _get_cache_str(self, s: str, limit: int) -> str:
        """Return a hash identifier for the KV cache
        based on the input string and the model config.

        Args:
            s (str): Context string that will be cached
            limit (int): Token length limit, before truncation occurs

        Returns:
            str: Filename hash used for caching
        """
        model_config = self.model.config.to_json_string(use_diff=False)
        hash_fn = hashlib.md5()
        for item in [model_config, s]:
            hash_fn.update(item.encode())
        hash_fn.update(struct.pack("<L", limit))

        return f"{hash_fn.hexdigest()}"

    @staticmethod
    def _kv_to_tuple(kv_cache: Tensor) -> KVCache:
        return tuple((layer[0], layer[1]) for layer in kv_cache)

    @staticmethod
    def _kv_to_tensor(kv_cache: KVCache) -> Tensor:
        """
        Args:
            past_key_values (KVCache): per-layer cache of (K, V) matrices, each
            of shape (batch_size, num_heads, max_sequence_length, embed_size_per_head)

        Returns:
            Tensor:
            (layer, 2, batch_size, num_heads, max_sequence_length, embed_size_per_head)
        """
        return torch.stack([torch.stack(kv) for kv in kv_cache])

    def _save_kv_cache(
        self,
        past_key_values: Tensor,
        filepaths: List[Path],
        sequence_lens: List[Optional[int]],
    ) -> None:
        """
        Given KV cache for a batch of sequences, save the cache for each sequence i
        separately into a Tensor of shape
        (num_layers, 2, 1, num_heads, sequence_lens[i], embed_size_per_head)

        Args:
            past_key_values (Tensor): as returned by _kv_to_tensor
            filepaths (List[Path]): per-sequence filepath
            sequence_lens (List[Optional[int]]): per-sequence sequence_length
        """
        batch_size = past_key_values.shape[2]
        assert (
            len(filepaths) == batch_size
        ), "Number of filepaths needs to match the batch size"
        assert (
            len(sequence_lens) == batch_size
        ), "Number of sequence lengths needs to match the batch size"

        for i, (sequence_len, filepath) in enumerate(zip(sequence_lens, filepaths)):
            filepath.parent.mkdir(parents=True, exist_ok=True)
            # Expect right-padded batch
            torch.save(
                past_key_values[:, :, i : i + 1, :, :sequence_len, :].clone(),
                filepath,
            )

    def prefill_with_cache(
        self, text: List[str], max_context_length: int, dir_path: Optional[str]
    ) -> List[Tensor]:
        """Given a batch of input strings, run the model and save
        the KV cache for each sequence individually.
        Note: uses right padding when batching.

        Args:
            text (List[str]): Batch of text inputs
            max_context_length (int): Maximum length of context (trims from the left
            to fit inside this limit)
            dir_path (Optional[str]): Directory for storing the KV cache

        Returns:
            List[Tensor]: [batch_size x
            (layer, 2, 1, num_heads, sequence_length, embed_size_per_head)]
        """
        with self.tokenizer_override(padding="right", truncation="left"):
            enc = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=max_context_length,
            )

        input_ids, attention_mask = enc["input_ids"], enc["attention_mask"]

        with torch.no_grad():
            out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        past_key_values = self._kv_to_tensor(out.past_key_values)

        # Get the actual length of each sequence in the batch
        sequence_lens = attention_mask.sum(dim=-1).tolist()

        if dir_path:
            # Get the filepath hash for each sequence
            filepaths = [
                Path(dir_path) / (self._get_cache_str(s, max_context_length) + ".pt")
                for s in text
            ]
            self._save_kv_cache(past_key_values, filepaths, sequence_lens)

        return [
            past_key_values[:, :, i : i + 1, :, :sequence_len, :]
            for i, sequence_len in enumerate(sequence_lens)
        ]

    def greedy_sample(
        self,
        ctxs: List[str],
        prompts: List[str],
        num_generated_tokens: int,
        max_prompt_and_generated_tokens: int = 256,
        use_cache: bool = True,
        cache_dir: str = "cache/",
    ) -> Tensor:
        """
        Sample greedily from the model assuming the prompts are ctx + prompt,
        where KV cache for the ctx can be loaded from memory.

        Args:
            ctxs (List[str]): List of context strings prepended to prompts
            prompts (List[str]): List of prompt strings
            num_generated_tokens (int): Number of generated tokens
            max_prompt_and_generated_tokens (int, optional): Number of tokens
            to allocate to prompt and generation. Defaults to 256.
            use_cache (bool, optional): Load ctx KV cache from disk if it exits.
            If not, first generate and save the cache. Defaults to True.
            cache_dir (str, optional): Directory for saving/loading KV cache.
            Defaults to "cache/".

        Returns:
            Tensor: Generated tokens. Shape - (batch_size, num_generated_tokens)
        """
        batch_size = len(ctxs)
        assert len(ctxs) == len(
            prompts
        ), "Number of context and prompt strings should be the same"
        max_context_length = self.max_length - max_prompt_and_generated_tokens

        if use_cache:
            filepaths = [
                Path(cache_dir) / (self._get_cache_str(ctx, max_context_length) + ".pt")
                for ctx in ctxs
            ]
            cache_exists = [filepath.exists() for filepath in filepaths]

            # Generate KV cache where missing
            if not all(cache_exists):
                ctxs_to_cache = [
                    ctx for ctx, exist in zip(ctxs, cache_exists) if not exist
                ]
                self.prefill_with_cache(
                    ctxs_to_cache,
                    max_context_length=max_context_length,
                    dir_path=cache_dir,
                )

            # All KV caches should be available now
            cache_exists = [filepath.exists() for filepath in filepaths]
            assert all(cache_exists), "Issue generating KV cache"
            cache = [torch.load(filepath) for filepath in filepaths]
        else:
            cache = self.prefill_with_cache(
                ctxs, max_context_length=max_context_length, dir_path=None
            )

        # Left-pad the KV caches to maximum sequence length in batch
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

        # Tokenize prompts with left padding (easier generation)
        with self.tokenizer_override(padding="left", truncation="left"):
            prompts_enc = self.tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            )

        input_ids = prompts_enc["input_ids"]
        attention_mask_right = prompts_enc["attention_mask"]
        if max_prompt_and_generated_tokens < input_ids.shape[1] + num_generated_tokens:
            logger.warning(
                "length of prompt (%d) + generation (%d) is more than reserved (%d)",
                input_ids.shape[1],
                num_generated_tokens,
                max_prompt_and_generated_tokens,
            )

        # Apply correct position ids given padding tokens
        q_len = input_ids.shape[-1]
        q_no_pad_lens = attention_mask_right.sum(dim=-1)
        position_ids = torch.stack(
            [
                torch.arange(seq_len - (q_len - q_no_pad_len), seq_len + q_no_pad_len)
                for seq_len, q_no_pad_len in zip(seq_lens, q_no_pad_lens)
            ]
        )
        position_ids.masked_fill_(position_ids < 0, 0)

        # Concatanate cache and prompts attention masks
        attention_mask = torch.cat([attention_mask_left, attention_mask_right], dim=1)

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

# Test only the masking + approx top-k integration part of ann_attention

# from llminference.methods import ann_attention as ann
import torch

import llminference.methods.sparse_attention as sa


def test_approx_topk_integration() -> None:
    k, local_k = 128, 32

    score = torch.randn((8, 1, 2048), dtype=torch.float)
    logmask = torch.zeros((8, 1, 2048), dtype=torch.float)

    causal_index = sa.causal_index(logmask)
    is_local = (0 <= causal_index) & (causal_index < local_k + 1)
    topk_score = score.masked_fill(is_local, torch.finfo(score.dtype).max)

    # ====== Previous implementation with masking local ======
    expected_indices = topk_score.topk(min(k + 1, score.shape[-1]), -1).indices

    # ====== New implementation without masking ======
    local_idx = topk_score.size(-1) - local_k - 1
    shape = (*topk_score.shape[:-1], k + 1)
    indices = torch.empty(shape, dtype=torch.int64, device=topk_score.device)

    # Non-local top-k indices
    indices[..., : k - local_k] = torch.topk(
        topk_score[..., :local_idx], k - local_k, dim=-1
    ).indices

    # Local indices
    indices[..., k - local_k :] = torch.arange(start=local_idx, end=topk_score.size(-1))

    torch.testing.assert_close(
        indices.sort(dim=-1)[0], expected_indices.sort(dim=-1)[0]
    )

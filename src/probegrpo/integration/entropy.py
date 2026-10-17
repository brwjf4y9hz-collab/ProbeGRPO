"""Bound transient dense-entropy memory without changing logits, precision or token order."""

from __future__ import annotations

from collections.abc import Callable


def chunked_dense_entropy(logits, *, entropy_fn: Callable, chunk_size: int):
    """Apply the original entropy formula to at most chunk_size token rows at once.

    Unlike flatten/reshape, recursive views never copy the entire vocabulary
    tensor when logits are non-contiguous. No detach, precision conversion,
    vocabulary truncation, token truncation or change to the loss is performed.
    Autograd remains intact; this bounds transient forward-only work, not all
    model/activation memory retained for an entropy backward pass.
    """
    import torch

    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("entropy chunk_size must be a positive integer")
    if logits.ndim < 2 or logits.shape[-1] == 0:
        raise ValueError("expected logits shaped (..., token, nonempty vocabulary)")
    if logits.numel() == 0:
        return logits.sum(dim=-1)
    if logits.ndim == 2:
        return torch.cat([entropy_fn(logits[start:start + chunk_size])
                          for start in range(0, logits.shape[0], chunk_size)], dim=0)
    return torch.stack([chunked_dense_entropy(part, entropy_fn=entropy_fn, chunk_size=chunk_size)
                        for part in logits.unbind(0)], dim=0)

"""Numerically stable softmax implementations for performance experiments."""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # Allows CPU-only static analysis and clear runtime errors.
    triton = None
    tl = None


def naive_softmax(x: torch.Tensor) -> torch.Tensor:
    """Stable softmax expressed as separate PyTorch operations."""
    numerator = torch.exp(x - x.max(dim=1, keepdim=True).values)
    return numerator / numerator.sum(dim=1, keepdim=True)


def eager_softmax(x: torch.Tensor) -> torch.Tensor:
    """PyTorch's optimized eager implementation."""
    return torch.softmax(x, dim=1)


if triton is not None:

    @triton.jit
    def _softmax_kernel(
        output_ptr,
        input_ptr,
        input_row_stride,
        output_row_stride,
        n_cols: tl.constexpr,
        block_size: tl.constexpr,
    ):
        row = tl.program_id(axis=0)
        offsets = tl.arange(0, block_size)
        mask = offsets < n_cols

        input_row = input_ptr + row * input_row_stride
        values = tl.load(input_row + offsets, mask=mask, other=-float("inf"))
        values = values.to(tl.float32)
        values = values - tl.max(values, axis=0)
        numerator = tl.exp(values)
        denominator = tl.sum(numerator, axis=0)
        output = numerator / denominator

        output_row = output_ptr + row * output_row_stride
        tl.store(output_row + offsets, output, mask=mask)


def triton_softmax(x: torch.Tensor) -> torch.Tensor:
    """Run a fused row-wise Triton softmax kernel.

    The educational kernel supports contiguous 2D CUDA tensors whose row width
    can be processed by one Triton program. It intentionally favors clarity
    over covering every possible shape and layout.
    """
    if triton is None:
        raise RuntimeError("Triton is not installed.")
    if not x.is_cuda:
        raise ValueError("triton_softmax requires a CUDA tensor.")
    if x.ndim != 2:
        raise ValueError("triton_softmax expects a 2D tensor.")
    if not x.is_contiguous():
        x = x.contiguous()

    rows, cols = x.shape
    block_size = triton.next_power_of_2(cols)
    if block_size > 65_536:
        raise ValueError("This educational kernel supports at most 65,536 columns.")

    num_warps = 4
    if block_size >= 2_048:
        num_warps = 8
    if block_size >= 8_192:
        num_warps = 16

    output = torch.empty_like(x)
    _softmax_kernel[(rows,)](
        output,
        x,
        x.stride(0),
        output.stride(0),
        n_cols=cols,
        block_size=block_size,
        num_warps=num_warps,
    )
    return output

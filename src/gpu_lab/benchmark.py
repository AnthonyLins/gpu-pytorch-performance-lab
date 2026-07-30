"""Correctness checks and synchronized CUDA latency measurements."""

from __future__ import annotations

import statistics
from collections.abc import Callable, Iterable

import numpy as np
import torch


def validate_against_reference(
    implementation: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, float]:
    """Compare an implementation with float32 torch.softmax."""
    reference = torch.softmax(x.float(), dim=1)
    actual = implementation(x).float()
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    absolute = (actual - reference).abs()
    relative = absolute / reference.abs().clamp_min(1e-12)
    return {
        "max_absolute_error": float(absolute.max().item()),
        "max_relative_error": float(relative.max().item()),
    }


def benchmark_cuda(
    implementation: Callable[[torch.Tensor], torch.Tensor],
    x: torch.Tensor,
    *,
    warmup: int = 25,
    repetitions: int = 100,
) -> dict[str, float]:
    """Measure steady-state CUDA latency using CUDA events."""
    if not x.is_cuda:
        raise ValueError("CUDA timing requires a CUDA tensor.")
    if warmup < 1 or repetitions < 2:
        raise ValueError("Use at least one warm-up and two measured repetitions.")

    for _ in range(warmup):
        implementation(x)
    torch.cuda.synchronize()

    samples: list[float] = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        implementation(x)
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)))

    percentiles = np.percentile(samples, [10, 50, 90])
    return {
        "latency_p10_ms": float(percentiles[0]),
        "latency_median_ms": float(percentiles[1]),
        "latency_p90_ms": float(percentiles[2]),
        "latency_mean_ms": float(statistics.fmean(samples)),
        "latency_stdev_ms": float(statistics.stdev(samples)),
    }


def estimate_effective_bandwidth_gbps(
    shape: Iterable[int], element_size: int, latency_ms: float
) -> float:
    """Estimate bandwidth assuming one input read and one output write."""
    elements = int(np.prod(tuple(shape)))
    transferred_bytes = 2 * elements * element_size
    return float(transferred_bytes / (latency_ms * 1e-3) / 1e9)

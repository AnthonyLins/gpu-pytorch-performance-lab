import pytest
import torch

from gpu_lab.softmax import eager_softmax, naive_softmax, triton_softmax


def test_naive_softmax_is_stable_for_large_values() -> None:
    x = torch.tensor([[10_000.0, 10_001.0, 9_999.0]])
    actual = naive_softmax(x)
    expected = eager_softmax(x)
    torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual.sum(dim=1), torch.ones(1))


def test_triton_rejects_cpu_tensor() -> None:
    with pytest.raises((ValueError, RuntimeError)):
        triton_softmax(torch.randn(4, 16))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_triton_matches_pytorch_on_cuda() -> None:
    x = torch.randn(128, 1024, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(
        triton_softmax(x).float(),
        eager_softmax(x).float(),
        atol=1e-3,
        rtol=1e-3,
    )

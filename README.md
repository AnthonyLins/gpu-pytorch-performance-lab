# GPU and PyTorch Performance Lab

A compact, reproducible study of GPU performance engineering for machine learning. It compares a numerically stable row-wise softmax implemented with:

1. PyTorch eager execution;
2. `torch.compile`;
3. a custom Triton kernel.

The project is designed for Google Colab with an NVIDIA GPU. It emphasizes correct benchmarking, numerical validation, profiler evidence, and cautious interpretation rather than headline speedup numbers.

## Why softmax?

Softmax is simple enough to inspect but technically meaningful:

- it requires a row-wise reduction;
- it is sensitive to numerical overflow;
- intermediate reads and writes make the naive implementation bandwidth-heavy;
- kernel fusion can reduce global-memory traffic;
- it appears in attention mechanisms used by modern neural networks.

## Technical questions

- When does `torch.compile` reduce Python and kernel-launch overhead?
- For which matrix shapes does a fused Triton implementation help?
- Is the workload limited by memory bandwidth or arithmetic throughput?
- How do warm-up, synchronization, and compilation affect measurements?
- Are the implementations numerically equivalent within a stated tolerance?

## Open in Google Colab

After publishing the repository, replace `YOUR-USERNAME` and use:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-USERNAME/gpu-pytorch-performance-lab/blob/main/notebooks/softmax_benchmark_colab.ipynb)

The notebook installs the package, detects the GPU, validates the implementations, runs benchmarks, and exports a Chrome trace from PyTorch Profiler.

## Repository structure

```text
gpu-pytorch-performance-lab/
├── notebooks/
│   └── softmax_benchmark_colab.ipynb
├── src/gpu_lab/
│   ├── softmax.py
│   └── benchmark.py
├── tests/
│   └── test_softmax.py
├── results/
│   └── README.md
├── requirements.txt
└── pyproject.toml
```

## Benchmarking principles

GPU execution is asynchronous. Wall-clock timing without synchronization can measure kernel dispatch rather than completion. This repository therefore:

- performs warm-up iterations;
- synchronizes the device around timing boundaries;
- reports median and percentile latency;
- separates compilation from steady-state execution;
- validates output before benchmarking;
- records the GPU, software versions, dtype, and shape;
- avoids generalizing results beyond the tested environment.

## Metrics

- median latency in milliseconds;
- p10 and p90 latency;
- effective memory bandwidth;
- speedup relative to PyTorch eager;
- maximum absolute and relative numerical error.

Effective bandwidth is a simplified estimate based on bytes read and written. It is useful for comparisons but is not a substitute for hardware-counter analysis in Nsight Compute.

## Profiler evidence

The Colab notebook exports a PyTorch Profiler Chrome trace. Open the JSON trace with:

- `chrome://tracing`, or
- Perfetto at https://ui.perfetto.dev/

For deeper local analysis, use Nsight Systems to inspect launches and CPU/GPU overlap, and Nsight Compute to inspect memory throughput, occupancy, and instruction-level behavior.

## Expected interpretation

Do not assume the custom kernel will always win. Results depend on:

- GPU architecture;
- PyTorch and Triton versions;
- dtype;
- row width;
- number of rows;
- compiler behavior;
- cache state;
- compilation mode.

Small tensors may be dominated by launch overhead. Large rows may exceed the practical limits of a single fused program and require a different strategy.

## Author

Anthony José da Cunha Carneiro Lins, PhD  
Background in CUDA/C++, multi-GPU optimization, applied AI, PyTorch, TensorFlow, computer vision, and AI systems  
ORCID: [0000-0002-7153-841X](https://orcid.org/0000-0002-7153-841X)

## License

MIT

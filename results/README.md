# Benchmark results

Generated CSV files and profiler traces belong in this directory and are ignored by default because they depend on the runtime GPU and software versions.

When publishing a result, document:

- UTC timestamp;
- GPU model and compute capability;
- PyTorch, CUDA, and Triton versions;
- input shape and dtype;
- warm-up and repetition counts;
- compilation mode;
- latency distribution;
- correctness tolerance;
- known limitations.

Do not publish a single speedup value without the corresponding environment and workload.

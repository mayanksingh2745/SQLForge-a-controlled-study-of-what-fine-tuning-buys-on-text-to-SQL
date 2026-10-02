"""Optimization layer: Quantization, hardware profiling, and latency benchmarks.

Planned for future steps:
- AWQ and GGUF quantization wrappers.
- Inference latency benchmarking harnesses (p50, p95, throughput tokens/sec).
- VRAM memory footprint profilers.
"""

from typing import Protocol


class ModelOptimizer(Protocol):
    """Protocol for post-training quantization and model optimization."""

    def optimize(self, model_path: str, format_type: str) -> str:
        """Quantize model weights and export optimized artifact."""
        ...

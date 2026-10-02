"""Training layer: SFT trainers, LoRA/QLoRA configuration, and checkpointing.

Planned for future steps:
- SFTTrainer integration with Hugging Face PEFT and TRL.
- Quantized BitsAndBytes 4-bit loading pipelines.
- Learning rate schedulers, gradient checkpointing, and evaluation loss tracking.
"""

from pathlib import Path
from typing import Protocol

from sqlforge.schemas.experiments import ExperimentConfig


class FineTuningEngine(Protocol):
    """Protocol for executing model training runs."""

    def train(self, config: ExperimentConfig) -> Path:
        """Execute fine-tuning and return checkpoint directory path."""
        ...

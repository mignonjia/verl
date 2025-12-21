from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, ProcessorMixin

from verl.utils.dataset.rl_dataset import RLHFDataset


@dataclass
class HAPODatasetConfig:
    """
    Lightweight dataclass wrapper to document extra fields expected by HAPO.

    The underlying Parquet / JSONL rows should contain at least:
      - question_id: str
      - question: str
      - gold_answer: str
      - teacher_cots: List[str]

    These fields are passed through untouched and later consumed in the trainer.
    """

    prompt_key: str = "prompt"


class HAPODataset(RLHFDataset):
    """
    A thin wrapper around `RLHFDataset` that ensures HAPO-specific fields are
    available in each sample.

    Notes:
    - Tokenization / chat-templating is fully delegated to `RLHFDataset`.
    - Any extra columns present in the underlying data (e.g. `question`,
      `gold_answer`, `teacher_cots`) are preserved and appear in the
      `non_tensor_batch` of `DataProto` after collation.
    - The VERL trainer will access these fields to build hindsight prompts
      and compute correctness.
    """

    def __init__(
        self,
        data_files: str | List[str],
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        config: DictConfig,
        max_samples: int = -1,
    ):
        # Reuse RLHFDataset behaviour; we just keep the interface explicit.
        super().__init__(
            data_files=data_files,
            tokenizer=tokenizer,
            processor=processor,
            config=config,
            max_samples=max_samples,
        )

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = super().__getitem__(idx)

        # Sanity: make sure required HAPO fields are present, but don't fail hard;
        # missing fields will be caught later in the trainer.
        for key in ("question_id", "question", "gold_answer", "teacher_cots"):
            row.setdefault(key, None)

        return row


def get_hapo_dataset_cls() -> type[Dataset]:
    """
    Small helper to be used by Hydra configs, if desired.

    Example in config:
      data:
        custom_cls:
          path: hapo/data.py
          name: HAPODataset
    """

    return HAPODataset

from dataclasses import dataclass
from typing import Any, Protocol, Sequence

import numpy as np

from config import GenerateConfig


@dataclass
class PolicyRuntime:
    model: Any
    resize_size: Any
    processor: Any = None
    action_head: Any = None
    proprio_projector: Any = None
    noisy_action_projector: Any = None


class PolicyAdapter(Protocol):
    name: str

    def initialize(self, cfg: GenerateConfig) -> PolicyRuntime:
        ...

    def predict_actions(
        self,
        cfg: GenerateConfig,
        runtime: PolicyRuntime,
        observation: dict,
        task_description: str,
    ) -> Sequence[np.ndarray]:
        ...

    def postprocess_action(self, cfg: GenerateConfig, action: np.ndarray) -> np.ndarray:
        ...

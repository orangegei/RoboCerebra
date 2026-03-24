#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper for running RoboCerebra evaluation with Pi0.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus

from config import GenerateConfig
from eval_policy import run_evaluation


@dataclass
class Pi0GenerateConfig(GenerateConfig):
    model_family: str = "pi0"
    pretrained_checkpoint: Union[str, Path] | None = None
    openpi_config_name: str = "pi0_libero"


@draccus.wrap()
def eval_pi0(cfg: Pi0GenerateConfig) -> float:
    return run_evaluation(cfg)


if __name__ == "__main__":
    eval_pi0()

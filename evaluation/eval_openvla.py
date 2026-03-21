#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Thin wrapper for running RoboCerebra evaluation with OpenVLA.
"""

from dataclasses import dataclass

import draccus

from config import GenerateConfig
from eval_policy import run_evaluation


@dataclass
class OpenVLAGenerateConfig(GenerateConfig):
    model_family: str = "openvla"


@draccus.wrap()
def eval_openvla(cfg: OpenVLAGenerateConfig) -> float:
    return run_evaluation(cfg)


if __name__ == "__main__":
    eval_openvla()

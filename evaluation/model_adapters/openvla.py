from typing import Sequence

import numpy as np

from config import GenerateConfig
from model_adapters.base import PolicyRuntime


class OpenVLAAdapter:
    name = "openvla"

    def initialize(self, cfg: GenerateConfig) -> PolicyRuntime:
        from experiments.robot.openvla_utils import (
            get_action_head,
            get_noisy_action_projector,
            get_processor,
            get_proprio_projector,
        )
        from experiments.robot.robot_utils import get_image_resize_size, get_model

        model = get_model(cfg)
        proprio_projector = (
            get_proprio_projector(cfg, model.llm_dim, proprio_dim=8) if cfg.use_proprio else None
        )
        action_head = (
            get_action_head(cfg, model.llm_dim) if (cfg.use_l1_regression or cfg.use_diffusion) else None
        )
        noisy_action_projector = (
            get_noisy_action_projector(cfg, model.llm_dim) if cfg.use_diffusion else None
        )
        processor = get_processor(cfg)

        unnorm_key = cfg.task_suite_name
        if unnorm_key not in model.norm_stats and f"{unnorm_key}_no_noops" in model.norm_stats:
            unnorm_key = f"{unnorm_key}_no_noops"
        assert unnorm_key in model.norm_stats, f"Action un-norm key {unnorm_key} not found!"
        cfg.unnorm_key = unnorm_key

        return PolicyRuntime(
            model=model,
            resize_size=get_image_resize_size(cfg),
            processor=processor,
            action_head=action_head,
            proprio_projector=proprio_projector,
            noisy_action_projector=noisy_action_projector,
        )

    def predict_actions(
        self,
        cfg: GenerateConfig,
        runtime: PolicyRuntime,
        observation: dict,
        task_description: str,
    ) -> Sequence[np.ndarray]:
        from experiments.robot.robot_utils import get_action

        return get_action(
            cfg,
            runtime.model,
            observation,
            task_description,
            runtime.processor,
            runtime.action_head,
            runtime.proprio_projector,
            runtime.noisy_action_projector,
            use_film=cfg.use_film,
        )

    def postprocess_action(self, cfg: GenerateConfig, action: np.ndarray) -> np.ndarray:
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action

        del cfg
        action = normalize_gripper_action(action, binarize=True)
        return invert_gripper_action(action)

from typing import Sequence

import numpy as np

from config import GenerateConfig
from model_adapters.base import PolicyRuntime


class Pi0Adapter:
    name = "pi0"
    _DEFAULT_OPENPI_CONFIG = "pi0_libero"
    _DEFAULT_RESIZE_SIZE = 224

    def _get_openpi_config_name(self, cfg: GenerateConfig) -> str:
        return getattr(cfg, "openpi_config_name", self._DEFAULT_OPENPI_CONFIG)

    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape={image.shape}")

        if image.shape[0] == 3 and image.shape[-1] != 3:
            image = np.transpose(image, (1, 2, 0))

        if np.issubdtype(image.dtype, np.floating):
            if image.max() <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        elif image.dtype != np.uint8:
            image = image.astype(np.uint8)

        return image

    def initialize(self, cfg: GenerateConfig) -> PolicyRuntime:
        from openpi.policies import policy_config
        from openpi.training import config as openpi_config

        if cfg.pretrained_checkpoint is None:
            raise ValueError("pretrained_checkpoint must not be None for pi0 evaluation.")

        config_name = self._get_openpi_config_name(cfg)
        train_config = openpi_config.get_config(config_name)
        policy = policy_config.create_trained_policy(train_config, str(cfg.pretrained_checkpoint))

        return PolicyRuntime(
            model=policy,
            resize_size=self._DEFAULT_RESIZE_SIZE,
            processor={"openpi_config_name": config_name},
        )

    def predict_actions(
        self,
        cfg: GenerateConfig,
        runtime: PolicyRuntime,
        observation: dict,
        task_description: str,
    ) -> Sequence[np.ndarray]:
        policy = runtime.model

        example = {
            "observation/image": self._normalize_image(observation["full_image"]),
            "observation/wrist_image": self._normalize_image(observation["wrist_image"]),
            "observation/state": np.asarray(observation["state"], dtype=np.float32),
            "prompt": task_description,
        }

        outputs = policy.infer(example)
        actions = np.asarray(outputs["actions"], dtype=np.float32)
        if actions.ndim == 1:
            actions = actions[None, :]

        n = min(cfg.num_open_loop_steps, actions.shape[0])
        return [actions[i].copy() for i in range(n)]

    def postprocess_action(self, cfg: GenerateConfig, action: np.ndarray) -> np.ndarray:
        del cfg
        # openpi policy outputs are already unnormalized; keep only gripper binarization here.
        action = np.asarray(action, dtype=np.float32).copy()
        action[-1] = 1.0 if action[-1] >= 0.0 else -1.0
        return action

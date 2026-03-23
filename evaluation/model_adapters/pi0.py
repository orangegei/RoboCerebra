from typing import Sequence

import numpy as np
import torch

from config import GenerateConfig
from model_adapters.base import PolicyRuntime


class Pi0Adapter:
    name = "pi0"

    def initialize(self, cfg: GenerateConfig) -> PolicyRuntime:
        from lerobot.policies.pi0 import PI0Policy
        from lerobot.policies.factory import make_pre_post_processors

        model_id = str(cfg.pretrained_checkpoint)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        policy = PI0Policy.from_pretrained(model_id).to(device).eval()
        local_paligemma = "/seu_nvme/home/linli/213221090/MyModels/google/paligemma-3b-pt-224"
        preprocess, postprocess = make_pre_post_processors(
            policy.config,
            model_id,
            preprocessor_overrides={
                "device_processor": {"device": str(device)},
                "tokenizer_processor": {"tokenizer_name": local_paligemma},
            },
        )

        return PolicyRuntime(
            model=policy,
            resize_size=policy.config.image_resolution[0],
            processor=preprocess,
            action_head=postprocess,
        )

    def predict_actions(
        self,
        cfg: GenerateConfig,
        runtime: PolicyRuntime,
        observation: dict,
        task_description: str,
    ) -> Sequence[np.ndarray]:
        policy = runtime.model
        preprocess = runtime.processor

        # Build observation dict matching lerobot dataset format
        frame = {
            "observation.images.image": torch.from_numpy(observation["full_image"]).permute(2, 0, 1).float() / 255.0,
            "observation.images.image2": torch.from_numpy(observation["wrist_image"]).permute(2, 0, 1).float() / 255.0,
            "observation.state": torch.from_numpy(observation["state"]).float(),
            "task": task_description,
        }

        batch = preprocess(frame)

        with torch.inference_mode():
            # predict_action_chunk returns (1, chunk_size, action_dim)
            actions = policy.predict_action_chunk(batch)

        postprocess = runtime.action_head
        actions = postprocess(actions)

        # Slice to num_open_loop_steps and convert to list of numpy arrays
        n = cfg.num_open_loop_steps
        actions_np = actions[0, :n].cpu().numpy()
        return [actions_np[i] for i in range(actions_np.shape[0])]

    def postprocess_action(self, cfg: GenerateConfig, action: np.ndarray) -> np.ndarray:
        # Pi0 postprocessor already denormalizes actions.
        # Binarize gripper: last dim is gripper, threshold at 0.5
        action = action.copy()
        action[-1] = 1.0 if action[-1] > 0.5 else -1.0
        return action

from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from PIL import Image

from config import GenerateConfig


@dataclass
class VLMRuntime:
    model: Any
    processor: Any
    device: str


def _to_pil_image(image: np.ndarray) -> Image.Image:
    if isinstance(image, Image.Image):
        return image

    if not isinstance(image, np.ndarray):
        raise TypeError(f"Unsupported image type: {type(image)!r}")

    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)

    return Image.fromarray(image)


def initialize(cfg: GenerateConfig) -> Optional[VLMRuntime]:
    if not cfg.use_vlm_desc:
        return None
    if not cfg.vlm_model_path_or_name:
        raise ValueError("cfg.vlm_model_path_or_name must be set when cfg.use_vlm_desc=True")

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    processor = AutoProcessor.from_pretrained(cfg.vlm_model_path_or_name, trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        cfg.vlm_model_path_or_name,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    return VLMRuntime(model=model, processor=processor, device=device)


def generate_description(
    cfg: GenerateConfig,
    runtime: VLMRuntime,
    observation: dict,
    image: np.ndarray | None = None,
) -> str:
    if runtime is None:
        raise ValueError("VLM runtime is not initialized")

    import torch

    prompt = cfg.vlm_prompt.strip() or "Describe the robot's current task-relevant scene as a short action instruction."

    primary_image = image if image is not None else observation["full_image"]
    images = [_to_pil_image(primary_image)]
    if cfg.vlm_use_wrist_image:
        images.append(_to_pil_image(observation["wrist_image"]))

    messages = [
        {
            "role": "user",
            "content": [{"type": "image"} for _ in images] + [{"type": "text", "text": prompt}],
        }
    ]
    text = runtime.processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = runtime.processor(images=images, text=text, return_tensors="pt")
    inputs = {key: value.to(runtime.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.no_grad():
        generated_ids = runtime.model.generate(**inputs, max_new_tokens=cfg.vlm_max_new_tokens)

    prompt_length = inputs["input_ids"].shape[1]
    generated_text = runtime.processor.batch_decode(
        generated_ids[:, prompt_length:], skip_special_tokens=True
    )[0].strip()
    return generated_text

from typing import Union
import numpy as np
from PIL import Image

def _to_uint8_image(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0, 255], supporting float/uint8 inputs."""
    if image.dtype == np.uint8:
        return image
    if np.issubdtype(image.dtype, np.floating):
        # assume float in [0,1] or [0,255]
        img = image.copy()
        if img.max() <= 1.0:
            img = img * 255.0
        img = np.clip(img, 0, 255).astype(np.uint8)
        return img
    return np.clip(image, 0, 255).astype(np.uint8)


def save_image(image: np.ndarray, path: Union[str, "os.PathLike[str]"]) -> None:
    """Saves a numpy array as an image file.
    image = obs["agentview_image"]
    image = image[::-1, ::-1]

    输入:
      image: np.ndarray, 形状(H, W, 3)或(H, W), dtype可为float/uint8
      path: str 或 PathLike

    输出:
      None（将图像写入磁盘）
    """
    img = _to_uint8_image(image)
    Image.fromarray(img).save(path)


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img
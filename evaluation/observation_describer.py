"""Backward-compatible re-exports for legacy observation description code.

当前模块只保留旧入口，实际实现已经迁移到 `evaluation.vlm_planner`。
"""

try:
    from .vlm_planner import VLMRuntime, generate_description, initialize
except ImportError:
    from vlm_planner import VLMRuntime, generate_description, initialize

__all__ = [
    "VLMRuntime",
    "generate_description",
    "initialize",
]

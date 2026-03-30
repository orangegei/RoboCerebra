import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional


@dataclass
class SubtaskPlanningResult:
    """第一阶段子任务规划结果。

    字段说明：
    - candidate_subtasks: VLM 给出的候选子任务列表
    - selected_subtask_index: 被选中的候选下标，允许为 None
    - selected_subtask_description: 被选中的候选文本，允许为 None

    示例：
        SubtaskPlanningResult(
            candidate_subtasks=[
                "pick cream_cheese_1",
                "pick popcorn_1",
            ],
            selected_subtask_index=0,
            selected_subtask_description="pick cream_cheese_1",
        )
    """
    candidate_subtasks: List[str]
    selected_subtask_index: Optional[int]
    selected_subtask_description: Optional[str]


@dataclass
class ActionPlanningResult:
    """第二阶段动作规划结果。

    字段说明：
    - candidate_actions: VLM 给出的候选动作列表
    - selected_action_index: 被选中的候选下标，允许为 None
    - selected_action_description: 被选中的候选文本，允许为 None

    示例：
        ActionPlanningResult(
            candidate_actions=[
                "move gripper above cream_cheese_1",
                "grasp cream_cheese_1",
            ],
            selected_action_index=0,
            selected_action_description="move gripper above cream_cheese_1",
        )
    """
    candidate_actions: List[str]
    selected_action_index: Optional[int]
    selected_action_description: Optional[str]


def _normalize_text(value: str) -> str:
    """将文本归一化成适合比较和落盘的形式。

    主要做两件事：
    - 去掉首尾空白
    - 把连续空白折叠成单个空格

    输入示例：
        "  grasp   cream_cheese_1  "

    输出示例：
        "grasp cream_cheese_1"
    """
    return " ".join(value.split()).strip()


def _normalize_candidate_list(raw_value: Any, field_name: str) -> List[str]:
    """校验并归一化候选列表字段。

    这个函数用于处理 `candidate_subtasks` 或 `candidate_actions`：
    - 必须是 list
    - 每个元素必须是非空字符串
    - 每个元素会经过 `_normalize_text()` 处理
    - 列表不能为空

    输入示例：
        raw_value = [" move above object ", "grasp object"]
        field_name = "candidate_actions"

    输出示例：
        ["move above object", "grasp object"]
    """
    if not isinstance(raw_value, list):
        raise ValueError(f"{field_name} must be a list, got {type(raw_value)!r}")

    normalized: List[str] = []
    for item in raw_value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} items must be strings, got {type(item)!r}")
        text = _normalize_text(item)
        if not text:
            raise ValueError(f"{field_name} items must be non-empty strings")
        normalized.append(text)

    if not normalized:
        raise ValueError(f"{field_name} must not be empty")
    return normalized


def _normalize_optional_index(raw_value: Any, field_name: str) -> Optional[int]:
    """校验可选的索引字段。

    允许的输入只有两类：
    - int
    - None

    不接受 bool，因为在 Python 里 `bool` 是 `int` 的子类，
    如果不单独拦截，`True/False` 会被误当成 1/0。

    输入示例：
        raw_value = 1

    输出示例：
        1
    """
    if raw_value is None:
        return None
    if isinstance(raw_value, bool) or not isinstance(raw_value, int):
        raise ValueError(f"{field_name} must be an integer or null, got {type(raw_value)!r}")
    return raw_value


def _normalize_optional_description(raw_value: Any, field_name: str) -> Optional[str]:
    """校验可选的描述字段，并做文本归一化。

    允许的输入只有两类：
    - str
    - None

    如果传入字符串，会去掉多余空白，并要求结果不能为空。

    输入示例：
        raw_value = "  grasp cream_cheese_1 "

    输出示例：
        "grasp cream_cheese_1"
    """
    if raw_value is None:
        return None
    if not isinstance(raw_value, str):
        raise ValueError(f"{field_name} must be a string or null, got {type(raw_value)!r}")
    description = _normalize_text(raw_value)
    if not description:
        raise ValueError(f"{field_name} must be a non-empty string when provided")
    return description


def _resolve_selection(
    candidates: List[str],
    selected_index: Optional[int],
    selected_description: Optional[str],
    index_field_name: str,
    description_field_name: str,
) -> tuple[Optional[int], Optional[str]]:
    """统一解析“选中项”的 index 和 description。

    这个函数负责把以下几种情况整理成一致结果：
    - 只给了 index，则自动补 description
    - 只给了 description，则自动补 index
    - 两者都给了，则检查是否一致
    - 两者都没给，则保持为 None

    同时，它还会检查：
    - index 是否越界
    - description 是否真的存在于候选列表里

    输入示例：
        candidates = ["move above", "grasp"]
        selected_index = 1
        selected_description = None

    输出示例：
        (1, "grasp")
    """
    if selected_index is not None and not 0 <= selected_index < len(candidates):
        raise ValueError(
            f"{index_field_name}={selected_index} is out of range for {len(candidates)} candidates"
        )

    if selected_description is not None:
        matched_indices = [idx for idx, candidate in enumerate(candidates) if candidate == selected_description]
        if not matched_indices:
            raise ValueError(
                f"{description_field_name} does not match any candidate: {selected_description!r}"
            )
        matched_index = matched_indices[0]
        if selected_index is None:
            selected_index = matched_index
        elif selected_index != matched_index:
            raise ValueError(
                f"{index_field_name}={selected_index} does not match {description_field_name}={selected_description!r}"
            )

    if selected_index is not None and selected_description is None:
        selected_description = candidates[selected_index]

    return selected_index, selected_description


def _extract_json_payload(raw_text: str) -> Dict[str, Any]:
    """从 VLM 原始文本中提取 JSON 对象。

    支持三种常见输入形式：
    1. 纯 JSON 文本
    2. 包在 ```json ... ``` 代码块里的 JSON
    3. 前后带解释文字，但中间包含一个 JSON object

    如果最终提取出来的不是 dict，会报错。

    输入示例：
        raw_text = '''
        下面是结果：
        ```json
        {"candidate_actions": ["move above", "grasp"], "selected_action_index": 0}
        ```
        '''

    输出示例：
        {
            "candidate_actions": ["move above", "grasp"],
            "selected_action_index": 0,
        }
    """
    if not isinstance(raw_text, str):
        raise TypeError(f"raw_text must be a string, got {type(raw_text)!r}")

    text = raw_text.strip()
    if not text:
        raise ValueError("raw_text must not be empty")

    fenced_start = text.find("```")
    if fenced_start != -1:
        fence_end = text.find("```", fenced_start + 3)
        if fence_end != -1:
            fenced_body = text[fenced_start + 3:fence_end].strip()
            if fenced_body.startswith("json"):
                fenced_body = fenced_body[4:].strip()
            text = fenced_body

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        start = text.find("{")
        if start == -1:
            raise ValueError("Failed to find JSON object in raw_text") from None

        depth = 0
        end = None
        in_string = False
        escape = False
        for idx in range(start, len(text)):
            ch = text[idx]
            if in_string:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_string = False
                continue

            if ch == '"':
                in_string = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = idx + 1
                    break

        if end is None:
            raise ValueError("Failed to extract a complete JSON object from raw_text") from None

        payload = json.loads(text[start:end])

    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object, got {type(payload)!r}")
    return payload


def parse_subtask_planning_output(raw_text: str) -> SubtaskPlanningResult:
    """解析第一阶段 VLM 输出，得到稳定的子任务规划结果。

    期望输入 JSON 至少包含：
    - `candidate_subtasks`

    可选包含：
    - `selected_subtask_index`
    - `selected_subtask_description`

    典型输入示例：
        {
          "candidate_subtasks": [
            "pick cream_cheese_1",
            "pick popcorn_1"
          ],
          "selected_subtask_index": 0
        }

    典型输出示例：
        SubtaskPlanningResult(
            candidate_subtasks=["pick cream_cheese_1", "pick popcorn_1"],
            selected_subtask_index=0,
            selected_subtask_description="pick cream_cheese_1",
        )
    """
    payload = _extract_json_payload(raw_text)
    candidate_subtasks = _normalize_candidate_list(payload.get("candidate_subtasks"), "candidate_subtasks")
    selected_subtask_index = _normalize_optional_index(
        payload.get("selected_subtask_index"),
        "selected_subtask_index",
    )
    selected_subtask_description = _normalize_optional_description(
        payload.get("selected_subtask_description"),
        "selected_subtask_description",
    )
    selected_subtask_index, selected_subtask_description = _resolve_selection(
        candidates=candidate_subtasks,
        selected_index=selected_subtask_index,
        selected_description=selected_subtask_description,
        index_field_name="selected_subtask_index",
        description_field_name="selected_subtask_description",
    )
    return SubtaskPlanningResult(
        candidate_subtasks=candidate_subtasks,
        selected_subtask_index=selected_subtask_index,
        selected_subtask_description=selected_subtask_description,
    )


def parse_action_planning_output(raw_text: str) -> ActionPlanningResult:
    """解析第二阶段 VLM 输出，得到稳定的动作规划结果。

    期望输入 JSON 至少包含：
    - `candidate_actions`

    可选包含：
    - `selected_action_index`
    - `selected_action_description`

    典型输入示例：
        {
          "candidate_actions": [
            "move gripper above cream_cheese_1",
            "grasp cream_cheese_1"
          ],
          "selected_action_description": "grasp cream_cheese_1"
        }

    典型输出示例：
        ActionPlanningResult(
            candidate_actions=[
                "move gripper above cream_cheese_1",
                "grasp cream_cheese_1",
            ],
            selected_action_index=1,
            selected_action_description="grasp cream_cheese_1",
        )
    """
    payload = _extract_json_payload(raw_text)
    candidate_actions = _normalize_candidate_list(payload.get("candidate_actions"), "candidate_actions")
    selected_action_index = _normalize_optional_index(
        payload.get("selected_action_index"),
        "selected_action_index",
    )
    selected_action_description = _normalize_optional_description(
        payload.get("selected_action_description"),
        "selected_action_description",
    )
    selected_action_index, selected_action_description = _resolve_selection(
        candidates=candidate_actions,
        selected_index=selected_action_index,
        selected_description=selected_action_description,
        index_field_name="selected_action_index",
        description_field_name="selected_action_description",
    )
    return ActionPlanningResult(
        candidate_actions=candidate_actions,
        selected_action_index=selected_action_index,
        selected_action_description=selected_action_description,
    )


__all__ = [
    "ActionPlanningResult",
    "SubtaskPlanningResult",
    "parse_action_planning_output",
    "parse_subtask_planning_output",
]

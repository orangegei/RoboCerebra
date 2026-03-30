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


def _require_non_empty_text(value: Any, field_name: str) -> str:
    """校验必填文本字段，并返回归一化后的字符串。

    这个函数主要用于 prompt builder 的输入校验，确保像
    `language_instruction`、`formal_goal`、`selected_subtask_description`
    这些字段在构建 prompt 前就是合法的。

    输入示例：
        value = "  grasp cream_cheese_1 "
        field_name = "selected_subtask_description"

    输出示例：
        "grasp cream_cheese_1"
    """
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value)!r}")

    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _normalize_task_tree(task_tree: Any) -> Dict[str, Any]:
    """校验任务树对象，并转成稳定的 JSON 字符串输入。

    这里要求 `task_tree` 必须是 dict，因为后续会直接把它序列化进 prompt，
    作为 VLM 的历史上下文。

    输入示例：
        task_tree = {"root": {"children": []}}

    输出示例：
        {"root": {"children": []}}
    """
    if not isinstance(task_tree, dict):
        raise TypeError(f"task_tree must be a dict, got {type(task_tree)!r}")
    return task_tree


def _serialize_task_tree_for_prompt(task_tree: Dict[str, Any]) -> str:
    """将任务树序列化成适合拼接到 prompt 中的 JSON 字符串。

    使用 `ensure_ascii=False` 保留原始文本，可读性更好；
    使用 `indent=2` 让模型更容易看清层级结构。

    输入示例：
        {"root": {"children": []}}

    输出示例：
        '{\\n  "root": {\\n    "children": []\\n  }\\n}'
    """
    return json.dumps(task_tree, ensure_ascii=False, indent=2)


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


def build_subtask_planning_prompt(
    current_image: Any,
    language_instruction: str,
    formal_goal: str,
    task_tree: Dict[str, Any],
) -> str:
    """构建第一阶段子任务规划 prompt。

    这个 prompt 的目标是让 VLM 基于：
    - 当前图像
    - 全局任务指令
    - formal goal
    - 当前任务树 JSON

    输出最小 schema 的严格 JSON，只包含：
    - `candidate_subtasks`
    - `selected_subtask_index`
    - `selected_subtask_description`

    注意：
    - `current_image` 在这里不会被序列化进文本 prompt
    - 它存在的意义是保持接口和未来多模态调用一致
    - 真正调用模型时，应将当前图像与本 prompt 一起发送

    输入示例：
        current_image = <numpy image>
        language_instruction = "Organize selected food items into the white_storage_box"
        formal_goal = "(And ...)"
        task_tree = {"root": {"children": []}}

    输出示例：
        一个字符串 prompt，要求模型只返回如下格式：
        {
          "candidate_subtasks": ["...", "..."],
          "selected_subtask_index": 0,
          "selected_subtask_description": "..."
        }
    """
    del current_image

    normalized_instruction = _require_non_empty_text(language_instruction, "language_instruction")
    normalized_goal = _require_non_empty_text(formal_goal, "formal_goal")
    normalized_task_tree = _normalize_task_tree(task_tree)
    task_tree_json = _serialize_task_tree_for_prompt(normalized_task_tree)

    return (
        "You are a robotic task planner. You will be given the current image together with this prompt.\n"
        "Your job in this stage is to propose candidate subtasks and select exactly one subtask.\n"
        "Use only the current image, the global instruction, the formal goal, and the current planning tree.\n"
        "Return strict JSON only. Do not add any explanation, markdown, code fences, or extra fields.\n"
        "The JSON object must contain exactly these keys:\n"
        '- "candidate_subtasks": a non-empty list of short strings\n'
        '- "selected_subtask_index": an integer index into candidate_subtasks, or null\n'
        '- "selected_subtask_description": the selected string from candidate_subtasks, or null\n'
        "The selected description must match one item in candidate_subtasks.\n"
        "Keep the output minimal and task-relevant.\n\n"
        f"language_instruction:\n{normalized_instruction}\n\n"
        f"formal_goal:\n{normalized_goal}\n\n"
        "current_planning_tree_json:\n"
        f"{task_tree_json}\n\n"
        "Output strict JSON only."
    )


def build_action_planning_prompt(
    current_image: Any,
    language_instruction: str,
    formal_goal: str,
    task_tree: Dict[str, Any],
    selected_subtask_description: str,
) -> str:
    """构建第二阶段动作规划 prompt。

    这个 prompt 的目标是让 VLM 基于：
    - 当前图像
    - 全局任务指令
    - formal goal
    - 当前任务树 JSON
    - 当前已选 subtask

    输出最小 schema 的严格 JSON，只包含：
    - `candidate_actions`
    - `selected_action_index`
    - `selected_action_description`

    注意：
    - `current_image` 在这里不会被序列化进文本 prompt
    - 真正调用模型时，应将当前图像与本 prompt 一起发送
    - 动作应更细粒度，能直接作为后续 VLA 的 desc 候选

    输入示例：
        current_image = <numpy image>
        language_instruction = "Organize selected food items into the white_storage_box"
        formal_goal = "(And ...)"
        task_tree = {"root": {"children": []}}
        selected_subtask_description = "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1"

    输出示例：
        一个字符串 prompt，要求模型只返回如下格式：
        {
          "candidate_actions": ["...", "..."],
          "selected_action_index": 0,
          "selected_action_description": "..."
        }
    """
    del current_image

    normalized_instruction = _require_non_empty_text(language_instruction, "language_instruction")
    normalized_goal = _require_non_empty_text(formal_goal, "formal_goal")
    normalized_task_tree = _normalize_task_tree(task_tree)
    normalized_subtask = _require_non_empty_text(
        selected_subtask_description,
        "selected_subtask_description",
    )
    task_tree_json = _serialize_task_tree_for_prompt(normalized_task_tree)

    return (
        "You are a robotic task planner. You will be given the current image together with this prompt.\n"
        "Your job in this stage is to propose candidate atomic actions for the selected subtask and select exactly one action.\n"
        "Use only the current image, the global instruction, the formal goal, the current planning tree, and the selected subtask.\n"
        "Return strict JSON only. Do not add any explanation, markdown, code fences, or extra fields.\n"
        "The JSON object must contain exactly these keys:\n"
        '- "candidate_actions": a non-empty list of short strings\n'
        '- "selected_action_index": an integer index into candidate_actions, or null\n'
        '- "selected_action_description": the selected string from candidate_actions, or null\n'
        "The selected description must match one item in candidate_actions.\n"
        "Keep the output minimal, concrete, and directly executable as a short VLA description.\n\n"
        f"language_instruction:\n{normalized_instruction}\n\n"
        f"formal_goal:\n{normalized_goal}\n\n"
        f"selected_subtask:\n{normalized_subtask}\n\n"
        "current_planning_tree_json:\n"
        f"{task_tree_json}\n\n"
        "Output strict JSON only."
    )


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
    "build_action_planning_prompt",
    "build_subtask_planning_prompt",
    "parse_action_planning_output",
    "parse_subtask_planning_output",
]

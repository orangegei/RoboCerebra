import json
import re
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List


@dataclass
class TaskPlanRootNode:
    node_id: str = "root"
    node_type: str = "task"
    description: str = ""
    goal_summary: List[str] = field(default_factory=list)
    children: List[Dict[str, Any]] = field(default_factory=list)
    history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class TaskPlanSchema:
    task_id: str = ""
    language_instruction: str = ""
    formal_goal: str = ""
    root: TaskPlanRootNode = field(default_factory=TaskPlanRootNode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_task_plan_root(
    task_id: str = "",
    language_instruction: str = "",
    formal_goal: str = "",
    description: str = "",
    goal_summary: List[str] | None = None,
) -> Dict[str, Any]:
    """构造一个最小可用的任务树根节点。

    这个函数只负责生成 root-level schema，不解析 BDDL，也不写文件。
    适合在测试里快速造一个空任务树，或者在别处已经拿到结构化信息后手动组装。

    输入示例：
        task_id = "LIBERO_Coffee_Table_Manipulation"
        language_instruction = "Organize selected food items into the white_storage_box"
        formal_goal = "(And ...)"
        description = "Organize selected food items into the white_storage_box"
        goal_summary = ["cream_cheese_1 in white_storage_box_1_bottom_side"]

    输出示例：
        {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "language_instruction": "Organize selected food items into the white_storage_box",
          "formal_goal": "(And ...)",
          "root": {
            "node_id": "root",
            "node_type": "task",
            "description": "Organize selected food items into the white_storage_box",
            "goal_summary": ["cream_cheese_1 in white_storage_box_1_bottom_side"],
            "children": [],
            "history": []
          }
        }
    """
    plan = TaskPlanSchema(
        task_id=task_id,
        language_instruction=language_instruction,
        formal_goal=formal_goal,
        root=TaskPlanRootNode(
            description=description,
            goal_summary=list(goal_summary or []),
        ),
    )
    return plan.to_dict()


def _normalize_text(value: str) -> str:
    return " ".join(value.split()).strip()


def _require_non_empty_string(value: Any, field_name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{field_name} must be a string, got {type(value)!r}")
    normalized = _normalize_text(value)
    if not normalized:
        raise ValueError(f"{field_name} must be a non-empty string")
    return normalized


def _require_list(value: Any, field_name: str) -> List[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a list, got {type(value)!r}")
    return value


def _require_dict(value: Any, field_name: str) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field_name} must be a dict, got {type(value)!r}")
    return value


def _get_root(plan: Dict[str, Any]) -> Dict[str, Any]:
    return _require_dict(plan.get("root"), "plan.root")


def _iter_subtask_nodes(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    root = _get_root(plan)
    children = _require_list(root.get("children"), "plan.root.children")
    return [child for child in children if isinstance(child, dict) and child.get("node_type") == "subtask"]


def _iter_action_nodes(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    actions: List[Dict[str, Any]] = []
    for subtask in _iter_subtask_nodes(plan):
        subtask_children = _require_list(subtask.get("children"), f"{subtask.get('node_id', 'subtask')}.children")
        for child in subtask_children:
            if isinstance(child, dict) and child.get("node_type") == "action":
                actions.append(child)
    return actions


def _find_subtask_node_by_id(plan: Dict[str, Any], subtask_id: str) -> Dict[str, Any]:
    normalized_id = _require_non_empty_string(subtask_id, "subtask_id")
    for node in _iter_subtask_nodes(plan):
        if node.get("node_id") == normalized_id:
            return node
    raise ValueError(f"Subtask node not found: {normalized_id!r}")


def _find_subtask_node_by_description(plan: Dict[str, Any], description: str) -> Dict[str, Any] | None:
    normalized_description = _require_non_empty_string(description, "description")
    for node in _iter_subtask_nodes(plan):
        if _normalize_text(node.get("description", "")) == normalized_description:
            return node
    return None


def _find_action_node_by_description(subtask_node: Dict[str, Any], description: str) -> Dict[str, Any] | None:
    normalized_description = _require_non_empty_string(description, "description")
    children = _require_list(subtask_node.get("children"), f"{subtask_node.get('node_id', 'subtask')}.children")
    for node in children:
        if isinstance(node, dict) and node.get("node_type") == "action":
            if _normalize_text(node.get("description", "")) == normalized_description:
                return node
    return None


def _next_node_id(plan: Dict[str, Any], prefix: str) -> str:
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")
    max_index = 0

    for node in _iter_subtask_nodes(plan) + _iter_action_nodes(plan):
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            continue
        match = pattern.match(node_id)
        if match:
            max_index = max(max_index, int(match.group(1)))

    return f"{prefix}_{max_index + 1:03d}"


def _read_bddl_text(bddl_path: str | Path) -> str:
    return Path(bddl_path).read_text(encoding="utf-8")


def _extract_parenthesized_block(text: str, block_name: str) -> str:
    marker = f"(:{block_name}"
    start = text.find(marker)
    if start == -1:
        raise ValueError(f"Missing BDDL block: :{block_name}")

    depth = 0
    end = None
    for idx in range(start, len(text)):
        ch = text[idx]
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                end = idx + 1
                break

    if end is None:
        raise ValueError(f"Unclosed BDDL block: :{block_name}")
    return text[start:end]


def _normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_bddl_text(bddl_path: str | Path) -> Dict[str, str]:
    """从 BDDL 文本里提取任务树初始化所需的最小字段。

    当前只解析：
    - `task_id`
    - `language_instruction`
    - `formal_goal`

    输入示例：
        bddl_path = "sample_task.bddl"

    输出示例：
        {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "language_instruction": "Organize selected food items into the white_storage_box",
          "formal_goal": "(And ...)"
        }
    """
    text = _read_bddl_text(bddl_path)

    problem_match = re.search(r"\(define\s+\(problem\s+([^\s\)]+)\)", text, flags=re.IGNORECASE)
    if not problem_match:
        raise ValueError("Failed to extract BDDL problem id")

    language_block = _extract_parenthesized_block(text, "language")
    language_match = re.match(r"\(:language\s*(.*?)\)$", _normalize_whitespace(language_block), flags=re.IGNORECASE)
    if not language_match:
        raise ValueError("Failed to extract BDDL language instruction")

    goal_block = _extract_parenthesized_block(text, "goal")
    normalized_goal_block = _normalize_whitespace(goal_block)
    goal_match = re.match(r"\(:goal\s*(.*)\)$", normalized_goal_block, flags=re.IGNORECASE)
    if not goal_match:
        raise ValueError("Failed to extract BDDL goal block")

    return {
        "task_id": problem_match.group(1),
        "language_instruction": language_match.group(1).strip(),
        "formal_goal": goal_match.group(1).strip(),
    }


def _get_robosuite_parse_problem():
    try:
        from libero.libero.envs.bddl_utils import robosuite_parse_problem
        return robosuite_parse_problem
    except ImportError:
        repo_root = Path(__file__).resolve().parent.parent
        libero_src = repo_root / "LIBERO" / "libero"
        if str(libero_src) not in sys.path:
            sys.path.append(str(libero_src))
        from libero.envs.bddl_utils import robosuite_parse_problem
        return robosuite_parse_problem


def parse_bddl_metadata(bddl_path: str | Path) -> Dict[str, Any]:
    """借助 LIBERO 的 BDDL parser 提取结构化元数据。

    这个函数适合用于后续构建任务树 bootstrap 信息，例如 goal summary、
    objects、fixtures、initial_state 等。

    输入示例：
        bddl_path = "sample_task.bddl"

    输出示例：
        {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "fixtures": {...},
          "objects": {...},
          "goal_state": [["In", "cream_cheese_1", "white_storage_box_1_bottom_side"]],
          "language_instruction": "Organize selected food items into the white_storage_box"
        }
    """
    text_info = parse_bddl_text(bddl_path)
    robosuite_parse_problem = _get_robosuite_parse_problem()
    parsed = robosuite_parse_problem(str(bddl_path))

    return {
        "task_id": text_info["task_id"],
        "fixtures": parsed["fixtures"],
        "objects": parsed["objects"],
        "obj_of_interest": parsed["obj_of_interest"],
        "initial_state": parsed["initial_state"],
        "goal_state": parsed["goal_state"],
        "language_instruction": text_info["language_instruction"],
    }


def normalize_goal_predicate(predicate: List[str]) -> str:
    """把单条 goal predicate 规范化成稳定的自然语言短句。

    输入示例：
        predicate = ["In", "cream_cheese_1", "white_storage_box_1_bottom_side"]

    输出示例：
        "cream_cheese_1 in white_storage_box_1_bottom_side"
    """
    if not isinstance(predicate, list):
        raise ValueError(f"Goal predicate must be a list, got {type(predicate)!r}")
    if len(predicate) != 3:
        raise ValueError(f"Goal predicate must contain exactly 3 items, got {len(predicate)}: {predicate!r}")

    relation, subject, target = predicate
    if not all(isinstance(item, str) and item.strip() for item in (relation, subject, target)):
        raise ValueError(f"Goal predicate items must be non-empty strings: {predicate!r}")

    return f"{subject} {relation.lower()} {target}"


def normalize_goal_state(goal_state: List[List[str]]) -> List[str]:
    """把整个 `goal_state` 列表规范化成 `goal_summary` 列表。

    输入示例：
        goal_state = [
            ["In", "cream_cheese_1", "white_storage_box_1_bottom_side"],
            ["In", "popcorn_1", "white_storage_box_1_right_side"],
        ]

    输出示例：
        [
          "cream_cheese_1 in white_storage_box_1_bottom_side",
          "popcorn_1 in white_storage_box_1_right_side",
        ]
    """
    if not isinstance(goal_state, list):
        raise ValueError(f"goal_state must be a list, got {type(goal_state)!r}")
    return [normalize_goal_predicate(predicate) for predicate in goal_state]


def build_task_plan_from_bddl(
    text_info: Dict[str, str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    """根据 BDDL 文本信息和结构化元数据生成初始任务树。

    这是 bootstrap 的核心纯函数：
    - 校验 text info 和 metadata 一致
    - 从 `goal_state` 生成 `root.goal_summary`
    - 初始化空的 `children` 和 `history`

    输入示例：
        text_info = {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "language_instruction": "Organize selected food items into the white_storage_box",
          "formal_goal": "(And ...)"
        }
        metadata = {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "goal_state": [["In", "cream_cheese_1", "white_storage_box_1_bottom_side"]],
          "language_instruction": "Organize selected food items into the white_storage_box"
        }

    输出示例：
        {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "language_instruction": "Organize selected food items into the white_storage_box",
          "formal_goal": "(And ...)",
          "root": {
            "node_id": "root",
            "node_type": "task",
            "description": "Organize selected food items into the white_storage_box",
            "goal_summary": ["cream_cheese_1 in white_storage_box_1_bottom_side"],
            "children": [],
            "history": []
          }
        }
    """
    if text_info["task_id"] != metadata["task_id"]:
        raise ValueError(
            f"Mismatched task ids between text info and metadata: "
            f"{text_info['task_id']!r} != {metadata['task_id']!r}"
        )
    if text_info["language_instruction"] != metadata["language_instruction"]:
        raise ValueError(
            "Mismatched language instructions between text info and metadata: "
            f"{text_info['language_instruction']!r} != {metadata['language_instruction']!r}"
        )

    goal_summary = normalize_goal_state(metadata["goal_state"])
    return build_task_plan_root(
        task_id=text_info["task_id"],
        language_instruction=text_info["language_instruction"],
        formal_goal=text_info["formal_goal"],
        description=text_info["language_instruction"],
        goal_summary=goal_summary,
    )


def bootstrap_task_plan_from_bddl_file(
    bddl_path: str | Path,
    output_dir: str | Path | None = None,
    filename: str = "vlm_planning_tree.json",
) -> tuple[Dict[str, Any], Path]:
    """从单个 BDDL 文件一站式初始化并落盘任务树。

    这是给上层调用最方便的 bootstrap 入口，内部依次执行：
    - `parse_bddl_text(...)`
    - `parse_bddl_metadata(...)`
    - `build_task_plan_from_bddl(...)`
    - `write_task_plan_json(...)`

    输入示例：
        bddl_path = "sample_task.bddl"
        output_dir = "./case_dir"

    输出示例：
        (
          {"task_id": "LIBERO_Coffee_Table_Manipulation", "root": {...}},
          Path("./case_dir/vlm_planning_tree.json")
        )
    """
    text_info = parse_bddl_text(bddl_path)
    metadata = parse_bddl_metadata(bddl_path)
    plan = build_task_plan_from_bddl(text_info, metadata)

    target_dir = Path(output_dir) if output_dir is not None else Path(bddl_path).parent
    output_path = write_task_plan_json(plan, target_dir, filename=filename)
    return plan, output_path


def clone_task_plan(plan: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝一份任务树，并在拷贝前先做 schema 校验。

    常见用途是：
    - 每个 episode 从同一个 base task tree 派生出独立副本
    - 避免不同 episode 之间共享 `children` / `history`

    输入示例：
        plan = {"task_id": "task_1", "root": {"children": [], "history": [], ...}}

    输出示例：
        一个内容相同、但与原对象完全独立的新 dict
    """
    validate_task_plan(plan)
    return deepcopy(plan)


def load_task_plan_json(path: str | Path) -> Dict[str, Any]:
    """从磁盘读取任务树 JSON，并立即做合法性校验。

    输入示例：
        path = "./case_dir/vlm_planning_tree.json"

    输出示例：
        {
          "task_id": "LIBERO_Coffee_Table_Manipulation",
          "language_instruction": "...",
          "formal_goal": "...",
          "root": {...}
        }
    """
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_task_plan(plan)
    return plan


def validate_task_plan(plan: Dict[str, Any]) -> None:
    """校验任务树 schema 是否满足当前代码的最小约束。

    当前主要校验：
    - 顶层字段是否齐全
    - `root` 结构是否合法
    - `subtask` / `action` 节点结构是否合法
    - `history` 里的 step、id、description、candidate 列表是否合法
    - `node_id` 是否重复

    输入示例：
        plan = {"task_id": "task_1", "language_instruction": "...", "formal_goal": "...", "root": {...}}

    输出示例：
        无返回值；如果结构不合法会直接抛异常
    """
    _require_dict(plan, "plan")
    _require_non_empty_string(plan.get("task_id"), "plan.task_id")
    _require_non_empty_string(plan.get("language_instruction"), "plan.language_instruction")
    _require_non_empty_string(plan.get("formal_goal"), "plan.formal_goal")

    root = _get_root(plan)
    if root.get("node_id") != "root":
        raise ValueError(f"plan.root.node_id must be 'root', got {root.get('node_id')!r}")
    if root.get("node_type") != "task":
        raise ValueError(f"plan.root.node_type must be 'task', got {root.get('node_type')!r}")
    _require_non_empty_string(root.get("description"), "plan.root.description")

    goal_summary = _require_list(root.get("goal_summary"), "plan.root.goal_summary")
    for idx, item in enumerate(goal_summary):
        _require_non_empty_string(item, f"plan.root.goal_summary[{idx}]")

    history = _require_list(root.get("history"), "plan.root.history")
    for idx, entry in enumerate(history):
        _require_dict(entry, f"plan.root.history[{idx}]")
        if "step" in entry:
            step = entry["step"]
            if isinstance(step, bool) or not isinstance(step, int):
                raise TypeError(f"plan.root.history[{idx}].step must be an integer, got {type(step)!r}")
        for key in (
            "selected_subtask_id",
            "selected_subtask_description",
            "selected_action_id",
            "selected_action_description",
        ):
            if key in entry and entry[key] is not None:
                _require_non_empty_string(entry[key], f"plan.root.history[{idx}].{key}")
        for key in ("candidate_subtasks", "candidate_actions"):
            if key in entry:
                values = _require_list(entry[key], f"plan.root.history[{idx}].{key}")
                for item_idx, item in enumerate(values):
                    _require_non_empty_string(item, f"plan.root.history[{idx}].{key}[{item_idx}]")

    children = _require_list(root.get("children"), "plan.root.children")
    seen_node_ids = {"root"}
    for idx, child in enumerate(children):
        node = _require_dict(child, f"plan.root.children[{idx}]")
        if node.get("node_type") != "subtask":
            raise ValueError(f"plan.root.children[{idx}].node_type must be 'subtask', got {node.get('node_type')!r}")

        node_id = _require_non_empty_string(node.get("node_id"), f"plan.root.children[{idx}].node_id")
        if node_id in seen_node_ids:
            raise ValueError(f"Duplicate node_id found: {node_id!r}")
        seen_node_ids.add(node_id)

        _require_non_empty_string(node.get("description"), f"plan.root.children[{idx}].description")
        if "expected_effect" in node and node["expected_effect"] is not None:
            _require_non_empty_string(node["expected_effect"], f"plan.root.children[{idx}].expected_effect")

        action_children = _require_list(node.get("children"), f"plan.root.children[{idx}].children")
        for action_idx, action in enumerate(action_children):
            action_node = _require_dict(action, f"plan.root.children[{idx}].children[{action_idx}]")
            if action_node.get("node_type") != "action":
                raise ValueError(
                    "plan.root.children[{idx}].children[{action_idx}].node_type must be 'action', "
                    f"got {action_node.get('node_type')!r}"
                )

            action_id = _require_non_empty_string(
                action_node.get("node_id"),
                f"plan.root.children[{idx}].children[{action_idx}].node_id",
            )
            if action_id in seen_node_ids:
                raise ValueError(f"Duplicate node_id found: {action_id!r}")
            seen_node_ids.add(action_id)

            _require_non_empty_string(
                action_node.get("description"),
                f"plan.root.children[{idx}].children[{action_idx}].description",
            )
            if "step" in action_node and action_node["step"] is not None:
                step = action_node["step"]
                if isinstance(step, bool) or not isinstance(step, int):
                    raise TypeError(
                        f"plan.root.children[{idx}].children[{action_idx}].step must be an integer, got {type(step)!r}"
                    )


def upsert_subtask_node(
    plan: Dict[str, Any],
    *,
    description: str,
    expected_effect: str | None = None,
) -> Dict[str, Any]:
    """按描述查找或创建一个 `subtask` 节点。

    规则：
    - 如果 `root.children` 里已经有相同 `description` 的 subtask，就直接复用
    - 如果没有，就创建新的 `subtask_###`
    - 如果传入了 `expected_effect`，会写回已有节点或新节点

    输入示例：
        description = "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1"
        expected_effect = "cream_cheese_1 in white_storage_box_1_bottom_side"

    输出示例：
        {
          "node_id": "subtask_001",
          "node_type": "subtask",
          "description": "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1",
          "expected_effect": "cream_cheese_1 in white_storage_box_1_bottom_side",
          "children": []
        }
    """
    validate_task_plan(plan)
    normalized_description = _require_non_empty_string(description, "description")
    normalized_effect = None
    if expected_effect is not None:
        normalized_effect = _require_non_empty_string(expected_effect, "expected_effect")

    existing = _find_subtask_node_by_description(plan, normalized_description)
    if existing is not None:
        if normalized_effect is not None:
            existing["expected_effect"] = normalized_effect
        return existing

    root = _get_root(plan)
    children = _require_list(root.get("children"), "plan.root.children")
    node = {
        "node_id": _next_node_id(plan, "subtask"),
        "node_type": "subtask",
        "description": normalized_description,
        "children": [],
    }
    if normalized_effect is not None:
        node["expected_effect"] = normalized_effect
    children.append(node)
    return node


def upsert_action_node(
    plan: Dict[str, Any],
    *,
    subtask_id: str,
    step: int | None,
    description: str,
) -> Dict[str, Any]:
    """在指定 subtask 下按描述查找或创建一个 `action` 节点。

    规则：
    - 先通过 `subtask_id` 找到父 subtask
    - 如果该 subtask 下已有相同 `description` 的 action，就复用
    - 如果没有，就创建新的 `action_###`
    - 如果传入 `step`，会写到复用节点或新节点里

    输入示例：
        subtask_id = "subtask_001"
        step = 0
        description = "move gripper above cream_cheese_1"

    输出示例：
        {
          "node_id": "action_001",
          "node_type": "action",
          "step": 0,
          "description": "move gripper above cream_cheese_1"
        }
    """
    validate_task_plan(plan)
    if step is not None and (isinstance(step, bool) or not isinstance(step, int)):
        raise TypeError(f"step must be an integer or None, got {type(step)!r}")

    subtask_node = _find_subtask_node_by_id(plan, subtask_id)
    normalized_description = _require_non_empty_string(description, "description")
    existing = _find_action_node_by_description(subtask_node, normalized_description)
    if existing is not None:
        if step is not None:
            existing["step"] = step
        return existing

    children = _require_list(subtask_node.get("children"), f"{subtask_id}.children")
    node = {
        "node_id": _next_node_id(plan, "action"),
        "node_type": "action",
        "description": normalized_description,
    }
    if step is not None:
        node["step"] = step
    children.append(node)
    return node


def record_planning_step(
    plan: Dict[str, Any],
    *,
    step: int,
    selected_subtask_description: str,
    selected_action_description: str | None = None,
    candidate_subtasks: List[str] | None = None,
    candidate_actions: List[str] | None = None,
    expected_effect: str | None = None,
) -> Dict[str, Any]:
    """把一轮规划结果写回任务树。

    这个函数是当前任务树的统一写入口，内部会：
    - 调 `upsert_subtask_node(...)` 维护 `root.children` 里的 subtask
    - 如果给了 `selected_action_description`，调 `upsert_action_node(...)`
    - 向 `root.history` 追加一条轨迹记录

    `history` 会同时保存：
    - `selected_subtask_id` / `selected_subtask_description`
    - `selected_action_id` / `selected_action_description`
    - `candidate_subtasks` / `candidate_actions`

    输入示例：
        step = 0
        selected_subtask_description = "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1"
        selected_action_description = "move gripper above cream_cheese_1"
        candidate_subtasks = ["Pick and place cream_cheese_1 into the bottom side of white_storage_box_1"]
        candidate_actions = ["move gripper above cream_cheese_1", "grasp cream_cheese_1"]
        expected_effect = "cream_cheese_1 in white_storage_box_1_bottom_side"

    输出示例：
        {
          "step": 0,
          "selected_subtask_id": "subtask_001",
          "selected_subtask_description": "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1",
          "candidate_subtasks": [
            "Pick and place cream_cheese_1 into the bottom side of white_storage_box_1"
          ],
          "selected_action_id": "action_001",
          "selected_action_description": "move gripper above cream_cheese_1",
          "candidate_actions": [
            "move gripper above cream_cheese_1",
            "grasp cream_cheese_1"
          ]
        }
    """
    validate_task_plan(plan)
    if isinstance(step, bool) or not isinstance(step, int):
        raise TypeError(f"step must be an integer, got {type(step)!r}")

    subtask_node = upsert_subtask_node(
        plan,
        description=selected_subtask_description,
        expected_effect=expected_effect,
    )

    action_node = None
    if selected_action_description is not None:
        action_node = upsert_action_node(
            plan,
            subtask_id=subtask_node["node_id"],
            step=step,
            description=selected_action_description,
        )

    history_entry: Dict[str, Any] = {
        "step": step,
        "selected_subtask_id": subtask_node["node_id"],
        "selected_subtask_description": subtask_node["description"],
    }

    if candidate_subtasks is not None:
        history_entry["candidate_subtasks"] = [
            _require_non_empty_string(item, "candidate_subtasks[]") for item in candidate_subtasks
        ]

    if action_node is not None:
        history_entry["selected_action_id"] = action_node["node_id"]
        history_entry["selected_action_description"] = action_node["description"]

    if candidate_actions is not None:
        history_entry["candidate_actions"] = [
            _require_non_empty_string(item, "candidate_actions[]") for item in candidate_actions
        ]

    root = _get_root(plan)
    history = _require_list(root.get("history"), "plan.root.history")
    history.append(history_entry)
    return history_entry


def write_task_plan_json(
    plan: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "vlm_planning_tree.json",
) -> Path:
    """把任务树写成 JSON 文件。

    输入示例：
        plan = {"task_id": "task_1", "root": {...}}
        output_dir = "./case_dir"
        filename = "vlm_planning_tree.json"

    输出示例：
        Path("./case_dir/vlm_planning_tree.json")
    """
    output_path = Path(output_dir) / filename
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path

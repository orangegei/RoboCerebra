import json
import re
import sys
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
    if not isinstance(predicate, list):
        raise ValueError(f"Goal predicate must be a list, got {type(predicate)!r}")
    if len(predicate) != 3:
        raise ValueError(f"Goal predicate must contain exactly 3 items, got {len(predicate)}: {predicate!r}")

    relation, subject, target = predicate
    if not all(isinstance(item, str) and item.strip() for item in (relation, subject, target)):
        raise ValueError(f"Goal predicate items must be non-empty strings: {predicate!r}")

    return f"{subject} {relation.lower()} {target}"


def normalize_goal_state(goal_state: List[List[str]]) -> List[str]:
    if not isinstance(goal_state, list):
        raise ValueError(f"goal_state must be a list, got {type(goal_state)!r}")
    return [normalize_goal_predicate(predicate) for predicate in goal_state]


def build_task_plan_from_bddl(
    text_info: Dict[str, str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
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


def write_task_plan_json(
    plan: Dict[str, Any],
    output_dir: str | Path,
    filename: str = "vlm_planning_tree.json",
) -> Path:
    output_path = Path(output_dir) / filename
    output_path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path

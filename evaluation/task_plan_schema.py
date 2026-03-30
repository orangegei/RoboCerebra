from dataclasses import asdict, dataclass, field
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
    task: str = ""
    language_instruction: str = ""
    formal_goal: str = ""
    status: str = "running"
    root: TaskPlanRootNode = field(default_factory=TaskPlanRootNode)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_task_plan_root(
    task: str = "",
    language_instruction: str = "",
    formal_goal: str = "",
    description: str = "",
    goal_summary: List[str] | None = None,
    status: str = "running",
) -> Dict[str, Any]:
    plan = TaskPlanSchema(
        task=task,
        language_instruction=language_instruction,
        formal_goal=formal_goal,
        status=status,
        root=TaskPlanRootNode(
            description=description,
            goal_summary=list(goal_summary or []),
        ),
    )
    return plan.to_dict()

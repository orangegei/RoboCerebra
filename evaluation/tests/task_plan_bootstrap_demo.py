import argparse
import json
from pathlib import Path

from evaluation.task_planner import (
    build_task_plan_from_bddl,
    parse_bddl_metadata,
    parse_bddl_text,
    write_task_plan_json,
)

def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap a task planning tree from a BDDL file.")
    parser.add_argument("bddl_path", type=Path, help="Path to the input .bddl file")
    
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to write vlm_planning_tree.json into. Defaults to the BDDL parent directory.",
    )
    args = parser.parse_args()

    text_info = parse_bddl_text(args.bddl_path)
    metadata = parse_bddl_metadata(args.bddl_path)
    plan = build_task_plan_from_bddl(text_info, metadata)
    print(f"text_info: {text_info}")
    print(f"metadata: {metadata}")

    output_dir = args.output_dir or args.bddl_path.parent
    output_path = write_task_plan_json(plan, output_dir)

    print(f"Wrote planning tree to: {output_path}")
    print(json.dumps(plan, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# $ python grad/demo_state_predicate_scan.py /home/shenhaotian/grad/RoboCerebra/RoboCerebra/RoboCerebra_trainset/coffee_table/case1
# $ python grad/demo_state_predicate_scan.py /home/shenhaotian/grad/RoboCerebra/RoboCerebra/RoboCerebraBench/Ideal/case1
"""
Scan demo states, set sim state at each timestep, check predicates, and save images
when new subtasks complete.

Usage:
  python grad/demo_state_predicate_scan.py /path/to/task_dir
  python grad/demo_state_predicate_scan.py /path/to/task_dir --out /path/to/output
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_ROOT = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from evaluation.task_runner import load_task_data, setup_task_environment
from evaluation.robocerebra_logging import log_message
from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image


def _ensure_output_dir(base_dir: Path, task_type: str, case_name: str) -> Path:
    safe_task_type = task_type.replace(" ", "_")
    safe_case_name = case_name.replace(" ", "_")
    out_dir = base_dir / safe_task_type / safe_case_name
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir


def _save_views(obs, out_dir: Path, t: int, diff: int, total_completed: int) -> None:
    agent_img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    agent_path = out_dir / f"t{t:06d}_diff{diff}_total{total_completed}_agent.png"
    wrist_path = out_dir / f"t{t:06d}_diff{diff}_total{total_completed}_wrist.png"

    Image.fromarray(agent_img).save(agent_path)
    Image.fromarray(wrist_img).save(wrist_path)


def _set_state_from_demo(env, demo_state: np.ndarray) -> None:
    state = np.asarray(demo_state, dtype=np.float64).ravel()
    expected_len = env.sim.get_state().flatten().shape[0]

    if state.shape[0] == expected_len:
        flat_state = state
    elif state.shape[0] == expected_len + 1:
        flat_state = state[1:]
    else:
        raise ValueError(
            f"demo_state length {state.shape[0]} does not match expected {expected_len} or {expected_len + 1}"
        )

    env.sim.set_state_from_flattened(flat_state)
    env.sim.forward()
    env._post_process()
    env._update_observables(force=True)


def scan_task(task_dir: Path, out_dir: Path) -> None:
    env, bddl_file_path, error = setup_task_environment(task_dir)
    if error:
        raise RuntimeError(error)

    orig_states, goal, goal_steps, error = load_task_data(task_dir)
    if error:
        env.close()
        raise RuntimeError(error)

    task_type = task_dir.parent.name
    case_name = task_dir.name
    save_dir = _ensure_output_dir(out_dir, task_type, case_name)

    total_completed_prev = None

    for t, demo_state in enumerate(orig_states):
        _set_state_from_demo(env, demo_state)
        obs = env._get_observations()
        _, total_completed_now, _ = env._check_success(goal)

        if total_completed_prev is None:
            total_completed_prev = total_completed_now
            continue

        diff = total_completed_now - total_completed_prev
        if diff > 0:
            _save_views(obs, save_dir, t, diff, total_completed_now)
            log_message(
                f"[OK] {task_type}/{case_name} t={t} diff={diff} total={total_completed_now} saved to {save_dir}"
            )
            total_completed_prev = total_completed_now

    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan demo states and save images on subtask completion.")
    parser.add_argument("task_dir", type=str, help="Path to a task directory containing demo.hdf5 and .bddl")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output directory for saved images (default: ./rollouts/demo_state_scan-<timestamp>)",
    )
    args = parser.parse_args()

    task_dir = Path(args.task_dir).resolve()
    if not task_dir.exists():
        raise FileNotFoundError(f"Task dir not found: {task_dir}")

    if args.out:
        out_dir = Path(args.out).resolve()
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = PROJECT_ROOT / "rollouts" / f"demo_state_scan-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    scan_task(task_dir, out_dir)


if __name__ == "__main__":
    main()

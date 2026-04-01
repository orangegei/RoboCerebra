#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared RoboCerebra evaluation entrypoint.
"""

import logging
import os
import random
import sys
from collections import deque
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import tqdm
import wandb

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from config import GenerateConfig, validate_config
from episode import (
    finalize_episode,
    handle_dynamic_movement,
    handle_segment_transition,
    initialize_episode_state,
    update_completion_tracking,
)
from model_adapters import get_policy_adapter
from resume import create_step_based_resume_handler
from robocerebra_logging import (
    get_rollout_task_dir,
    log_message,
    save_results_log,
    setup_logging,
)
from task_planner import (
    bootstrap_task_plan_from_bddl_file,
    clone_task_plan,
    record_planning_step,
    write_task_plan_json,
)
from task_runner import (
    load_task_data,
    setup_task_descriptions,
    setup_task_environment,
    validate_task_configuration,
)
from utils import get_task_directories


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _set_seed_everywhere(seed: int) -> None:
    try:
        import torch
    except Exception:
        torch = None

    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def _get_libero_dummy_action(model_family: str):
    try:
        from experiments.robot.libero.libero_utils import get_libero_dummy_action
        return get_libero_dummy_action(model_family)
    except ImportError:
        return [0, 0, 0, 0, 0, 0, -1]


def run_episode(
    cfg: GenerateConfig,
    env,
    naming_step_desc: Sequence[str],
    model_step_desc: Sequence[str],
    step_states: Sequence[np.ndarray] | None,
    policy_adapter,
    policy_runtime,
    goal: Any,
    log_file=None,
    episode_idx: int = 0,
    distractor_info: Optional[Dict[str, Any]] = None,
    task_line: str | None = None,
    task_name: str = "",
    wait_flag=True,
    task_type: str = "",
    case_name: str = "",
    initial_state: Optional[np.ndarray] = None,
    resume_handler: Optional[Dict[str, Any]] = None,
    planner_runtime: Any = None,
    current_task_tree: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, int, int]:
    """Run a single evaluation episode."""

    del task_name

    segment_count = len(naming_step_desc) if cfg.task_description_suffix else len(model_step_desc)
    full_description = task_line or "" if cfg.complete_description else None

    obs, episode_stats = initialize_episode_state(
        cfg, env, goal, step_states, initial_state, task_type, case_name, resume_handler, log_file
    )

    if cfg.dynamic and distractor_info:
        rng = np.random.default_rng()
        toggle_dir = -1
        seg_mid_moved = False
        resume_trigger_step = None

    seg_increment_accum = 0
    action_queue: deque[np.ndarray] = deque(maxlen=cfg.num_open_loop_steps)
    replay_images_all: List[np.ndarray] = []
    replay_images_seg: List[np.ndarray] = []
    t = 0
    max_steps = cfg.switch_steps * segment_count
    prev_step_idx = 0

    if not wait_flag:
        comp_start_dict, total_completed_prev, _ = env._check_success(goal)
        if cfg.dynamic_shift_description:
            log_message(
                f"[Dynamic Shift] Final completion baseline: {total_completed_prev} total, details: {comp_start_dict}",
                log_file,
            )

    while t < max_steps:
        if t < cfg.num_steps_wait and wait_flag:
            obs, _, _, _ = env.step(_get_libero_dummy_action(cfg.model_family))
            t += 1
            continue

        if t == cfg.num_steps_wait and wait_flag:
            comp_start_dict, total_completed_prev, _ = env._check_success(goal)
            if cfg.dynamic_shift_description:
                log_message(
                    f"[Dynamic Shift] Post-wait completion baseline: {total_completed_prev} total, details: {comp_start_dict}",
                    log_file,
                )

        step_idx = (t // cfg.switch_steps) % segment_count

        if t % cfg.switch_steps == 0 and cfg.dynamic and distractor_info:
            seg_mid_moved = False

        if t > 0 and step_idx != prev_step_idx:
            comp_start_dict, replay_images_seg, seg_increment_accum, _, skip_increment, new_trigger = (
                handle_segment_transition(
                    cfg,
                    env,
                    goal,
                    step_idx,
                    prev_step_idx,
                    seg_increment_accum,
                    replay_images_seg,
                    episode_idx,
                    naming_step_desc,
                    task_type,
                    case_name,
                    comp_start_dict,
                    step_states,
                    episode_stats,
                    resume_handler,
                    log_file,
                )
            )
            episode_stats["skip_increment"] = skip_increment
            if new_trigger is not None:
                resume_trigger_step = t

        prev_step_idx = step_idx

        if cfg.dynamic and distractor_info:
            obs, toggle_dir, resume_trigger_step, seg_mid_moved = handle_dynamic_movement(
                cfg, env, distractor_info, step_idx, resume_trigger_step, t, rng, toggle_dir, seg_mid_moved, log_file
            )

        from utils import prepare_observation

        # obs
        observation, img = prepare_observation(obs, policy_runtime.resize_size)
        replay_images_all.append(img)
        replay_images_seg.append(img)

        planner_selected_desc = None
        should_replan = cfg.vlm_planner_force_single_step or (not action_queue)
        if (
            (not cfg.use_task_tree_desc_baseline)
            and should_replan
            and cfg.use_vlm_planner
            and planner_runtime is not None
            and current_task_tree is not None
        ):
            try:
                from vlm_planner import plan_actions, plan_subtasks

                subtask_result = plan_subtasks(cfg, planner_runtime, observation, current_task_tree)
                if subtask_result.selected_subtask_description is not None:
                    planner_selected_desc = subtask_result.selected_subtask_description
                    selected_action_description = None
                    candidate_actions = None
                    try:
                        action_result = plan_actions(
                            cfg,
                            planner_runtime,
                            observation,
                            current_task_tree,
                            subtask_result.selected_subtask_description,
                        )
                        selected_action_description = action_result.selected_action_description
                        candidate_actions = action_result.candidate_actions
                        if selected_action_description is None:
                            log_message(
                                f"[WARN] Action planner returned no selected_action_description at step {t}",
                                log_file,
                            )
                        else:
                            planner_selected_desc = selected_action_description
                    except Exception as action_exc:
                        log_message(f"[WARN] Action planning failed at step {t}: {action_exc}", log_file)

                    record_planning_step(
                        current_task_tree,
                        step=t,
                        selected_subtask_description=subtask_result.selected_subtask_description,
                        candidate_subtasks=subtask_result.candidate_subtasks,
                        selected_action_description=selected_action_description,
                        candidate_actions=candidate_actions,
                    )
                else:
                    log_message(
                        f"[WARN] Subtask planner returned no selected_subtask_description at step {t}, skip record",
                        log_file,
                    )
            except Exception as exc:
                log_message(f"[WARN] Subtask planning failed at step {t}: {exc}", log_file)

        if cfg.use_task_tree_desc_baseline and current_task_tree is not None:
            language_instruction = current_task_tree.get("language_instruction")
            normalized_language_instruction = None
            if isinstance(language_instruction, str):
                normalized_language_instruction = " ".join(language_instruction.split()).strip()
                if not normalized_language_instruction:
                    normalized_language_instruction = None

            normalized_goal_summary: List[str] = []
            root = current_task_tree.get("root")
            if isinstance(root, dict):
                goal_summary = root.get("goal_summary")
                if isinstance(goal_summary, list):
                    for item in goal_summary:
                        if isinstance(item, str):
                            normalized_item = " ".join(item.split()).strip()
                            if normalized_item:
                                normalized_goal_summary.append(normalized_item)

            if normalized_language_instruction is not None and normalized_goal_summary:
                desc = (
                    f"{normalized_language_instruction}. "
                    f"Goal summary: {'; '.join(normalized_goal_summary)}"
                )
            elif normalized_language_instruction is not None:
                desc = normalized_language_instruction
            elif normalized_goal_summary:
                desc = f"Goal summary: {'; '.join(normalized_goal_summary)}"
            elif cfg.task_description_suffix != "" and not cfg.complete_description:
                desc = naming_step_desc[step_idx]
            else:
                desc = full_description if cfg.complete_description else model_step_desc[step_idx]
        elif planner_selected_desc is not None:
            desc = planner_selected_desc
        elif cfg.task_description_suffix != "" and not cfg.complete_description:
            desc = naming_step_desc[step_idx]
        else: # desc
            desc = full_description if cfg.complete_description else model_step_desc[step_idx]

        if not action_queue: # infer
            actions = policy_adapter.predict_actions(cfg, policy_runtime, observation, desc)
            action_queue.extend(actions)
        raw_action = action_queue.popleft()

        obs, _, _, _ = env.step(policy_adapter.postprocess_action(cfg, raw_action).tolist())
        t += 1

        seg_diff, total_completed_prev = update_completion_tracking(
            env, goal, total_completed_prev, episode_stats, step_idx, log_file
        )
        seg_increment_accum += seg_diff

        if episode_stats["skip_increment"]:
            episode_stats["skip_increment"] = False

    return finalize_episode(
        cfg,
        env,
        goal,
        replay_images_all,
        replay_images_seg,
        episode_idx,
        prev_step_idx,
        seg_increment_accum,
        naming_step_desc,
        task_type,
        case_name,
        episode_stats,
        log_file,
    )


def run_task(
    cfg: GenerateConfig,
    task_type: str,
    task_dir: Path,
    policy_adapter,
    policy_runtime,
    planner_runtime: Any = None,
    log_file=None,
) -> Tuple[int, int, int, int, Dict]:
    """Evaluate a single task directory."""

    env, bddl_file_path, error = setup_task_environment(task_dir, log_file)
    if error:
        return 0, 0, 0, 0, {
            "task_type": task_type,
            "case_name": task_dir.name,
            "episodes": 0,
            "successes": 0,
            "success_rate": 0,
            "agent_subtasks": 0,
            "possible_subtasks": 0,
            "subtask_rate": 0,
            "bddl_file": bddl_file_path,
            "used_init_files": cfg.use_init_files,
            "has_step_annotations": False,
            "error": error,
        }

    orig_states, goal, goal_steps, error = load_task_data(task_dir, log_file)
    if error:
        return 0, 0, 0, 0, {
            "task_type": task_type,
            "case_name": task_dir.name,
            "episodes": 0,
            "successes": 0,
            "success_rate": 0,
            "agent_subtasks": 0,
            "possible_subtasks": 0,
            "subtask_rate": 0,
            "bddl_file": bddl_file_path,
            "used_init_files": cfg.use_init_files,
            "has_step_annotations": bool(goal_steps),
            "error": error,
        }

    naming_step_desc, model_step_desc, start_indices, task_line, error = setup_task_descriptions(
        cfg, task_dir, log_file
    )
    if error:
        return 0, 0, 0, 0, {
            "task_type": task_type,
            "case_name": task_dir.name,
            "episodes": 0,
            "successes": 0,
            "success_rate": 0,
            "agent_subtasks": 0,
            "possible_subtasks": 0,
            "subtask_rate": 0,
            "bddl_file": bddl_file_path,
            "used_init_files": cfg.use_init_files,
            "has_step_annotations": bool(goal_steps),
            "error": error,
        }

    is_valid, base_result = validate_task_configuration(
        cfg, naming_step_desc, start_indices, model_step_desc, goal, goal_steps, bddl_file_path, task_type, task_dir.name, log_file
    )
    if not is_valid:
        return 0, 0, 0, 0, base_result

    rollout_task_dir = get_rollout_task_dir(
        task_suite=f"{cfg.task_suite_name}_{task_type}",
        task_name=task_dir.name,
    )

    # Initialize base task planning tree from BDDL file (if available)
    base_task_tree: Optional[Dict[str, Any]] = None
    try:
        base_task_tree, task_plan_path = bootstrap_task_plan_from_bddl_file(
            bddl_file_path,
            output_dir=rollout_task_dir,
            filename="vlm_planning_tree_base.json",
        )
        log_message(f"Initialized task planning tree at {task_plan_path}", log_file)
    except Exception as exc:
        log_message(f"[WARN] Failed to initialize task planning tree for {task_dir.name}: {exc}", log_file)

    from utils import load_init_state, setup_dynamic_distractor_info

    distractor_info = setup_dynamic_distractor_info(cfg, task_dir, env, naming_step_desc, log_file)
    initial_states = [load_init_state(cfg, task_type, task_dir.name, log_file)] if cfg.use_init_files else None
    wait_flag = start_indices[0] == 0
    step_states = [orig_states[idx] for idx in start_indices]
    resume_handler = create_step_based_resume_handler(goal, goal_steps) if goal and goal_steps else {}

    episodes = cfg.num_trials_per_task
    successes = 0
    task_agent_subtasks = 0
    task_possible_subtasks = 0

    for ep_idx in tqdm.tqdm(range(episodes)):
        current_task_tree = clone_task_plan(base_task_tree) if base_task_tree is not None else None

        initial_state = None
        if initial_states and initial_states[0] is not None:
            if cfg.initial_states_path == "DEFAULT":
                initial_state = initial_states[0]
            else:
                initial_state = initial_states[ep_idx % len(initial_states)]

        succ, ep_subtasks, ep_goals = run_episode(
            cfg,
            env,
            naming_step_desc,
            model_step_desc,
            step_states,
            policy_adapter,
            policy_runtime,
            goal,
            log_file,
            episode_idx=ep_idx,
            distractor_info=distractor_info,
            task_line=task_line,
            task_name=task_dir.name,
            wait_flag=wait_flag,
            task_type=task_type,
            case_name=task_dir.name,
            initial_state=initial_state,
            resume_handler=resume_handler,
            planner_runtime=planner_runtime,
            current_task_tree=current_task_tree,
        )
        if current_task_tree is not None:
            try:
                episode_tree_path = write_task_plan_json(
                    current_task_tree,
                    rollout_task_dir,
                    filename=f"vlm_planning_tree_episode={ep_idx}.json",
                )
                log_message(f"Saved episode task planning tree at {episode_tree_path}", log_file)
            except Exception as exc:
                log_message(
                    f"[WARN] Failed to save episode task planning tree for {task_dir.name} ep={ep_idx}: {exc}",
                    log_file,
                )
        successes += int(succ)
        task_agent_subtasks += ep_subtasks
        task_possible_subtasks += ep_goals

    try:
        env.close()
    except Exception:
        pass

    task_result = {
        "task_type": task_type,
        "case_name": task_dir.name,
        "episodes": episodes,
        "successes": successes,
        "success_rate": successes / episodes if episodes > 0 else 0,
        "agent_subtasks": task_agent_subtasks,
        "possible_subtasks": task_possible_subtasks,
        "subtask_rate": task_agent_subtasks / task_possible_subtasks if task_possible_subtasks > 0 else 0,
        "bddl_file": bddl_file_path,
        "used_init_files": cfg.use_init_files,
        "has_step_annotations": bool(goal_steps),
        "configuration": {
            "dynamic": cfg.dynamic,
            "dynamic_shift_description": cfg.dynamic_shift_description,
            "resume": cfg.resume,
            "complete_description": cfg.complete_description,
            "excludes_forced_completions": cfg.dynamic_shift_description,
            "use_vlm_planner": cfg.use_vlm_planner,
            "use_task_tree_desc_baseline": cfg.use_task_tree_desc_baseline,
            "desc_source": (
                "task_tree_language_instruction_plus_goal_summary"
                if cfg.use_task_tree_desc_baseline
                else ("vlm_planner_then_default_fallback" if cfg.use_vlm_planner else "default_task_description")
            ),
        },
    }

    return episodes, successes, task_agent_subtasks, task_possible_subtasks, task_result


def _configure_task_type(cfg: GenerateConfig, task_type: str) -> None:
    if task_type == "Ideal":
        cfg.dynamic = False
        cfg.dynamic_shift_description = False
        cfg.resume = True
    elif task_type == "Mix":
        cfg.dynamic = True
        cfg.dynamic_shift_description = True
        cfg.resume = True
    elif task_type == "Random_Disturbance":
        cfg.dynamic = True
        cfg.dynamic_shift_description = False
        cfg.resume = True
    elif task_type == "Observation_Mismatching":
        cfg.dynamic = False
        cfg.dynamic_shift_description = True
        cfg.resume = True
    else:
        cfg.dynamic = False
        cfg.dynamic_shift_description = False
        cfg.resume = True


def run_evaluation(cfg: GenerateConfig) -> float:
    """Shared evaluation function used by model-specific wrappers."""

    validate_config(cfg)
    _set_seed_everywhere(cfg.seed)

    policy_adapter = get_policy_adapter(cfg.model_family)
    policy_runtime = policy_adapter.initialize(cfg)

    log_file, _, run_id, results_log_filepath = setup_logging(cfg)

    planner_runtime = None
    if cfg.use_vlm_planner and not cfg.use_task_tree_desc_baseline:
        from vlm_planner import initialize_vlm_runtime

        planner_runtime = initialize_vlm_runtime(cfg.vlm_model_path_or_name, enabled=True)
        log_message("Initialized VLM planner runtime", log_file)
    elif cfg.use_vlm_planner and cfg.use_task_tree_desc_baseline:
        log_message(
            "Skipped VLM planner runtime initialization because task-tree desc baseline is enabled",
            log_file,
        )

    log_message("Starting RoboCerebra evaluation", log_file)
    log_message(f"Model family: {cfg.model_family}", log_file)
    log_message(f"RoboCerebra root: {cfg.robocerebra_root}", log_file)
    log_message(f"Init files root: {cfg.init_files_root}", log_file)
    log_message(f"Use init files: {cfg.use_init_files}", log_file)
    log_message(f"Use VLM planner: {cfg.use_vlm_planner}", log_file)
    log_message(f"Use task-tree desc baseline: {cfg.use_task_tree_desc_baseline}", log_file)
    log_message(
        "Policy desc source: task_tree(language_instruction + goal_summary)"
        if cfg.use_task_tree_desc_baseline
        else (
            "Policy desc source: vlm_planner selected action/subtask with fallback"
            if cfg.use_vlm_planner
            else "Policy desc source: default task description"
        ),
        log_file,
    )
    log_message(f"Task types: {cfg.task_types}", log_file)
    log_message(
        f"Dynamic parameters - dynamic: {cfg.dynamic}, dynamic_shift_description: {cfg.dynamic_shift_description}, resume: {cfg.resume}",
        log_file,
    )

    task_dirs = get_task_directories(cfg)

    total_eps = 0
    total_success = 0
    total_agent_subtasks = 0
    total_possible_subtasks = 0
    results_by_task_type = {}
    all_task_results = []

    for task_type in cfg.task_types:
        task_type_dirs = [(tt, td) for tt, td in task_dirs if tt == task_type]
        if not task_type_dirs:
            log_message(f"No tasks found for task type: {task_type}", log_file)
            continue

        original_dynamic = cfg.dynamic
        original_dynamic_shift = cfg.dynamic_shift_description
        original_resume = cfg.resume

        _configure_task_type(cfg, task_type)

        log_message(f"Evaluating {len(task_type_dirs)} tasks for task type: {task_type}", log_file)
        log_message(
            f"Task type {task_type} - dynamic: {cfg.dynamic}, dynamic_shift_description: {cfg.dynamic_shift_description}, resume: {cfg.resume}",
            log_file,
        )

        task_type_episodes = 0
        task_type_successes = 0
        task_type_agent_subtasks = 0
        task_type_possible_subtasks = 0

        for _, task_dir in task_type_dirs:
            eps, succ, subtasks, possible, task_result = run_task(
                cfg, task_type, task_dir, policy_adapter, policy_runtime, planner_runtime, log_file
            )
            all_task_results.append(task_result)
            task_type_episodes += eps
            task_type_successes += succ
            task_type_agent_subtasks += subtasks
            task_type_possible_subtasks += possible
            total_eps += eps
            total_success += succ
            total_agent_subtasks += subtasks
            total_possible_subtasks += possible

        task_type_success_rate = task_type_successes / task_type_episodes if task_type_episodes > 0 else 0
        task_type_subtask_rate = (
            task_type_agent_subtasks / task_type_possible_subtasks if task_type_possible_subtasks > 0 else 0
        )

        results_by_task_type[task_type] = {
            "episodes": task_type_episodes,
            "successes": task_type_successes,
            "success_rate": task_type_success_rate,
            "subtask_rate": task_type_subtask_rate,
            "agent_subtasks": task_type_agent_subtasks,
            "possible_subtasks": task_type_possible_subtasks,
        }

        log_message(
            f"Task type {task_type} complete: "
            f"Episode success rate: {task_type_success_rate:.2%} ({task_type_successes}/{task_type_episodes}), "
            f"Subtask success rate: {task_type_subtask_rate:.2%} ({task_type_agent_subtasks}/{task_type_possible_subtasks})",
            log_file,
        )

        cfg.dynamic = original_dynamic
        cfg.dynamic_shift_description = original_dynamic_shift
        cfg.resume = original_resume

    overall_success_rate = total_success / total_eps if total_eps > 0 else 0
    overall_subtask_rate = total_agent_subtasks / total_possible_subtasks if total_possible_subtasks > 0 else 0

    log_message("=" * 60, log_file)
    log_message("FINAL RESULTS", log_file)
    log_message("=" * 60, log_file)

    for task_type, results in results_by_task_type.items():
        log_message(
            f"{task_type}: Episode {results['success_rate']:.2%} ({results['successes']}/{results['episodes']}), "
            f"Subtask {results['subtask_rate']:.2%} ({results['agent_subtasks']}/{results['possible_subtasks']})",
            log_file,
        )

    log_message(
        f"OVERALL: Episode {overall_success_rate:.2%} ({total_success}/{total_eps}), "
        f"Subtask {overall_subtask_rate:.2%} ({total_agent_subtasks}/{total_possible_subtasks})",
        log_file,
    )

    save_results_log(
        results_log_filepath,
        cfg,
        results_by_task_type,
        total_eps,
        total_success,
        total_agent_subtasks,
        total_possible_subtasks,
        run_id,
        all_task_results,
    )

    if cfg.use_wandb:
        for task_type, results in results_by_task_type.items():
            wandb.log(
                {
                    f"success_rate/{task_type}": results["success_rate"],
                    f"subtask_rate/{task_type}": results["subtask_rate"],
                    f"num_episodes/{task_type}": results["episodes"],
                }
            )
        wandb.log(
            {
                "success_rate/overall": overall_success_rate,
                "subtask_rate/overall": overall_subtask_rate,
                "num_episodes/total": total_eps,
            }
        )

    if log_file:
        log_file.close()

    return overall_success_rate

# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RoboCerebra is a benchmark for evaluating long-horizon robotic manipulation with System 2 reasoning capabilities in vision-language models. Published at NeurIPS 2025 (arXiv:2506.06677). Dataset on Hugging Face: `qiukingballball/RoboCerebraBench`.

## Commands

### Environment Setup
```bash
# LIBERO-only (Python 3.8.13)
conda create -n libero python=3.8.13 && conda activate libero
cd LIBERO && pip install -r requirements.txt && pip install -e .

# OpenVLA evaluation (Python 3.10)
conda create -n openvla-oft python=3.10 -y && conda activate openvla-oft
```

### Running Evaluation
```bash
cd evaluation/
python eval_openvla.py --task_types ["Ideal", "Random_Disturbance"]
CUDA_VISIBLE_DEVICES=0 python eval_openvla.py --num_trials_per_task 5
```

### Dataset Conversion (RLDS)
```bash
cd rlds_dataset_builder/
python regenerate_robocerebra_dataset.py --robocerebra_raw_data_dir "/path/to/data" --robocerebra_target_dir "./converted_hdf5/out"
cd RoboCerebraDataset && CUDA_VISIBLE_DEVICES="" tfds build --overwrite
```

## Architecture

### Evaluation Pipeline (`evaluation/`)

Entry points (`eval_openvla.py`, `eval_pi0.py`) are thin wrappers using `draccus` for CLI args. They call `run_evaluation()` in `eval_policy.py`.

**Execution flow:**
`run_evaluation()` → per task type: `_configure_task_type()` sets dynamic/resume flags → `run_task()` per task dir → `run_episode()` per trial

**Key modules:**
- `config.py` — `GenerateConfig` dataclass with all parameters. Task type flags (`dynamic`, `dynamic_shift_description`, `resume`) are auto-configured per task type in `_configure_task_type()`.
- `task_runner.py` — `setup_task_environment()`, `load_task_data()`, `setup_task_descriptions()` — loads BDDL, demonstrations, and step descriptions from task directories.
- `episode.py` — `initialize_episode_state()`, `handle_dynamic_movement()`, `handle_segment_transition()`, `finalize_episode()` — manages episode lifecycle including distractor objects and subtask tracking.
- `utils.py` — Data loading, observation preparation, dynamic distractor setup.
- `resume.py` — Step-based resume handler for interrupted evaluations.
- `robocerebra_logging.py` — Structured logging, JSON results, video rollout saving, optional WandB.

### Model Adapter Pattern (`evaluation/model_adapters/`)

`PolicyAdapter` protocol in `base.py` defines: `initialize()`, `predict_actions()`, `postprocess_action()`. `PolicyRuntime` dataclass holds model state. Registry in `registry.py` maps model family strings to adapter instances. Currently: `openvla` (implemented), `pi0` (scaffold).

To add a new model: implement `PolicyAdapter` in a new file, register in `registry.py`.

### Task Types and Their Config Flags

| Task Type | `dynamic` | `dynamic_shift_description` | `resume` |
|---|---|---|---|
| Ideal | false | false | true |
| Random_Disturbance | true | false | true |
| Mix | true | true | true |
| Observation_Mismatching | false | true | true |
| Memory_Execution / Memory_Exploration | false | false | true |

### Dataset Builder (`rlds_dataset_builder/`)

Two-step conversion: raw data → HDF5 (`regenerate_robocerebra_dataset.py`) → RLDS (`RoboCerebraDataset/`).

## Configuration Placeholders

Before running, replace in `evaluation/config.py`:
- `pretrained_checkpoint` — model checkpoint path
- `robocerebra_root` — benchmark dataset root
- `init_files_root` — init files directory
- `wandb_entity` / `wandb_project` — WandB settings (optional)

## Key Constants

- Max steps per task: 400 (configurable via `TASK_MAX_STEPS`)
- Switch steps between segments: 150 (`cfg.switch_steps`)
- Wait steps before policy acts: 15 (`cfg.num_steps_wait`)
- Open-loop action chunk size: 8 (`cfg.num_open_loop_steps`)
- Environment image resolution: 256×256
- Robot: Panda arm, OSC_POSE controller, 20 Hz control frequency
- `MOVABLE_OBJECT_LIST` in `config.py` defines valid distractor objects (30 items)

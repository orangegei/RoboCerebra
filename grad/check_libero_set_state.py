# python grad/check_libero_set_state.py /home/shenhaotian/grad/RoboCerebra/RoboCerebra_Bench/Ideal/case1/demo.hdf5

# demo_states shape: (T, 71)
# sim.data.qpos: shape=(37,) dtype=float64
# sim.data.qvel: shape=(33,) dtype=float64


import sys
import numpy as np
import h5py
from pathlib import Path
PEOJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PEOJECT_ROOT))

from robosuite import load_controller_config
import libero.libero.envs.bddl_utils as BDDLUtils
from libero.libero.envs import * 
from grad.utils import get_libero_image, save_image


def load_demo_state(hdf5_path):
    with h5py.File(hdf5_path, "r") as f:
        # 你现在只有 demo_1；如果未来有多个 demo_x，可以改成遍历
        states = f["data"]["demo_1"]["states"][:]
    return states

def try_getters(env):
    """返回一个 dict: {name: np.ndarray}，尽可能多地拿到各种 state 表示"""
    out = {}

    # 1) 常见 env 层 getter
    for name in ["get_state", "get_flattened_state", "get_env_state"]:
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                s = fn()
                out[f"env.{name}()"] = np.asarray(s).copy()
            except Exception as e:
                out[f"env.{name}()_ERR"] = repr(e)

    # 2) 常见 sim getter
    sim = getattr(env, "sim", None)
    if sim is not None:
        # mujoco_py / dm_control 风格
        fn = getattr(sim, "get_state", None)
        if callable(fn):
            try:
                s = fn()
                out["sim.get_state()"] = s  # 可能不是 ndarray
            except Exception as e:
                out["sim.get_state()_ERR"] = repr(e)

        # qpos qvel
        data = getattr(sim, "data", None)
        if data is not None:
            try:
                qpos = np.asarray(data.qpos).copy()
                qvel = np.asarray(data.qvel).copy()
                out["sim.data.qpos"] = qpos
                out["sim.data.qvel"] = qvel
                out["qpos+qvel(flat)"] = np.concatenate([qpos, qvel], axis=0)
            except Exception as e:
                out["sim.data.qpos/qvel_ERR"] = repr(e)

        # model sizes (帮助判断 71 是不是 qpos+qvel)
        model = getattr(sim, "model", None)
        if model is not None:
            try:
                out["sim.model.nq"] = int(model.nq)
                out["sim.model.nv"] = int(model.nv)
            except Exception as e:
                out["sim.model.nq/nv_ERR"] = repr(e)

    return out

def try_setters(env, demo_state_71):
    """按可能接口逐个尝试 set，并返回成功的方式字符串；全失败则返回 None"""
    demo_state_71 = np.asarray(demo_state_71).astype(np.float64).ravel()

    # 1) env 层 setter（最可能直接吃 71 维）
    for name in ["set_state", "set_flattened_state", "set_env_state"]:
        fn = getattr(env, name, None)
        if callable(fn):
            try:
                fn(demo_state_71)
                return f"env.{name}(71)"
            except Exception:
                pass

    # 2) sim 层：如果 71 = nq+nv，就拆开写 qpos/qvel
    sim = getattr(env, "sim", None)
    if sim is not None and hasattr(sim, "model"):
        nq = int(sim.model.nq)
        nv = int(sim.model.nv)
        if nq + nv == demo_state_71.shape[0]:
            qpos = demo_state_71[:nq]
            qvel = demo_state_71[nq:nq+nv]
            try:
                sim.data.qpos[:] = qpos
                sim.data.qvel[:] = qvel
                sim.forward()
                return f"sim.data.qpos/qvel (split {nq}+{nv})"
            except Exception:
                pass

    # 3) sim.set_state：需要 mujoco state 对象，71 大概率对不上，但也试一次
    if sim is not None:
        fn = getattr(sim, "set_state", None)
        if callable(fn):
            try:
                fn(demo_state_71)  # 很可能报错
                sim.forward()
                return "sim.set_state(raw71)"
            except Exception:
                pass
    
    if demo_state_71.shape[0] == (nq + nv + 1):
        candidate = demo_state_71[1:]   # 去掉第一维
        qpos = candidate[:nq]
        qvel = candidate[nq:nq+nv]
        sim.data.qpos[:] = qpos
        sim.data.qvel[:] = qvel
        sim.forward()
        return "sim.data.qpos/qvel (demo[1:])"

    return None

def render_one_frame(env, out_path="debug_render.png"):
    # 不同 wrapper 的 render 返回值可能不一样
    frame = None
    try:
        frame = env.render()
    except Exception:
        pass

    # 常见：env.render(mode="rgb_array")
    if frame is None:
        try:
            frame = env.render(mode="rgb_array")
        except Exception:
            pass

    if frame is None:
        print("[WARN] render() 没拿到 rgb_array（可能只支持 onscreen viewer）。")
        return False

    # 保存
    from PIL import Image
    Image.fromarray(frame).save(out_path)
    print(f"[OK] Saved render to {out_path}")
    return True

def make_env(bddl_file_path):
    """
    根据 bddl_file_path 初始化对应的 LIBERO 环境，返回 env 对象。
    这里直接参考 evaluation/task_runner.py 中的环境初始化代码，确保参数一致。
    """
    problem_info = BDDLUtils.get_problem_info(bddl_file_path)
    problem_name = problem_info["problem_name"]
    controller_config = load_controller_config(default_controller="OSC_POSE")
    env = TASK_MAPPING[problem_name](
        bddl_file_name=bddl_file_path,
        robots=["Panda"],
        controller_configs=controller_config,
        has_renderer=False,
        has_offscreen_renderer=True,
        camera_names=["agentview", "robot0_eye_in_hand"],
        ignore_done=True,
        use_camera_obs=True,
        reward_shaping=True,
        camera_heights=256,
        camera_widths=256,
        control_freq=20,
    )
    return env
    

def main():
    hdf5_path = sys.argv[1]
    demo_states = load_demo_state(hdf5_path)
    print("demo_states shape:", demo_states.shape)  # (T,71)

    env = make_env(bddl_file_path="/home/shenhaotian/grad/RoboCerebra/RoboCerebra_Bench/Ideal/case1/COFFEE_TABLESCENE_organize_selected_food_items_into_the_white_storage_box.bddl")
    obs = env.reset()

    print("\n=== After reset: probing getters ===")
    got = try_getters(env)
    for k, v in got.items():
        if isinstance(v, np.ndarray):
            print(f"{k}: shape={v.shape} dtype={v.dtype}")
        else:
            print(f"{k}: {v}")

    print("\n=== Try set demo_states[0] back ===")
    method = try_setters(env, demo_states[50])
    if method is None:
        print("[FAIL] No setter method accepted the 71-dim state.")
        print("Next step: compare 71 with (nq+nv) and/or search your env wrapper for state serialization.")
        return

    print("[OK] set succeeded via:", method)
    obs = env._get_observations()
    img = get_libero_image(obs)
    save_image(img, "grad/set_state_render.png")
    print("Saved set_state_render.png")

    # 尝试渲染一帧（验收点）
    # render_one_frame(env, "state0_render.png")
    env.close()

if __name__ == "__main__":
    main()
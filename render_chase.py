# -*- coding: utf-8 -*-
"""渲染 Ant-v5 策略（跟随相机版）：输出 mp4/gif + 指标。

修复：mjv_defaultFreeCamera 固定在原点，机器人走远了就出画面。
这里改用 tracking camera，始终跟随 torso 并保持机位朝向。
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train import ActorCritic


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/ant_standard")
    ap.add_argument("--seconds", type=float, default=15.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=str, default="rollout_standard")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--distance", type=float, default=4.0)
    ap.add_argument("--elevation", type=float, default=-18.0)
    ap.add_argument("--azimuth", type=float, default=140.0)
    args = ap.parse_args()

    import gymnasium as gym
    import mujoco

    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=False)
    net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    net.load_state_dict(ckpt["net"])
    net.eval()

    env = gym.make("Ant-v5")
    raw = env.unwrapped
    model, data = raw.model, raw.data
    renderer = mujoco.Renderer(model, height=480, width=640)

    cam = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(cam)
    cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    cam.trackbodyid = 1                      # Ant 的 torso body id = 1
    cam.distance = args.distance
    cam.elevation = args.elevation
    cam.azimuth = args.azimuth

    obs, _ = env.reset(seed=args.seed)
    frames, vxs, rewards, heights, xs = [], [], [], [], []

    dt_per_step = 5 * model.opt.timestep
    n_steps = int(args.seconds / dt_per_step)

    for t in range(n_steps):
        with torch.no_grad():
            act = net.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        obs, rew, term, trunc, _ = env.step(np.clip(act, -1, 1))
        vxs.append(float(raw.data.qvel[0]))
        heights.append(float(raw.data.qpos[2]))
        xs.append(float(raw.data.qpos[0]))
        rewards.append(float(rew))
        if t % 2 == 0:
            renderer.update_scene(data, camera=cam)
            frames.append(renderer.render().copy())
        if term or trunc:
            break

    renderer.close()
    env.close()

    out = Path(args.out)
    wrote = None
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(out) + ".mp4", frames, fps=args.fps, macro_block_size=1)
        wrote = str(out) + ".mp4"
    except Exception as e:
        print(f"[warn] mp4 failed: {e}; fallback gif")
        from PIL import Image
        imgs = [Image.fromarray(f) for f in frames]
        imgs[0].save(str(out) + ".gif", save_all=True, append_images=imgs[1:],
                     duration=int(1000 / args.fps), loop=0)
        wrote = str(out) + ".gif"

    vx = np.array(vxs)
    metrics = {
        "policy": args.run,
        "steps": int(len(vx)),
        "duration_s": round(len(vx) * dt_per_step, 2),
        "mean_vx_m_s": round(float(vx.mean()), 3),
        "max_vx_m_s": round(float(vx.max()), 3),
        "distance_m": round(float(xs[-1] - xs[0]), 2),
        "mean_torso_height_m": round(float(np.mean(heights)), 3),
        "mean_reward_per_step": round(float(np.mean(rewards)), 3),
        "survived_full": bool(len(vx) >= n_steps),
        "frames": len(frames),
        "video": wrote,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

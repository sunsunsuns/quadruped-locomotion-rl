# -*- coding: utf-8 -*-
"""评估 + 渲染：加载训练好的策略，跑 rollout，输出视频与指标。

用法：
  python render.py --run runs/ant_veltrack --seconds 10 --out rollout
输出：
  rollout.mp4 / rollout.gif   策略行走视频
  metrics.json                速度跟踪误差、存活时长等
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from train import ActorCritic, make_env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=str, default="runs/ant_veltrack")
    ap.add_argument("--seconds", type=float, default=10.0)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--out", type=str, default="rollout")
    ap.add_argument("--v-target", type=float, default=None)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    run_dir = Path(args.run)
    ckpt = torch.load(run_dir / "policy.pt", map_location="cpu", weights_only=False)
    trained_args = ckpt.get("args", {})
    v_target = args.v_target if args.v_target is not None else trained_args.get("v_target", 1.0)

    net = ActorCritic(ckpt["obs_dim"], ckpt["act_dim"])
    net.load_state_dict(ckpt["net"])
    net.eval()

    env = make_env(args.seed, v_target)()

    # 用 mujoco 的离屏渲染出帧
    import mujoco
    raw = env.unwrapped
    model, data = raw.model, raw.data
    renderer = mujoco.Renderer(model, height=480, width=640)

    obs, _ = env.reset(seed=args.seed)
    frames, vxs, rewards = [], [], []
    n_steps = int(args.seconds * 1000 / model.opt.timestep / 5)  # Ant-v5 每 step 5 个 mj step
    step_count = 0
    done = False

    while step_count < n_steps and not done:
        with torch.no_grad():
            act = net.actor(torch.as_tensor(obs, dtype=torch.float32)).numpy()
        obs, rew, term, trunc, info = env.step(np.clip(act, -1, 1))
        vxs.append(info.get("vx", np.nan))
        rewards.append(rew)
        if step_count % 2 == 0:  # 每 2 个决策步取一帧，控制视频体积
            renderer.update_scene(data, camera=-1)
            frames.append(renderer.render().copy())
        step_count += 1
        done = term or trunc

    renderer.close()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fps = args.fps

    # 优先 mp4（imageio-ffmpeg），失败则 gif
    wrote = None
    try:
        import imageio.v2 as imageio
        imageio.mimsave(str(out) + ".mp4", frames, fps=fps, macro_block_size=1)
        wrote = str(out) + ".mp4"
    except Exception as e:
        print(f"[warn] mp4 failed ({e}), fallback to gif")
        try:
            from PIL import Image
            imgs = [Image.fromarray(f) for f in frames]
            imgs[0].save(str(out) + ".gif", save_all=True, append_images=imgs[1:],
                         duration=int(1000 / fps), loop=0)
            wrote = str(out) + ".gif"
        except Exception as e2:
            print(f"[error] gif failed too: {e2}")

    vx_arr = np.array(vxs, dtype=float)
    metrics = {
        "v_target": v_target,
        "steps": int(step_count),
        "survived_s": round(step_count * 5 * model.opt.timestep, 2),
        "mean_vx": round(float(np.nanmean(vx_arr)), 3),
        "std_vx": round(float(np.nanstd(vx_arr)), 3),
        "mean_abs_track_err": round(float(np.nanmean(np.abs(vx_arr - v_target))), 3),
        "frames": len(frames),
        "video": wrote,
    }
    (run_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2),
                                          encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

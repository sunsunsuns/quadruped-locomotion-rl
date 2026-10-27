# -*- coding: utf-8 -*-
"""四足机器人 locomotion（速度跟踪）强化学习训练 —— MuJoCo Ant + PPO。

为什么用 MuJoCo Ant：
  - 它是标准四足 locomotion 基准（Isaac Lab 里的 Ant 任务同源），
    面试时"我在 MuJoCo 上训过四足速度跟踪策略"是完全可信、可对标的经历。
  - 纯 CPU 可训，不依赖 8GB+ 显存，适合 4GB 笔记本显卡先跑通闭环。

奖励设计（速度跟踪，对应 JD 里的 "velocity tracking / 站立 / 行走"）：
  r = 1.0 (存活) + 2.0 * exp(-4*(vx - v_target)^2) - 1.0*vy^2 - 0.1*wz^2 - 0.1*||a||^2

用法：
  python train.py --steps 1000000            # 训练
  python train.py --steps 20000 --tag smoke  # 快速自检
"""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium.vector import SyncVectorEnv


# ---------------------------------------------------------------- environment
def make_env(seed, v_target=1.0):
    import gymnasium as gym

    class VelTrackAnt(gym.Wrapper):
        """给 Ant-v5 套一层"速度跟踪"奖励与终止逻辑。"""

        def __init__(self):
            super().__init__(gym.make("Ant-v5"))
            self.v_target = v_target

        def reset(self, **kw):
            obs, info = self.env.reset(**kw)
            return obs, info

        def step(self, action):
            obs, _r, terminated, truncated, info = self.env.step(action)
            # Ant-v5 的 obs 布局：qpos(15) + qvel(14) + cfrc(84)
            qvel = obs[15:29]
            qpos = obs[:15]
            vx, vy = float(qvel[0]), float(qvel[1])
            wz = float(qvel[5])
            track = np.exp(-4.0 * (vx - self.v_target) ** 2)
            ctrl = float(np.sum(np.square(action)))
            # 躯干直立度：qpos[1] 是躯干四元数的 w 分量（未归一化），w 越接近 1 越直立。
            # 用 1-w^2 近似"倾倒程度"，避免策略靠"扑街前冲"骗速度奖励。
            upright = max(0.0, 1.0 - float(qpos[1]) ** 2)
            # v3 奖励：以「前进 + 跟踪 + 直立」为主，存活只给低保。
            # 教训链（很值得写进项目记录，面试也能讲）：
            #   v1 存活1.0+跟踪2.0        -> 收敛到"站着不动"（不动也能拿 ~1.0/步）
            #   v2 前进1.5+存活1.0        -> "扑街式前进"（vx 高但 ep_len 30~90）
            #   v2b 存活2.5+前进1.0+直立  -> 直接学会"站着"（存活权重太高压倒一切）
            #   v3 存活1.0+前进2.0+跟踪2.0+直立1.0 -> 移动收益显著高于不动，且用直立项防摔倒
            reward = (
                1.0                      # 存活低保
                + 2.0 * vx               # 前进（主驱动）
                + 2.0 * track            # 速度跟踪（主驱动，抑制冲过头）
                + 1.0 * upright          # 直立（防止"扑街式前冲"）
                - 1.0 * vy ** 2          # 侧向漂移
                - 0.1 * wz ** 2          # 自转
                - 0.05 * ctrl            # 能耗（放轻）
            )
            if not np.isfinite(obs).all():
                terminated = True
            info["vx"] = vx
            info["track"] = float(track)
            return obs, float(reward), terminated, truncated, info

    def thunk():
        env = VelTrackAnt()
        env.reset(seed=seed)
        try:
            env.action_space.seed(seed)
        except Exception:
            pass
        return env

    return thunk


# ---------------------------------------------------------------- networks
def layer_init(layer, std=np.sqrt(2), bias=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias)
    return layer


class ActorCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=128):
        super().__init__()
        self.actor = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, act_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, hidden)), nn.Tanh(),
            layer_init(nn.Linear(hidden, 1), std=1.0),
        )
        self.log_std = nn.Parameter(torch.zeros(act_dim))

    def dist(self, x):
        mu = self.actor(x)
        std = torch.exp(self.log_std).expand_as(mu)
        return torch.distributions.Normal(mu, std)

    def value(self, x):
        return self.critic(x).squeeze(-1)


# ---------------------------------------------------------------- PPO
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-env", type=int, default=8)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--minibatch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.005)
    ap.add_argument("--v-target", type=float, default=1.0)
    ap.add_argument("--tag", type=str, default="vel_track_ant")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    run_dir = Path("runs") / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    envs = SyncVectorEnv([make_env(args.seed + i, args.v_target) for i in range(args.n_env)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    print(f"[env] Ant-v5  obs={obs_dim}  act={act_dim}  n_env={args.n_env}  "
          f"v_target={args.v_target} m/s  threads={torch.get_num_threads()}")

    net = ActorCritic(obs_dim, act_dim)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, dtype=torch.float32)

    global_step = 0
    update = 0
    ep_returns, ep_lengths, ep_vx, ep_track = [], [], [], []
    cur_ret = np.zeros(args.n_env)
    cur_len = np.zeros(args.n_env)

    t0 = time.time()
    history = []
    batch_obs = int(args.n_env * args.rollout)
    n_updates = max(1, args.steps // batch_obs)

    for update in range(1, n_updates + 1):
        o_buf, a_buf, lp_buf, r_buf, d_buf, v_buf = [], [], [], [], [], []
        for _ in range(args.rollout):
            with torch.no_grad():
                dist = net.dist(obs)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = net.value(obs)
            a_np = act.clamp(-1, 1).numpy()
            nxt, rew, term, trunc, infos = envs.step(a_np)
            done = np.logical_or(term, trunc)

            o_buf.append(obs)
            a_buf.append(act)
            lp_buf.append(logp)
            r_buf.append(torch.as_tensor(rew, dtype=torch.float32))
            d_buf.append(torch.as_tensor(done.astype(np.float32)))
            v_buf.append(val)

            cur_ret += rew
            cur_len += 1
            for i, d in enumerate(done):
                if d:
                    ep_returns.append(cur_ret[i])
                    ep_lengths.append(cur_len[i])
                    ep_vx.append(infos.get("vx", np.zeros(args.n_env))[i] if isinstance(infos, dict) and "vx" in infos else np.nan)
                    ep_track.append(infos.get("track", np.zeros(args.n_env))[i] if isinstance(infos, dict) and "track" in infos else np.nan)
                    cur_ret[i] = 0.0
                    cur_len[i] = 0.0

            obs = torch.as_tensor(nxt, dtype=torch.float32)
            global_step += args.n_env

        # ---- GAE
        with torch.no_grad():
            last_val = net.value(obs)
        rew_t = torch.stack(r_buf)          # [T, N]
        done_t = torch.stack(d_buf)
        val_t = torch.stack(v_buf)
        adv = torch.zeros_like(rew_t)
        gae = torch.zeros(args.n_env)
        for t in reversed(range(args.rollout)):
            nxt_val = last_val if t == args.rollout - 1 else val_t[t + 1]
            nonterm = 1.0 - done_t[t]
            delta = rew_t[t] + args.gamma * nxt_val * nonterm - val_t[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + val_t

        b_obs = torch.stack(o_buf).reshape(-1, obs_dim)
        b_act = torch.stack(a_buf).reshape(-1, act_dim)
        b_logp = torch.stack(lp_buf).reshape(-1)
        b_adv = adv.reshape(-1)
        b_ret = ret.reshape(-1)
        b_adv = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)

        idx = torch.randperm(b_obs.shape[0])
        for _ in range(args.epochs):
            for start in range(0, b_obs.shape[0], args.minibatch):
                mb = idx[start:start + args.minibatch]
                dist = net.dist(b_obs[mb])
                logp = dist.log_prob(b_act[mb]).sum(-1)
                ratio = torch.exp(logp - b_logp[mb])
                s1 = ratio * b_adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[mb]
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (net.value(b_obs[mb]) - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss + 0.5 * v_loss - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if update % 5 == 0 or update == n_updates:
            recent_r = float(np.mean(ep_returns[-20:])) if ep_returns else float("nan")
            recent_l = float(np.mean(ep_lengths[-20:])) if ep_lengths else float("nan")
            recent_vx = float(np.nanmean(ep_vx[-20:])) if ep_vx else float("nan")
            recent_tr = float(np.nanmean(ep_track[-20:])) if ep_track else float("nan")
            elapsed = time.time() - t0
            print(f"[{update:4d}/{n_updates}] step={global_step:>9,}  "
                  f"ret={recent_r:8.1f}  ep_len={recent_l:6.1f}  "
                  f"vx={recent_vx:5.2f}  track={recent_tr:4.2f}  "
                  f"{global_step/max(elapsed,1e-9)/1000:.1f}k steps/s")
            history.append(dict(update=update, step=global_step, ret=recent_r,
                                ep_len=recent_l, vx=recent_vx, track=recent_tr))
            torch.save({"net": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
                        "args": vars(args)}, run_dir / "policy.pt")
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save({"net": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
                "args": vars(args)}, run_dir / "policy.pt")
    print(f"\n[done] {global_step:,} steps in {time.time()-t0:.1f}s")
    print(f"[ckpt] {run_dir / 'policy.pt'}")


if __name__ == "__main__":
    main()

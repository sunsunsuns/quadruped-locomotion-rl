# -*- coding: utf-8 -*-
"""对照实验：用 Ant-v5 原生奖励训练（不改造奖励），验证 PPO 能否学出四足步态。

目的：区分两种可能
  A. 我的 PPO 实现/超参有问题 -> 原版奖励也学不出东西
  B. 是我的自定义奖励设计导致局部最优 -> 原版奖励应该能学出连续步态

用法：python train_standard.py --steps 1000000 --tag ant_standard
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

from train import ActorCritic, layer_init


def make_env(seed):
    import gymnasium as gym

    def thunk():
        env = gym.make("Ant-v5")   # 使用官方奖励，不做任何改造
        env.reset(seed=seed)
        return env

    return thunk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=1_000_000)
    ap.add_argument("--n-env", type=int, default=16)
    ap.add_argument("--rollout", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--minibatch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.99)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--ent-coef", type=float, default=0.0)
    ap.add_argument("--tag", type=str, default="ant_standard")
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.set_num_threads(max(1, os.cpu_count() - 1))

    run_dir = Path("runs") / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)

    envs = SyncVectorEnv([make_env(args.seed + i) for i in range(args.n_env)])
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    act_dim = int(np.prod(envs.single_action_space.shape))
    print(f"[env] Ant-v5 (原生奖励) obs={obs_dim} act={act_dim} n_env={args.n_env}")

    net = ActorCritic(obs_dim, act_dim)
    # 原生奖励下动作尺度较大，log_std 初始化略大一点利于探索
    with torch.no_grad():
        net.log_std.fill_(-0.5)
    opt = optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    obs, _ = envs.reset(seed=args.seed)
    obs = torch.as_tensor(obs, dtype=torch.float32)

    batch = args.n_env * args.rollout
    n_updates = max(1, args.steps // batch)
    cur_ret = np.zeros(args.n_env)
    cur_len = np.zeros(args.n_env)
    ep_ret, ep_len = [], []
    t0 = time.time()
    history = []
    global_step = 0

    for update in range(1, n_updates + 1):
        o_buf, a_buf, lp_buf, r_buf, d_buf, v_buf = [], [], [], [], [], []
        for _ in range(args.rollout):
            with torch.no_grad():
                dist = net.dist(obs)
                act = dist.sample()
                logp = dist.log_prob(act).sum(-1)
                val = net.value(obs)
            nxt, rew, term, trunc, _ = envs.step(act.numpy())
            done = np.logical_or(term, trunc)
            o_buf.append(obs); a_buf.append(act); lp_buf.append(logp)
            r_buf.append(torch.as_tensor(rew, dtype=torch.float32))
            d_buf.append(torch.as_tensor(done.astype(np.float32)))
            v_buf.append(val)
            cur_ret += rew; cur_len += 1
            for i, d in enumerate(done):
                if d:
                    ep_ret.append(cur_ret[i]); ep_len.append(cur_len[i])
                    cur_ret[i] = 0.0; cur_len[i] = 0.0
            obs = torch.as_tensor(nxt, dtype=torch.float32)
            global_step += args.n_env

        with torch.no_grad():
            last_val = net.value(obs)
        rew_t = torch.stack(r_buf); done_t = torch.stack(d_buf); val_t = torch.stack(v_buf)
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
            for s in range(0, b_obs.shape[0], args.minibatch):
                mb = idx[s:s + args.minibatch]
                dist = net.dist(b_obs[mb])
                logp = dist.log_prob(b_act[mb]).sum(-1)
                ratio = torch.exp(logp - b_logp[mb])
                s1 = ratio * b_adv[mb]
                s2 = torch.clamp(ratio, 1 - args.clip, 1 + args.clip) * b_adv[mb]
                pg_loss = -torch.min(s1, s2).mean()
                v_loss = 0.5 * (net.value(b_obs[mb]) - b_ret[mb]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pg_loss + 0.5 * v_loss - args.ent_coef * ent
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()

        if update % 5 == 0 or update == n_updates:
            rr = float(np.mean(ep_ret[-20:])) if ep_ret else float("nan")
            rl = float(np.mean(ep_len[-20:])) if ep_len else float("nan")
            el = time.time() - t0
            print(f"[{update:4d}/{n_updates}] step={global_step:>9,} ret={rr:8.1f} "
                  f"ep_len={rl:6.1f} {global_step/max(el,1e-9)/1000:.1f}k/s")
            history.append(dict(update=update, step=global_step, ret=rr, ep_len=rl))
            torch.save({"net": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
                        "args": vars(args)}, run_dir / "policy.pt")
            (run_dir / "history.json").write_text(
                json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")

    torch.save({"net": net.state_dict(), "obs_dim": obs_dim, "act_dim": act_dim,
                "args": vars(args)}, run_dir / "policy.pt")
    print(f"\n[done] {global_step:,} steps in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

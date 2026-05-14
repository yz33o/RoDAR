import argparse
import os
from datetime import datetime
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from algo.ppo.PPO import PPO
from utils_low_multimerge import make_env, set_random_seed

parser = argparse.ArgumentParser()
args = parser.parse_args([])

args.env_name = 'merge'
args.seed = 1234
args.gamma = 0.995
args.max_training_steps = 1e6
args.batch_size = 3000
args.max_ep_len = 5000
args.render = False
args.log_interval = 1

# PPO 特有参数（CAV policy）
args.lr_actor = 2e-4
args.lr_critic = 1e-3
args.eps_clip = 0.2
args.K_epochs = 20
args.has_continuous_action_space = True
args.action_std = 0.6

# -------------------- 新增：Obs-adv（分散式观测扰动） --------------------
# 逐维扰动上限（L∞ 约束）：[ego_v, lead_rel_v, lead_head, follow_rel_v, follow_head]
args.obs_adv_eps = [0.03, 0.03, 0.02, 0.03, 0.02]
# 扰动能量惩罚系数：越大 => 对抗者越“克制”（更像真实噪声）
args.obs_adv_lambda = 0.2
# Obs-adv PPO 超参
args.obs_adv_lr_actor = 3e-4
args.obs_adv_lr_critic = 1e-3
args.obs_adv_gamma = 0.99
args.obs_adv_K_epochs = 10
args.obs_adv_eps_clip = 0.2
# Obs-adv 初始 std（建议小一点）
args.obs_adv_action_std_init = 0.15

print(args)


class AdvPPO:
    def __init__(self, state_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, K_epochs=10, eps_clip=0.2):
        # 动作为标量（扰动加速度）
        self.agent = PPO(
            state_dim=state_dim,
            action_dim=1,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            K_epochs=K_epochs,
            eps_clip=eps_clip,
            has_continuous_action_space=True,
            action_std_init=0.4
        )

        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()

    def select_action(self, state):
        return self.agent.select_action(state)

    def store(self, s, a, r, mask):
        self.states.append(s)
        self.actions.append([a])
        self.rewards.append(r)
        self.masks.append(mask)

    def update(self):
        self.agent.buffer.states = [torch.FloatTensor(s) for s in self.states]
        self.agent.buffer.actions = [torch.FloatTensor(a) for a in self.actions]
        self.agent.buffer.rewards = self.rewards
        self.agent.buffer.is_terminals = [1 - m for m in self.masks]

        self.agent.update()
        self.clear()


class ObsAdvPPO:
    def __init__(self, state_dim, lr_actor=1e-4, lr_critic=1e-3,
                 gamma=0.99, K_epochs=10, eps_clip=0.2, action_std_init=0.15):
        self.state_dim = int(state_dim)

        self.agent = PPO(
            state_dim=self.state_dim,
            action_dim=self.state_dim,  # 输出 delta 向量
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            K_epochs=K_epochs,
            eps_clip=eps_clip,
            has_continuous_action_space=True,
            action_std_init=action_std_init
        )

        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()

    def select_action(self, obs):
        delta = self.agent.select_action(obs)
        if isinstance(delta, torch.Tensor):
            delta = delta.detach().cpu().numpy()
        delta = np.asarray(delta, dtype=np.float32).reshape(-1)
        if delta.shape[0] != self.state_dim:
            raise ValueError(f"ObsAdvPPO.select_action: expect dim {self.state_dim}, got {delta.shape}")
        return delta

    def store(self, s, a, r, mask):
        self.states.append(np.asarray(s, dtype=np.float32))
        self.actions.append(np.asarray(a, dtype=np.float32))
        self.rewards.append(float(r))
        self.masks.append(int(mask))

    def update(self):
        self.agent.buffer.states = [torch.FloatTensor(s) for s in self.states]
        self.agent.buffer.actions = [torch.FloatTensor(a) for a in self.actions]
        self.agent.buffer.rewards = self.rewards
        self.agent.buffer.is_terminals = [1 - m for m in self.masks]

        self.agent.update()
        self.clear()


def _make_eps_vec(eps, dim):
    if isinstance(eps, (int, float)):
        return np.full((dim,), float(eps), dtype=np.float32)
    eps = np.asarray(eps, dtype=np.float32).reshape(-1)
    if eps.shape[0] != dim:
        raise ValueError(f"obs_adv_eps dim mismatch: expect {dim}, got {eps.shape[0]}")
    return eps


def _clip_obs_to_plausible_range(obs):
    low = np.array([0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32)
    high = np.array([1.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    return np.clip(obs, low, high)


env = make_env(render=args.render)
set_random_seed(args.seed, env)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]

ppo_agent = PPO(
    state_dim, action_dim,
    args.lr_actor, args.lr_critic,
    args.gamma, args.K_epochs, args.eps_clip,
    args.has_continuous_action_space, args.action_std
)

adv_state_dim = len(env.get_global_state())
adv_agent = AdvPPO(
    state_dim=adv_state_dim,
    lr_actor=1e-4,
    lr_critic=1e-3,
    gamma=0.99,
    K_epochs=10,
    eps_clip=0.2
)

obs_eps_vec = _make_eps_vec(args.obs_adv_eps, state_dim)
obs_adv_agent = ObsAdvPPO(
    state_dim=state_dim,
    lr_actor=args.obs_adv_lr_actor,
    lr_critic=args.obs_adv_lr_critic,
    gamma=args.obs_adv_gamma,
    K_epochs=args.obs_adv_K_epochs,
    eps_clip=args.obs_adv_eps_clip,
    action_std_init=args.obs_adv_action_std_init
)

tag = f'rs{args.seed}'
writer = SummaryWriter(f'logs/{args.env_name}/AD_doubleMA_PPO_high_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

checkpoint_dir = f'checkpoint/{args.env_name}/AD_doubleMA_PPO_high_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(checkpoint_dir, exist_ok=True)


class MultiAgentMemory:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.masks = []

    def push(self, s, a, r, mask):
        self.states.append(s)
        self.actions.append(a)
        self.rewards.append(r)
        self.masks.append(mask)

    def clear(self):
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.masks.clear()


total_step = 0
i_episode = 0

while total_step < args.max_training_steps:

    memory = MultiAgentMemory()
    reward_batch = []
    speed_mean_batch, speed_std_batch = [], []
    fuel_batch, radical_accel_batch = [], []
    ramp_entry_time = {}
    merge_finish_time = {}
    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    print(">>>>>>>>>>>>>>> Interacting with Multi-Agent Env <<<<<<<<<<<<<<")

    num_steps = 0
    while num_steps < args.batch_size:

        state = env.reset()
        reward_sum = 0
        speed_list, accel_list, fuel_list = [], [], []
        radical_accel_count = 0

        for t in range(args.max_ep_len):

            rl_ids = env.k.vehicle.get_rl_ids()
            for rid in rl_ids:
                edge = env.k.vehicle.get_edge(rid)

                if edge == "inflow_merge" and rid not in ramp_entry_time:
                    ramp_entry_time[rid] = t

                if rid in ramp_entry_time and rid not in merge_finish_time:
                    if edge != "inflow_merge":
                        merge_finish_time[rid] = t

            if len(rl_ids) == 0:
                next_state, _, _, _ = env.step({})
                continue

            valid_rl_ids = [rid for rid in rl_ids if rid in state]

            if len(valid_rl_ids) == 0:
                next_state, _, _, _ = env.step({})
                state = next_state
                continue

            actions = {}
            obs_before = {}     # rid -> o
            delta_obs_dict = {} # rid -> delta
            obs_after = {}      # rid -> o_tilde

            for rid in valid_rl_ids:
                o = np.asarray(state[rid], dtype=np.float32).reshape(-1)
                o = _clip_obs_to_plausible_range(o)

                delta = obs_adv_agent.select_action(o)  # 5-dim
                delta = np.clip(delta, -obs_eps_vec, obs_eps_vec)

                o_tilde = _clip_obs_to_plausible_range(o + delta)

                obs_before[rid] = o
                delta_obs_dict[rid] = delta
                obs_after[rid] = o_tilde

                a = ppo_agent.select_action(o_tilde)  # 用扰动后的观测选动作
                actions[rid] = a

                if a < -2.0:
                    radical_accel_count += 1

            global_state = env.get_global_state()
            delta_adv = float(adv_agent.select_action(global_state))
            delta_adv = np.clip(delta_adv, -2, 1)
            env._apply_adv_disturbance(delta_adv)

            next_state, reward, done, _ = env.step(actions)

            veh_ids = env.k.vehicle.get_ids()
            if len(veh_ids) > 0:
                v_list = [
                    v for v in env.k.vehicle.get_speed(veh_ids)
                    if v is not None and not np.isnan(v) and v >= 0
                ]
                if len(v_list) > 0:
                    speed_list.append(np.mean(v_list))

                a_list = []
                for rid in rl_ids:
                    try:
                        a_tmp = env.k.vehicle.get_acc_controller(rid).get_accel(env)
                        if a_tmp is not None and not np.isnan(a_tmp):
                            a_list.append(a_tmp)
                    except Exception:
                        pass
                if len(a_list) > 0:
                    accel_list.extend(a_list)

                if len(v_list) > 0 and len(a_list) > 0:
                    v_mean = np.mean(v_list)
                    a_mean = np.mean(a_list)

            for rid in valid_rl_ids:
                reward_rid = reward.get(rid, 0.0) if isinstance(reward, dict) else float(reward)
                mask = 0 if done.get(rid, False) or done.get('__all__', False) else 1

                memory.push(
                    obs_after[rid],
                    np.array([actions[rid]]),
                    reward_rid,
                    mask
                )
                reward_sum += reward_rid

                # 存储 obs-adv 经验（reward 反向 + 扰动惩罚）
                delta = delta_obs_dict[rid]
                delta_normed = delta / (obs_eps_vec + 1e-8)
                delta_pen = float(np.mean(np.square(delta_normed)))
                obs_adv_r = -float(reward_rid) - float(args.obs_adv_lambda) * delta_pen
                obs_adv_agent.store(obs_before[rid], delta, obs_adv_r, mask)

            if isinstance(reward, dict) and len(reward) > 0:
                adv_r = -float(np.mean(list(reward.values())))
            else:
                adv_r = -float(reward) if reward is not None else 0.0
            adv_mask = 0 if done.get('__all__', False) else 1
            adv_agent.store(global_state, delta_adv, adv_r, adv_mask)

            if done.get('__all__', False):
                break

            state = next_state
            total_step += 1

        num_steps += (t + 1)

        reward_batch.append(reward_sum)
        merge_times = []
        for rid in merge_finish_time:
            if rid in ramp_entry_time:
                merge_times.append((merge_finish_time[rid] - ramp_entry_time[rid]) * sim_step)
        avg_merge_time = np.mean(merge_times) if len(merge_times) > 0 else 0.0

        if len(speed_list) > 0:
            speed_mean_batch.append(np.mean(speed_list))
            speed_std_batch.append(np.std(speed_list))

        radical_accel_batch.append(radical_accel_count)
        if len(fuel_list) > 0:
            fuel_batch.append(np.mean(fuel_list))

    writer.add_scalar('Reward/Average_reward', np.mean(reward_batch), total_step)
    writer.add_scalar('Speed/Average_speed', np.mean(speed_mean_batch), total_step)
    writer.add_scalar('Merge time/Average_merge_time', np.mean(merge_times), total_step)

    ppo_agent.buffer.states = [torch.FloatTensor(s) for s in memory.states]
    ppo_agent.buffer.actions = [torch.FloatTensor(a) for a in memory.actions]
    ppo_agent.buffer.rewards = memory.rewards
    ppo_agent.buffer.is_terminals = [1 - m for m in memory.masks]
    ppo_agent.update()

    adv_agent.update()
    obs_adv_agent.update()

    memory.clear()

    torch.save(ppo_agent.policy_old.state_dict(), checkpoint_dir + f'policy_{i_episode}.pth')
    torch.save(adv_agent.agent.policy_old.state_dict(), checkpoint_dir + f'adv_policy_{i_episode}.pth')
    torch.save(obs_adv_agent.agent.policy_old.state_dict(), checkpoint_dir + f'obs_adv_policy_{i_episode}.pth')

    print(
        f'Episode {i_episode}\tAverage reward: {np.mean(reward_batch):.2f}\t'
        f'Average speed {np.mean(speed_mean_batch):.2f}m/s\t'
        f'Average merge time: {np.mean(merge_times):.2f}\t'
    )

    i_episode += 1

writer.close()

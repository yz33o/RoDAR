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

# PPO 特有参数
args.lr_actor = 3e-4
args.lr_critic = 1e-3
args.eps_clip = 0.2
args.K_epochs = 20
args.has_continuous_action_space = True
args.action_std = 0.6

print(args)

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

tag = f'rs{args.seed}'
writer = SummaryWriter(f'logs/{args.env_name}/MA_PPO_low_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

checkpoint_dir = f'checkpoint/{args.env_name}/MA_PPO_low_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
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
    radical_accel_batch = []
    merge_time_all = []
    throughput_all = []
    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    print(">>>>>>>>>>>>>>> Interacting with Multi-Agent Env <<<<<<<<<<<<<<")

    num_steps = 0
    while num_steps < args.batch_size:

        state = env.reset()
        reward_sum = 0
        speed_list, accel_list, fuel_list = [], [], []
        radical_accel_count = 0
        throughput = 0
        ramp_entry_time = {}
        merge_finish_time = {}
        exit_count = 0
        prev_ids = set()

        for t in range(args.max_ep_len):
            # throughput
            current_ids = set(env.k.vehicle.get_ids())
            exited = prev_ids - current_ids  # 从上一步消失 → 完成路段
            exit_count += len(exited)
            prev_ids = current_ids
            rl_ids = env.k.vehicle.get_rl_ids()
            # merge 时间
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
            for rid in valid_rl_ids:
                a = ppo_agent.select_action(state[rid])
                actions[rid] = a

                if a > 1.47 or a < -2:
                    radical_accel_count += 1

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
                        a = env.k.vehicle.get_acc_controller(rid).get_accel(env)
                        if a is not None and not np.isnan(a):
                            a_list.append(a)
                    except:
                        pass
                if len(a_list) > 0:
                    accel_list.extend(a_list)


            for rid in valid_rl_ids:
                reward_rid = reward.get(rid, 0) if isinstance(reward, dict) else reward
                mask = 0 if done.get(rid, False) or done.get('__all__', False) else 1

                memory.push(
                    state[rid],
                    np.array([actions[rid]]),
                    reward_rid,
                    mask
                )

                reward_sum += reward_rid
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

        episode_time_sec = (t + 1) * sim_step
        throughput_per_hour = (exit_count / episode_time_sec) * 3600
        throughput_all.append(throughput_per_hour)
        if len(speed_list) > 0:
            speed_mean_batch.append(np.mean(speed_list))
            speed_std_batch.append(np.std(speed_list))

        radical_accel_batch.append(radical_accel_count)

    writer.add_scalar('Reward/Average_reward', np.mean(reward_batch), total_step)
    writer.add_scalar('Speed/Average_speed', np.mean(speed_mean_batch), total_step)
    writer.add_scalar('Merge time/Average_merge_time', np.mean(merge_times), total_step)

    ppo_agent.buffer.states = [torch.FloatTensor(s) for s in memory.states]
    ppo_agent.buffer.actions = [torch.FloatTensor(a) for a in memory.actions]
    ppo_agent.buffer.rewards = memory.rewards
    ppo_agent.buffer.is_terminals = [1 - m for m in memory.masks]

    ppo_agent.update()
    memory.clear()

    torch.save(ppo_agent.policy_old.state_dict(), checkpoint_dir + f'policy_{i_episode}.pth')

    print(f'Episode {i_episode}\tAverage reward:  {np.mean(reward_batch):.2f}\t'
          f'Average speed {np.mean(speed_mean_batch):.2f}m/s\t'
          f'MergeTime={avg_merge_time:.2f} s\t'
          f'Throughput={throughput_per_hour:.1f} veh/h\t'
          f'Radical_accel:{np.mean(radical_accel_batch):.2f}次')

    i_episode += 1

writer.close()


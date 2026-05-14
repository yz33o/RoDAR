# 测试 低密度环境训练模型


import os
import numpy as np
import torch
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from algo.ppo.PPO import PPO
from utils_1220high_multimerge import make_env, set_random_seed


NORMAL_POLICY_PATH =""
SEED = 1234
EPISODES = 30
ENV_NAME = "merge"
OBS_NOISE_EPS = np.array([0.05, 0.05, 0.04, 0.05, 0.04], dtype=np.float32)
OBS_NOISE_ALPHA = 0.8
def load_main_policy(state_dim, action_dim, model_path):
    agent = PPO(
        state_dim=state_dim,
        action_dim=action_dim,
        lr_actor=3e-4,
        lr_critic=1e-3,
        gamma=0.995,
        K_epochs=20,
        eps_clip=0.2,
        has_continuous_action_space=True,
        action_std_init=0.6,
    )
    ckpt = torch.load(model_path, map_location='cpu')
    agent.policy_old.load_state_dict(ckpt)
    print(f"[OK] Loaded main policy from: {model_path}")
    return agent


def evaluate_policy(main_agent, env):
    reward_all = []
    speed_all = []
    speedstd_all = []
    fuel_all = []
    radical_all = []
    collisions_batch = []
    throughput_all = []
    merge_time_all = []

    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    for ep in range(EPISODES):
        state = env.reset()

        ep_reward = 0.0
        speed_list = []
        radical_count = 0
        collisions_count = 0

        prev_ids = set()
        exit_count = 0

        ramp_entry_time = {}
        merge_finish_time = {}

        obs_noise_prev = {}

        for t in range(5000):

            current_ids = set(env.k.vehicle.get_ids())
            exited = prev_ids - current_ids
            exit_count += len(exited)
            prev_ids = current_ids

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
            for rid in valid_rl_ids:
                o = np.asarray(state[rid], dtype=np.float32)

                prev_noise = obs_noise_prev.get(rid, np.zeros_like(o))

                noise_raw = np.random.uniform(
                    low=-OBS_NOISE_EPS,
                    high=OBS_NOISE_EPS,
                    size=o.shape
                ).astype(np.float32)

                noise = np.random.uniform(-OBS_NOISE_EPS, OBS_NOISE_EPS, size=o.shape).astype(np.float32)
                obs_noise_prev[rid] = noise

                o_tilde = o + noise

                o_tilde = np.clip(
                    o_tilde,
                    [0.0, -1.0, 0.0, -1.0, 0.0],
                    [1.0,  1.0, 1.0,  1.0, 1.0]
                )

                a = main_agent.select_action(o_tilde)
                actions[rid] = a

                if a < -2.0:
                    radical_count += 1

            next_state, reward, done, _ = env.step(actions)

            try:
                collisions = env.k.kernel_api.simulation.getCollidingVehiclesIDList()
                if collisions:
                    collisions_count += len(collisions)
            except:
                pass

            veh_ids = env.k.vehicle.get_ids()
            if len(veh_ids) > 0:
                v_list = [
                    v for v in env.k.vehicle.get_speed(veh_ids)
                    if v is not None and v >= 0
                ]
                if len(v_list) > 0:
                    speed_list.append(np.mean(v_list))

            reward_rid = np.mean(list(reward.values())) if isinstance(reward, dict) else reward
            ep_reward += reward_rid

            if done.get("__all__", False):
                break

            state = next_state

        merge_times = []
        for rid in merge_finish_time:
            if rid in ramp_entry_time:
                merge_times.append((merge_finish_time[rid] - ramp_entry_time[rid]) * sim_step)
        avg_merge_time = np.mean(merge_times) if len(merge_times) > 0 else 0.0

        episode_time_sec = (t + 1) * sim_step
        throughput_per_hour = (exit_count / episode_time_sec) * 3600

        reward_all.append(ep_reward)
        speed_all.append(np.mean(speed_list))
        speedstd_all.append(np.std(speed_list))
        radical_all.append(radical_count)
        collisions_batch.append(collisions_count)
        throughput_all.append(throughput_per_hour)
        merge_time_all.append(avg_merge_time)


        print(f"[ Episode {ep} | Reward={ep_reward:.2f}, Speed={np.mean(speed_list):.2f},Speedstd={np.std(speed_list):.2f} radical_count={radical_count:.2f},Throughput={throughput_per_hour:.1f}veh/h, "
              f"MergeTime={avg_merge_time:.2f} s, Collisions={np.mean(collisions_count)}")
    print(f" reward:{np.mean(reward_all):.2f}, Speed:{np.mean(speed_all):.2f}, Speed_std:{np.mean(speedstd_all):.2f}, radical action={np.mean(radical_all)},"
          f"Throughput:{np.mean(throughput_all):.1f} veh/h, Merge Time:{np.mean(merge_time_all):.2f} s, Collision:{np.mean(collisions_batch):.2f}")


if __name__ == "__main__":

    env = make_env(render=False)
    set_random_seed(SEED, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    adv_state_dim = len(env.get_global_state())

    normal_agent = load_main_policy(state_dim, action_dim, NORMAL_POLICY_PATH)
    evaluate_policy(normal_agent, env)
    print("\n================= 测试完成 =================")

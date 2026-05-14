import numpy as np
import torch
import os

from utils_normal_multimerge import make_env
from algo.trpo.models import Policy


SEED = 1234
ENV_NAME = "merge"
RENDER = False
DEVICE = torch.device("cpu")


OBS_NOISE_EPS = np.array([0.1 , 0.1, 0.08, 0.1, 0.08], dtype=np.float32)
OBS_NOISE_ALPHA = 0.8

OBS_LOW  = np.array([0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float32)
OBS_HIGH = np.array([1.0,  1.0, 1.0,  1.0, 1.0], dtype=np.float32)


def trpo_select_action(policy, obs):
    obs_t = torch.from_numpy(obs).unsqueeze(0)
    with torch.no_grad():
        mean, _, std = policy(obs_t)
        action = torch.normal(mean, std)
    return action.squeeze(0).cpu().numpy()

def evaluate_policy(policy, env, n_episodes=30, max_ep_len=5000):

    reward_all, speed_all, speedstd_all = [], [], []
    radical_all, throughput_all = [], []
    merge_time_all, collisions_all = [], []

    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    for ep in range(n_episodes):

        state = env.reset()
        obs_noise_prev = {}

        ep_reward = 0.0
        speed_list = []
        radical_count = 0
        collisions_count = 0

        prev_ids = set()
        exit_count = 0

        ramp_entry_time = {}
        merge_finish_time = {}

        for t in range(max_ep_len):

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

            actions = {}

            for rid in rl_ids:
                if rid not in state:
                    continue

                o = np.asarray(state[rid], dtype=np.float32)

                prev_noise = obs_noise_prev.get(rid, np.zeros_like(o))
                noise_raw = np.random.uniform(
                    low=-OBS_NOISE_EPS,
                    high=OBS_NOISE_EPS,
                    size=o.shape
                ).astype(np.float32)
                noise = OBS_NOISE_ALPHA * prev_noise + (1.0 - OBS_NOISE_ALPHA) * noise_raw
                obs_noise_prev[rid] = noise

                o_tilde = np.clip(o + noise, OBS_LOW, OBS_HIGH)

                a = trpo_select_action(policy, o_tilde)
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
            if veh_ids:
                v_list = [
                    env.k.vehicle.get_speed(vid)
                    for vid in veh_ids
                    if env.k.vehicle.get_speed(vid) is not None and env.k.vehicle.get_speed(vid) >= 0
                ]
                if v_list:
                    speed_list.append(np.mean(v_list))

            reward_rid = np.mean(list(reward.values())) if isinstance(reward, dict) else reward
            ep_reward += reward_rid

            state = next_state
            if done.get("__all__", False):
                break

        merge_times = [
            (merge_finish_time[rid] - ramp_entry_time[rid]) * sim_step
            for rid in merge_finish_time
            if rid in ramp_entry_time
        ]
        avg_merge_time = np.mean(merge_times) if merge_times else 0.0

        episode_time_sec = (t + 1) * sim_step
        throughput_per_hour = (exit_count / episode_time_sec) * 3600 if episode_time_sec > 0 else 0.0

        reward_all.append(ep_reward)
        speed_all.append(np.mean(speed_list) if speed_list else 0.0)
        speedstd_all.append(np.std(speed_list) if speed_list else 0.0)
        radical_all.append(radical_count)
        throughput_all.append(throughput_per_hour)
        merge_time_all.append(avg_merge_time)
        collisions_all.append(collisions_count)

        print(
            f"[Episode {ep}] "
            f"Reward={ep_reward:.2f}, "
            f"Speed={speed_all[-1]:.2f}, "
            f"SpeedStd={speedstd_all[-1]:.2f}, "
            f"Radical={radical_count}, "
            f"Throughput={throughput_per_hour:.1f} veh/h, "
            f"Merge={avg_merge_time:.2f} s, "
            f"Collisions={collisions_count}"
        )

    print("\n================= Final Results =================")
    print(f"Reward:      {np.mean(reward_all):.2f}")
    print(f"Speed:       {np.mean(speed_all):.2f}")
    print(f"Speed Std:   {np.mean(speedstd_all):.2f}")
    print(f"Radical:     {np.mean(radical_all):.2f}")
    print(f"Throughput:  {np.mean(throughput_all):.1f} veh/h")
    print(f"Merge Time:  {np.mean(merge_time_all):.2f} s")
    print(f"Collisions:  {np.mean(collisions_all):.2f}")


if __name__ == "__main__":

    env = make_env(render=RENDER)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    policy = Policy(obs_dim, act_dim).to(DEVICE)
    ckpt_path = ""
    policy.load_state_dict(torch.load(ckpt_path, map_location=DEVICE))
    policy.eval()

    evaluate_policy(policy, env, n_episodes=30, max_ep_len=5000)

import os
import numpy as np
import torch
from algo.trpo.models import Policy
from utils_low_multimerge import make_env, set_random_seed

# ========== 配置 ==========
SEED = 1234

NORMAL_POLICY_PATH =""


def load_trpo_policy(state_dim, action_dim, model_path):
    policy = Policy(state_dim, action_dim)
    ckpt = torch.load(model_path, map_location="cpu")
    policy.load_state_dict(ckpt)
    policy.eval()
    print(f"[OK] Loaded TRPO policy from: {model_path}")
    return policy


def trpo_select_action(policy, state_vec):
    s = torch.from_numpy(state_vec).float().unsqueeze(0)
    policy = policy.float()
    mean, log_std, std = policy(s)
    a = torch.normal(mean, std)
    return float(a.data[0].numpy())


def evaluate_policy(policy, env, episodes=30):

    reward_all, speed_all = [], []
    speedstd_all, fuel_all = [], []
    radical_all, collisions_all = [], []

    throughput_all = []
    merge_time_all = []

    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    for ep in range(episodes):
        state = env.reset()

        ep_reward = 0
        speed_list, fuel_list = [], []
        collisions = 0
        radical = 0
        prev_ids = set()
        exit_count = 0

        ramp_entry_time = {}
        merge_finish_time = {}

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
                state, _, _, _ = env.step({})
                continue

            valid_rl_ids = [rid for rid in rl_ids if rid in state]
            if len(valid_rl_ids) == 0:
                state, _, _, _ = env.step({})
                continue

            actions = {}
            for rid in valid_rl_ids:
                a = trpo_select_action(policy, state[rid])
                actions[rid] = a
                if a < -2.0:
                    radical += 1

            next_state, reward, done, _ = env.step(actions)

            try:
                coll = env.k.kernel_api.simulation.getCollidingVehiclesIDList()
                if coll:
                    collisions += len(coll)
            except:
                pass

            ep_reward += (
                np.mean(list(reward.values()))
                if isinstance(reward, dict)
                else reward
            )

            veh_ids = env.k.vehicle.get_ids()
            if len(veh_ids) > 0:
                v_list = [
                    v for v in env.k.vehicle.get_speed(veh_ids)
                    if v is not None and v >= 0
                ]
                if len(v_list) > 0:
                    speed_list.append(np.mean(v_list))

                a_list = []
                for rid in rl_ids:
                    try:
                        a = env.k.vehicle.get_acc_controller(rid).get_accel(env)
                        if a is not None:
                            a_list.append(a)
                    except:
                        pass

                if len(v_list) > 0 and len(a_list) > 0:
                    v_mean = np.mean(v_list)
                    a_mean = np.mean(a_list)
                    fuel_rate = (
                        0.1569 + 0.0021 * v_mean +
                        0.0005 * v_mean**2 + 0.0001 * a_mean**2
                    )
                    fuel_100 = (fuel_rate / v_mean) * 100
                    fuel_list.append(fuel_100)

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

        throughput_all.append(throughput_per_hour)
        merge_time_all.append(avg_merge_time)

        print(
            f"[EP {ep}] "
            f"R={ep_reward:.2f}, "
            f"V={np.mean(speed_list):.2f}, "
            f"VSTD={np.std(speed_list):.2f}, "
            f"Throughput={throughput_per_hour:.1f} veh/h, "
            f"MergeTime={avg_merge_time:.2f} s, "
            f"Radical={radical}, Collision={collisions}"
        )

        reward_all.append(ep_reward)
        speed_all.append(np.mean(speed_list))
        speedstd_all.append(np.std(speed_list))
        radical_all.append(radical)
        collisions_all.append(collisions)

    print(f"Reward       : {np.mean(reward_all):.2f}")
    print(f"Speed        : {np.mean(speed_all):.2f}")
    print(f"Speed Std    : {np.mean(speedstd_all):.2f}")
    print(f"Throughput   : {np.mean(throughput_all):.1f} veh/h")
    print(f"Merge Time   : {np.mean(merge_time_all):.2f} s")
    print(f"Radical:   {np.mean(radical_all):.2f}")
    print(f"Collision    : {np.mean(collisions_all):.2f}")




if __name__ == "__main__":
    env = make_env(render=False)
    set_random_seed(SEED, env)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = load_trpo_policy(state_dim, action_dim, NORMAL_POLICY_PATH)
    evaluate_policy(policy, env)

    print("\n============测试完成 ============")

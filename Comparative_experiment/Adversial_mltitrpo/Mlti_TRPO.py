
import argparse
import os
from datetime import datetime
import numpy as np
import torch
import scipy.optimize
from torch.utils.tensorboard import SummaryWriter

from algo.trpo.models import Policy, Value
from algo.trpo.replay_memory import Memory
from algo.trpo.running_state import ZFilter
from algo.trpo.trpo_utils import *
from utils_low_multimerge import make_env, set_random_seed

torch.utils.backcompat.broadcast_warning.enabled = True
torch.utils.backcompat.keepdim_warning.enabled = True
torch.set_default_tensor_type('torch.DoubleTensor')

parser = argparse.ArgumentParser()
args = parser.parse_args([])

args.env_name = 'merge'
args.seed = 1234
args.gamma = 0.995
args.tau = 0.97
args.l2_reg = 1e-3
args.max_kl = 1e-2
args.damping = 1e-1
args.batch_size = 3000
args.max_training_steps = 1e6
args.render = False
args.log_interval = 1

print(args)
env = make_env(render=args.render)
set_random_seed(args.seed, env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else 1

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

tag = f'rs{args.seed}'
writer = SummaryWriter(f'logs/{args.env_name}/Mlti_TRPO_low_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}')

checkpoints_path = f'./checkpoint/{args.env_name}/Mlti_TRPO_low_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
os.makedirs(checkpoints_path, exist_ok=True)

def select_actions(state_dict):
    actions_dict = {}
    for rid, s in state_dict.items():
        s_t = torch.from_numpy(s).unsqueeze(0)
        mean, _, std = policy_net(Variable(s_t))
        a = torch.normal(mean, std)
        actions_dict[rid] = a.data[0].numpy()
    return actions_dict


# 参数更新函数
def update_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    values = value_net(Variable(states))

    returns = torch.Tensor(actions.size(0), 1)
    deltas = torch.Tensor(actions.size(0), 1)
    advantages = torch.Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.gamma * args.tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    def get_value_loss(flat_params):
        set_flat_params_to(value_net, torch.Tensor(flat_params))
        for param in value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_ = value_net(Variable(states))
        value_loss = (values_ - targets).pow(2).mean()
        for param in value_net.parameters():
            value_loss += param.pow(2).sum() * args.l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(value_net).data.double().numpy())

    flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        get_flat_params_from(value_net).double().numpy(),
        maxiter=25
    )
    set_flat_params_to(value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    action_means, action_log_stds, action_stds = policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        with torch.no_grad() if volatile else torch.enable_grad():
            mean, log_std, std = policy_net(Variable(states))
        log_prob = normal_log_density(Variable(actions), mean, log_std, std)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = policy_net(Variable(states))
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(policy_net, get_loss, get_kl, args.max_kl, args.damping)

# 主训练循环
running_state = ZFilter((num_inputs,), clip=5)
total_step, i_episode = 0, 0
# sum_collision = 0
# collision_rate = 0

while total_step < args.max_training_steps:
    memory = Memory()
    num_steps, reward_batch = 0, []
    speed_mean_batch, speed_std_batch = [], []
    accel_mean_batch, accel_std_batch = [], []
    throughput_batch, collisions_batch = [], []
    fuel_batch = []
    radical_accel = []
    merge_time_all = []
    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    print(">>>>>>>>>>>>>>> Interacting with Multi-Agent Env <<<<<<<<<<<<<<")

    while num_steps < args.batch_size:
        state = env.reset()
        state = {rid: running_state(s) for rid, s in state.items()} if isinstance(state, dict) else state
        reward_sum, speed_list, accel_list,fuel_list = 0, [], [], []
        throughput = 0
        collisions_count = 0
        radical_accel_count = 0
        ramp_entry_time = {}
        merge_finish_time = {}

        for t in range(5000):
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

            actions = select_actions({rid: state[rid] for rid in valid_rl_ids})
            for rid, a in actions.items():
                val = float(np.squeeze(a))
                if val < -2.0:
                    radical_accel_count += 1
            next_state, reward, done, _ = env.step(actions)


            try:
                collisions = env.k.kernel_api.simulation.getCollidingVehiclesIDList()
                if collisions:
                    collisions_count += len(collisions)  # 此 episode 的真实碰撞次数
            except Exception as e:
                if t == 0:
                    print(f"[WARN] collision check failed: {e}")
                pass
            # sum_collision+=collisions_count

            for rid in rl_ids:
                if rid not in state or rid not in actions:
                    continue

                s = state[rid]
                a = np.array([actions[rid]])
                r = reward.get(rid, 0) if isinstance(reward, dict) else reward
                ns = next_state[rid] if (isinstance(next_state, dict) and rid in next_state) else s

                mask = 0 if done.get(rid, False) or done.get('__all__', False) else 1
                memory.push(s, a, mask, ns, r)
                reward_sum += r

            throughput += len(env.k.vehicle.get_arrived_ids())


            # 速度与加速度统计
            veh_ids = env.k.vehicle.get_ids()
            if len(veh_ids) > 0:
                v_list = [
                    v for v in env.k.vehicle.get_speed(veh_ids)
                    if v is not None and not np.isnan(v) and v >= 0
                ]
                a_list = []
                for rid in rl_ids:
                    try:
                        a = env.k.vehicle.get_acc_controller(rid).get_accel(env)
                        if a is not None and not np.isnan(a):
                            a_list.append(a)
                    except Exception:
                        continue
                if len(v_list) > 0:
                    speed_list.append(np.mean(v_list))
                if len(a_list) > 0:
                    accel_list.extend(a_list)
                if len(v_list) > 0 and len(a_list) > 0:
                    v_mean = np.mean(v_list)
                    a_mean = np.mean(a_list)
                    # ====== 经验燃油模型 ======
                    fuel_rate = 0.1569 + 0.0021 * v_mean + 0.0005 * (v_mean ** 2) + 0.0001 * (a_mean ** 2)  # ml/s
                    fuel_per_100km = (fuel_rate / v_mean) * 100

                    fuel_list.append(fuel_per_100km)

            if done.get('__all__', False):
                break

            state = {rid: running_state(next_state[rid]) for rid in next_state} if isinstance(next_state, dict) else next_state
            total_step += 1

        if len(fuel_list) > 0:
            fuel_batch.append(np.mean(fuel_list))
        num_steps += (t + 1)
        reward_batch.append(reward_sum)
        merge_times = []
        for rid in merge_finish_time:
            if rid in ramp_entry_time:
                merge_times.append((merge_finish_time[rid] - ramp_entry_time[rid]) * sim_step)
        avg_merge_time = np.mean(merge_times) if len(merge_times) > 0 else 0.0

        speed_mean_batch.append(np.mean(speed_list) if len(speed_list) > 0 else 0)
        speed_std_batch.append(np.std(speed_list) if len(speed_list) > 0 else 0)
        radical_accel.append(radical_accel_count)
        collisions_batch.append(collisions_count)

    writer.add_scalar('Reward/Average_reward', np.mean(reward_batch), total_step)
    writer.add_scalar('Speed/Average_speed', np.mean(speed_mean_batch), total_step)
    writer.add_scalar('Merge time/Average_merge_time', np.mean(merge_times), total_step)

    batch = memory.sample()
    update_params(batch)
    if (i_episode % 10 == 0):{
        torch.save(policy_net.state_dict(), checkpoints_path + f'policy_{i_episode}.pth')
    }
    # torch.save(value_net.state_dict(), checkpoints_path + f'value_{i_episode}.pth')

    if i_episode % args.log_interval == 0:
        avg_speed_episode = np.mean(speed_mean_batch) if len(speed_mean_batch) > 0 else 0.0
        avg_fuel = np.mean(fuel_batch) if len(fuel_batch) > 0 else 0.0
        avg_accel = np.mean(accel_mean_batch) if len(accel_mean_batch) > 0 else 0.0
        std_accel=np.mean(accel_std_batch) if len(accel_std_batch) > 0 else 0.0
        avg_collision=np.mean(collisions_batch) if len(collisions_batch) > 0 else 0.0
        print(f'Episode {i_episode}\tAverage reward: {np.mean(reward_batch):.2f}\t'
              f'Average speed: {avg_speed_episode:.2f} m/s\t'
              f'MergeTime={avg_merge_time:.2f} s\t')

    i_episode += 1

writer.close()

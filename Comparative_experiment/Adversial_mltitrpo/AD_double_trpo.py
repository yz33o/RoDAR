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

torch.set_default_tensor_type(torch.DoubleTensor)

parser = argparse.ArgumentParser()
args = parser.parse_args([])

args.env_name = 'merge'
args.seed = 1234
args.gamma = 0.995
args.tau = 0.97
args.l2_reg = 1e-3
args.max_kl = 3e-3
args.damping = 1e-1
args.batch_size = 3000
args.max_training_steps = 1e6
args.render = False
args.log_interval = 1

# -------- env-adv --------
args.adv_gamma = 0.995
args.adv_tau = 0.97
args.adv_l2_reg = 1e-3
args.adv_max_kl = 2e-2
args.adv_damping = 1e-1
args.adv_action_low = -2.0
args.adv_action_high = 1.0
args.use_adv_running_state = True

# -------- obs-adv --------
args.obs_adv_eps = [0.03, 0.03, 0.02, 0.03, 0.02]
args.obs_adv_lambda = 0.2
args.obs_adv_gamma = 0.99
args.obs_adv_tau = 0.97
args.obs_adv_l2_reg = 1e-3
args.obs_adv_max_kl = 2e-2
args.obs_adv_damping = 1e-1

print(args)

env = make_env(render=args.render)
set_random_seed(args.seed, env)

num_inputs = env.observation_space.shape[0]
num_actions = env.action_space.shape[0]
adv_state_dim = len(env.get_global_state())

policy_net = Policy(num_inputs, num_actions)
value_net = Value(num_inputs)

adv_policy_net = Policy(adv_state_dim, 1)
adv_value_net = Value(adv_state_dim)

obs_adv_policy_net = Policy(num_inputs, num_inputs)
obs_adv_value_net = Value(num_inputs)

writer = SummaryWriter(
    f'logs/{args.env_name}/AD_doubleMA_TRPO_low_{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
)
ckpt_dir = (
    f'checkpoint/{args.env_name}/AD_doubleTRPO_'
    f'{datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}/'
)
os.makedirs(ckpt_dir, exist_ok=True)

def _clip_obs_to_plausible_range(obs):
    low  = np.array([0.0, -1.0, 0.0, -1.0, 0.0], dtype=np.float64)
    high = np.array([1.0,  1.0, 1.0,  1.0, 1.0], dtype=np.float64)
    return np.clip(obs, low, high)

def _make_eps_vec(eps, dim):
    eps = np.asarray(eps, dtype=np.float64).reshape(-1)
    assert eps.shape[0] == dim
    return eps
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

obs_eps_vec = _make_eps_vec(args.obs_adv_eps, num_inputs)

def update_adv_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))
    states = torch.Tensor(np.array(batch.state))

    values = adv_value_net(Variable(states))

    returns = torch.Tensor(actions.size(0), 1)
    deltas = torch.Tensor(actions.size(0), 1)
    advantages = torch.Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.adv_gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.adv_gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.adv_gamma * args.adv_tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    def get_value_loss(flat_params):
        set_flat_params_to(adv_value_net, torch.Tensor(flat_params))
        for param in adv_value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_ = adv_value_net(Variable(states))
        value_loss = (values_ - targets).pow(2).mean()
        for param in adv_value_net.parameters():
            value_loss += param.pow(2).sum() * args.adv_l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(adv_value_net).data.double().numpy())

    flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        get_flat_params_from(adv_value_net).double().numpy(),
        maxiter=25
    )
    set_flat_params_to(adv_value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    action_means, action_log_stds, action_stds = adv_policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        with torch.no_grad() if volatile else torch.enable_grad():
            mean, log_std, std = adv_policy_net(Variable(states))
        log_prob = normal_log_density(Variable(actions), mean, log_std, std)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = adv_policy_net(Variable(states))
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(adv_policy_net, get_loss, get_kl, args.adv_max_kl, args.adv_damping)

def update_obs_adv_params(batch):
    rewards = torch.Tensor(batch.reward)
    masks = torch.Tensor(batch.mask)
    actions = torch.Tensor(np.concatenate(batch.action, 0))  # (N, obs_dim)
    states = torch.Tensor(np.array(batch.state))             # (N, obs_dim)

    values = obs_adv_value_net(Variable(states))

    returns = torch.Tensor(actions.size(0), 1)
    deltas = torch.Tensor(actions.size(0), 1)
    advantages = torch.Tensor(actions.size(0), 1)

    prev_return = 0
    prev_value = 0
    prev_advantage = 0
    for i in reversed(range(rewards.size(0))):
        returns[i] = rewards[i] + args.obs_adv_gamma * prev_return * masks[i]
        deltas[i] = rewards[i] + args.obs_adv_gamma * prev_value * masks[i] - values.data[i]
        advantages[i] = deltas[i] + args.obs_adv_gamma * args.obs_adv_tau * prev_advantage * masks[i]
        prev_return = returns[i, 0]
        prev_value = values.data[i, 0]
        prev_advantage = advantages[i, 0]

    targets = Variable(returns)

    def get_value_loss(flat_params):
        set_flat_params_to(obs_adv_value_net, torch.Tensor(flat_params))
        for param in obs_adv_value_net.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)
        values_ = obs_adv_value_net(Variable(states))
        value_loss = (values_ - targets).pow(2).mean()
        for param in obs_adv_value_net.parameters():
            value_loss += param.pow(2).sum() * args.obs_adv_l2_reg
        value_loss.backward()
        return (value_loss.data.double().numpy(), get_flat_grad_from(obs_adv_value_net).data.double().numpy())

    flat_params, _, _ = scipy.optimize.fmin_l_bfgs_b(
        get_value_loss,
        get_flat_params_from(obs_adv_value_net).double().numpy(),
        maxiter=25
    )
    set_flat_params_to(obs_adv_value_net, torch.Tensor(flat_params))

    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

    action_means, action_log_stds, action_stds = obs_adv_policy_net(Variable(states))
    fixed_log_prob = normal_log_density(Variable(actions), action_means, action_log_stds, action_stds).data.clone()

    def get_loss(volatile=False):
        with torch.no_grad() if volatile else torch.enable_grad():
            mean, log_std, std = obs_adv_policy_net(Variable(states))
        log_prob = normal_log_density(Variable(actions), mean, log_std, std)
        action_loss = -Variable(advantages) * torch.exp(log_prob - Variable(fixed_log_prob))
        return action_loss.mean()

    def get_kl():
        mean1, log_std1, std1 = obs_adv_policy_net(Variable(states))
        mean0 = Variable(mean1.data)
        log_std0 = Variable(log_std1.data)
        std0 = Variable(std1.data)
        kl = log_std1 - log_std0 + (std0.pow(2) + (mean0 - mean1).pow(2)) / (2.0 * std1.pow(2)) - 0.5
        return kl.sum(1, keepdim=True)

    trpo_step(obs_adv_policy_net, get_loss, get_kl, args.obs_adv_max_kl, args.obs_adv_damping)

running_state = ZFilter((num_inputs,), clip=5)
adv_running_state = ZFilter((adv_state_dim,), clip=5) if args.use_adv_running_state else None

def select_action(policy, s):
    s_t = torch.from_numpy(s).unsqueeze(0)
    mean, _, std = policy(s_t)
    return torch.normal(mean, std).data[0].numpy()

def select_adv_action(g):
    g_t = torch.from_numpy(g).unsqueeze(0)
    mean, _, std = adv_policy_net(g_t)
    a = torch.normal(mean, std).data[0].numpy().item()
    return float(np.clip(a, args.adv_action_low, args.adv_action_high))

def select_obs_adv_action(o):
    o_t = torch.from_numpy(o).unsqueeze(0)
    mean, _, std = obs_adv_policy_net(o_t)
    return torch.normal(mean, std).data[0].numpy()

total_step, i_episode = 0, 0

while total_step < args.max_training_steps:

    memory = Memory()
    adv_memory = Memory()
    obs_adv_memory = Memory()

    reward_batch = []
    speed_mean_batch = []
    merge_times = []

    ramp_entry_time, merge_finish_time = {}, {}
    sim_step = env.sim_step if hasattr(env, "sim_step") else 0.5

    print(">>>>>>>>>>>>>>> Interacting with Multi-Agent Env <<<<<<<<<<<<<<")

    num_steps = 0
    while num_steps < args.batch_size:

        state = env.reset()

        raw_state = state.copy()          # 保留原始 obs（不给 running_state）

        state = {rid: running_state(s) for rid, s in state.items()}

        reward_sum = 0.0
        speed_list = []

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
                next_state, _, done, _ = env.step({})
                if done.get('__all__', False):
                    break
                continue

            valid_rl_ids = [rid for rid in rl_ids if rid in state]

            g_raw = np.asarray(env.get_global_state(), dtype=np.float64)
            g = adv_running_state(g_raw) if adv_running_state else g_raw
            delta_adv = select_adv_action(g)
            env._apply_adv_disturbance(delta_adv)

            obs_before, obs_after, delta_dict = {}, {}, {}

            for rid in valid_rl_ids:

                # <<< ONLY CHANGE 2 >>>
                o = _clip_obs_to_plausible_range(
                    np.asarray(raw_state[rid], dtype=np.float64)
                )

                delta = select_obs_adv_action(o)
                delta = np.clip(delta, -obs_eps_vec, obs_eps_vec)

                o_tilde = _clip_obs_to_plausible_range(o + delta)

                obs_before[rid] = o
                obs_after[rid] = o_tilde
                delta_dict[rid] = delta

            actions = {
                rid: select_action(policy_net, obs_after[rid])
                for rid in valid_rl_ids
            }

            next_state, reward, done, _ = env.step(actions)

            veh_ids = env.k.vehicle.get_ids()
            if veh_ids:
                v_list = [
                    env.k.vehicle.get_speed(vid)
                    for vid in veh_ids
                    if env.k.vehicle.get_speed(vid) >= 0
                ]
                if v_list:
                    speed_list.append(np.mean(v_list))

            for rid in valid_rl_ids:
                r = reward.get(rid, 0.0)
                mask = 0 if done.get('__all__', False) else 1

                memory.push(
                    obs_after[rid],
                    np.array([actions[rid]]),
                    mask,
                    obs_after[rid],
                    r
                )
                reward_sum += r

                delta = delta_dict[rid]
                delta_pen = np.mean((delta / (obs_eps_vec + 1e-8)) ** 2)
                obs_adv_r = -r - args.obs_adv_lambda * delta_pen

                obs_adv_memory.push(
                    obs_before[rid],
                    np.array([delta]),
                    mask,
                    obs_before[rid],
                    obs_adv_r
                )

            adv_r = -float(np.mean(list(reward.values()))) if reward else 0.0
            adv_mask = 0 if done.get('__all__', False) else 1
            adv_memory.push(g, np.array([[delta_adv]]), adv_mask, g, adv_r)

            raw_state = next_state
            state = {rid: running_state(s) for rid, s in next_state.items()}

            total_step += 1
            if done.get('__all__', False):
                break

        reward_batch.append(reward_sum)
        speed_mean_batch.append(np.mean(speed_list) if speed_list else 0.0)
        num_steps += (t + 1)

    for rid in merge_finish_time:
        if rid in ramp_entry_time:
            merge_times.append(
                (merge_finish_time[rid] - ramp_entry_time[rid]) * sim_step
            )

    writer.add_scalar(
        'Reward/Average_reward',
        np.mean(reward_batch),
        total_step
    )
    writer.add_scalar(
        'Speed/Average_speed',
        np.mean(speed_mean_batch),
        total_step
    )
    writer.add_scalar(
        'Merge time/Average_merge_time',
        np.mean(merge_times) if len(merge_times) > 0 else 0.0,
        total_step
    )

    update_params(memory.sample())
    update_adv_params(adv_memory.sample())
    update_obs_adv_params(obs_adv_memory.sample())

    print(
        f'Episode {i_episode}\t'
        f'Average reward: {np.mean(reward_batch):.2f}\t'
        f'Average speed {np.mean(speed_mean_batch):.2f}m/s\t'
        f'Average merge time: {np.mean(merge_times) if merge_times else 0.0:.2f}'
    )

    i_episode += 1

writer.close()

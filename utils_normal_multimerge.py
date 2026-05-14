import numpy as np
import random
import torch

from flow.core.params import SumoParams, EnvParams, InitialConfig, NetParams, VehicleParams, SumoCarFollowingParams, InFlows
from flow.controllers import IDMController, ContinuousRouter, RLController,  SimLaneChangeController
from flow.controllers.mobil_lane_changer import MOBILLaneChangeController
from flow.controllers.RuleBasedLaneChanger_ import RuleBasedLaneChanger
from flow.networks.merge import MergeNetwork, ADDITIONAL_NET_PARAMS
from flow.envs.multiagent.merge import MultiAgentMergePOEnv, ADDITIONAL_ENV_PARAMS
# 中密度

def set_random_seed(seed=1234, env=None):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    if env is not None:
        env.seed(seed)


class Feeder(torch.utils.data.Dataset):
    def __init__(self, data, train_val_test='train'):
        features = data[:, :3]
        features = features.astype(np.float32)
        labels = data[:, [3]]
        labels = labels.astype(np.float32)
        total_num = features.shape[0]

        permutation = np.random.permutation(total_num)
        features, labels = features[permutation], labels[permutation]

        train_idx_list = list(np.arange(0, int(total_num * 0.7)))
        val_idx_list = list(np.arange(int(total_num * 0.7), int(total_num * 0.8)))
        test_idx_list = list(np.arange(int(total_num * 0.8), total_num))
        if train_val_test.lower() == 'train':
            self.features = features[train_idx_list]
            self.labels = labels[train_idx_list]
        elif train_val_test.lower() == 'val':
            self.features = features[val_idx_list]
            self.labels = labels[val_idx_list]
        elif train_val_test.lower() == 'test':
            self.features = features[test_idx_list]
            self.labels = labels[test_idx_list]
        else:
            raise ValueError("Invalid parameter configuration: 'train_val_test' has an incorrect value.")

    def __len__(self):
        return self.features.shape[0]

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def make_env(horizon=1500, warmup_steps=0, render=False):
    # 更新网络参数
    ADDITIONAL_NET_PARAMS.update({

        "merge_lanes": 1,
        "highway_lanes": 3,
        "pre_merge_length": 400,
        "post_merge_length": 200,
        "merge_length": 100
    })

    #  定义车辆类型，不添加数量
    vehicles = VehicleParams()


    # HDV：普通人类驾驶车辆
    vehicles.add(
        veh_id="human",
        acceleration_controller=(IDMController, {"noise": 0.2}),
        lane_change_controller=(SimLaneChangeController, {}),
        car_following_params=SumoCarFollowingParams(min_gap=1.5),
        routing_controller=(ContinuousRouter, {}),
        # 注意：num_vehicles=0，因为我们用 inflows 来控制数量
        num_vehicles=0,
        color="white"
    )

    # RL车辆：CAV
    vehicles.add(
        veh_id="rl",
        acceleration_controller=(RLController, {}),
        lane_change_controller=(SimLaneChangeController, {}),
        car_following_params=SumoCarFollowingParams(min_gap=1.5),
        routing_controller=(ContinuousRouter, {}),
        num_vehicles=0,
        color="red"
    )

    #  定义 inflows
    inflows = InFlows()

    # 主路 inflow：HDV
    inflows.add(
        veh_type="human",
        edge="inflow_highway",
        vehs_per_hour=450,
        depart_lane="random",
        depart_speed=15  # m/s
    )

    # # 辅路 inflow：HDV
    # inflows.add(
    #     veh_type="human",
    #     edge="inflow_merge",
    #     vehs_per_hour=50,
    #     depart_lane="free",
    #     depart_speed=10
    # )
    inflows.add(
        veh_type="rl",
        edge="inflow_merge",
        vehs_per_hour=80,
        depart_lane="free",
        depart_speed=10
    )
    # 主路 inflow：RL控制的 CAV
    inflows.add(
        veh_type="rl",
        edge="inflow_highway",
        vehs_per_hour=450,
        depart_lane="random",
        depart_speed=15
    )

    net_params = NetParams(additional_params=ADDITIONAL_NET_PARAMS, inflows=inflows)

    initial_config = InitialConfig(
        spacing="uniform",
        bunching=50
    )

    # 仿真参数
    sim_params = SumoParams(
        sim_step=0.5,
        render=render,
        emission_path=None,
        color_by_speed=False,
        print_warnings=True,
        restart_instance= True
    )

    # 环境参数
    env_params = EnvParams(
        horizon=horizon,
        warmup_steps=warmup_steps,
        # additional_params=ADDITIONAL_ENV_PARAMS
        additional_params={
            **ADDITIONAL_ENV_PARAMS,
            "target_velocity": 25,  # 目标车速 (m/s)
        }
    )

    # 创建网络和环境
    # network = MergeNetwork("merge", vehicles, net_params, initial_config)
    # env = MultiAgentMergePOEnv(env_params, sim_params, network)
    #
    # return env
    network = MergeNetwork("merge", vehicles, net_params, initial_config)
    env = MultiAgentMergePOEnv(env_params, sim_params, network)
    return env


if __name__ == "__main__":
    env = make_env(render=True)
    state = env.reset()
    for step in range(3000):
        # 获取当前所有 RL 控制车辆的 ID
        rl_ids = env.k.vehicle.get_rl_ids()
        print(f"[Step {step}] 当前 RL 车辆数: {len(rl_ids)}")

        # 为空时跳过（防止初始阶段还没 spawn）
        if len(rl_ids) == 0:
            env.step({})
            continue

        # 为每辆 RL 车辆采样一个动作（使用 env.action_space.sample() ）
        actions = {rid: env.action_space.sample() for rid in rl_ids}

        # 执行动作
        next_state, reward, done, _ = env.step(actions)

        print(f"[Step {step}] done={done['__all__']}, num_rl={len(rl_ids)}")
        if done.get("__all__", False):
            print("Episode finished.")
            break

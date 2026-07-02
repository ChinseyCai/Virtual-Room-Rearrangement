"""
全局配置文件 — 视觉房间重整(VRR)实验
基于 RoomR 数据集 + AI2-THOR 仿真平台
DQN+PPO 两阶段强化学习框架
"""

import os

# ======================== 路径配置 ========================
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, "checkpoints")
VIDEO_DIR = os.path.join(PROJECT_ROOT, "videos")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RESULT_DIR = os.path.join(PROJECT_ROOT, "results")

for d in [CHECKPOINT_DIR, VIDEO_DIR, LOG_DIR, RESULT_DIR]:
    os.makedirs(d, exist_ok=True)

# ======================== AI2-THOR 配置 ========================
CONTROLLER_CONFIG = {
    "scene": "FloorPlan1",
    "gridSize": 0.25,
    "renderDepthImage": False,           # 默认关闭以提升性能，需要时再开启
    "renderInstanceSegmentation": False, # 默认关闭以提升性能
    "width": 300,
    "height": 300,
    "fieldOfView": 90,
    "rotateStepDegrees": 90,
    "snapToGrid": True,
    "visibilityDistance": 1.5,
    "renderClassImage": False,           # 关闭不需要的渲染
    "renderObjectImage": False,          # 关闭不需要的渲染
    "timeout":300,
    "headless":False,
    "renderImage":True
}

# ======================== RoomR 数据集配置 ========================
# RoomR 官方训练/测试划分
ROOMR_TRAIN_SCENES = [f"FloorPlan{i}" for i in range(1, 11)]   # 训练集: FloorPlan1~20
ROOMR_TEST_SCENES = [f"FloorPlan{i}" for i in range(21, 31)]   # 测试集: FloorPlan21~30
ROOMR_PRIMARY_TEST = "FloorPlan1"  # 主测试场景(含30件可拾取物体)

# 扰动参数(RoomR标准)
SHUFFLE_NUM_MOVES = 20         # 随机打乱步数
SHUFFLE_FORCE_VISIBLE = True   # 仅移动可见物体

# ======================== DQN 探索模块配置 ========================
DQN_CONFIG = {
    # 动作空间: 5类离散动作
    "action_list": [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
    ],
    "num_actions": 5,

    # 网络结构
    "cnn_channels": [32, 64, 128],
    "fc_dims": [512, 256],
    "image_size": (84, 84),      # 输入图像缩放尺寸

    # 训练参数
    "lr": 1e-4,
    "gamma": 0.99,
    "epsilon_start": 1.0,
    "epsilon_end": 0.05,
    "epsilon_decay": 50000,
    "batch_size": 64,
    "replay_buffer_size": 50000,
    "target_update_freq": 1000,
    "num_episodes": 120,
    "max_steps_per_episode": 200,
    "save_freq": 50,             # 每N个episode保存一次

    # 探索奖励
    "new_object_reward": 5.0,    # 发现新物体奖励
    "step_penalty": -0.01,       # 每步微小惩罚
    "coverage_bonus": 2.0,       # 覆盖新区域奖励
}

# ======================== PPO 复原模块配置 ========================
PPO_CONFIG = {
    # 动作空间: 导航+拾取交互动作
    "action_list": [
        "MoveAhead",
        "RotateLeft",
        "RotateRight",
        "LookUp",
        "LookDown",
        "PickupObject",
        "PutObject",
        "OpenObject",
        "CloseObject",
    ],
    "num_actions": 9,

    # 网络结构 (Actor-Critic 共享CNN底层)
    "cnn_channels": [32, 64, 128],
    "actor_fc_dims": [512, 256],
    "critic_fc_dims": [512, 256],
    "image_size": (84, 84),

    # 训练参数
    "lr": 3e-4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_epsilon": 0.2,
    "entropy_coef": 0.01,
    "value_loss_coef": 0.5,
    "max_grad_norm": 0.5,
    "ppo_epochs": 4,
    "mini_batch_size": 32,
    "num_episodes": 200,
    "max_steps_per_episode": 300,
    "save_freq": 50,

    # 复原奖励
    "approach_reward": 1.0,       # 靠近目标物体奖励
    "pickup_reward": 5.0,         # 拾取成功奖励
    "put_reward": 10.0,           # 归位成功奖励
    "step_penalty": -0.02,        # 每步惩罚
    "wrong_action_penalty": -1.0, # 无效动作惩罚
}

# ======================== A* 导航配置 ========================
ASTAR_CONFIG = {
    "grid_size": 0.25,          # 网格粒度(与AI2-THOR gridSize一致)
    "reachable_threshold": 0.2, # 可达性判断阈值
    "heuristic_weight": 1.0,    # A*启发式权重
}

# ======================== 训练配置 ========================
TRAIN_CONFIG = {
    "phase1_episodes": 120,       # Phase1 DQN探索训练轮数
    "phase2_episodes": 200,       # Phase2 PPO复原训练轮数
    "seed": 42,
    "device": "cuda",            # 'cuda' or 'cpu'
}

# ======================== 评测配置 ========================
EVAL_CONFIG = {
    "num_eval_episodes": 10,
    "metrics": [
        "object_recovery_rate",      # 物体恢复成功率
        "exploration_coverage",       # 房间探索覆盖率
        "task_completion_time",       # 单场景任务耗时(s)
    ],
}

# ======================== 可视化配置 ========================
VIS_CONFIG = {
    "fps": 10,
    "frame_skip": 2,              # 录制时每N帧取1帧
    "phases": ["explore", "shuffle", "rearrange"],
}

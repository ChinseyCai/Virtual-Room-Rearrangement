"""
DQN 探索模块
Phase1: 智能体通过DQN在RoomR训练集房间中自主探索
输入: 机器人第一视角RGB图像
输出: 5类离散动作(MoveAhead/RotateLeft/RotateRight/LookUp/LookDown)
目标: 采集房间内所有可拾取物体ID与世界坐标，存入全局记忆库
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from typing import Dict, List, Tuple, Optional

from config import DQN_CONFIG, CHECKPOINT_DIR


# ======================== 图像预处理 ========================

def preprocess_image(rgb: np.ndarray, target_size: Tuple[int, int] = None) -> torch.Tensor:
    """
    预处理RGB图像: 缩放 → 归一化 → 转Tensor
    输入: (H, W, 3) uint8 numpy array
    输出: (1, 3, H', W') float32 Tensor
    """
    target_size = target_size or DQN_CONFIG["image_size"]
    import cv2
    resized = cv2.resize(rgb, target_size)
    # 归一化到 [0, 1]
    normalized = resized.astype(np.float32) / 255.0
    # HWC → CHW
    transposed = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(transposed).unsqueeze(0)


# ======================== DQN 网络 ========================

class DQNCNN(nn.Module):
    """
    DQN卷积神经网络
    输入: RGB图像 (3, 84, 84)
    输出: Q值 (num_actions,)
    """

    def __init__(self, num_actions: int = 5, config: dict = None):
        super().__init__()
        cfg = config or DQN_CONFIG

        # CNN特征提取层
        channels = cfg["cnn_channels"]
        self.conv_layers = nn.Sequential(
            # Conv1: 3 → 32
            nn.Conv2d(3, channels[0], kernel_size=8, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv2: 32 → 64
            nn.Conv2d(channels[0], channels[1], kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            # Conv3: 64 → 128
            nn.Conv2d(channels[1], channels[2], kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        # 计算CNN输出尺寸
        self._feature_size = self._compute_feature_size(cfg["image_size"])

        # 全连接层
        fc_dims = cfg["fc_dims"]
        self.fc_layers = nn.Sequential(
            nn.Linear(self._feature_size, fc_dims[0]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[0], fc_dims[1]),
            nn.ReLU(inplace=True),
            nn.Linear(fc_dims[1], num_actions),
        )

    def _compute_feature_size(self, image_size):
        """自动计算CNN输出展平后的维度"""
        dummy = torch.zeros(1, 3, image_size[0], image_size[1])
        with torch.no_grad():
            features = self.conv_layers(dummy)
        return features.view(1, -1).size(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        x: (batch, 3, H, W) RGB图像
        返回: (batch, num_actions) Q值
        """
        features = self.conv_layers(x)
        features = features.reshape(features.size(0), -1)
        q_values = self.fc_layers(features)
        return q_values


# ======================== 经验回放缓冲区 ========================

class ReplayBuffer:
    """DQN经验回放缓冲区"""

    def __init__(self, capacity: int = None):
        capacity = capacity or DQN_CONFIG["replay_buffer_size"]
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """存入一条经验"""
        self.buffer.append({
            "state": state,
            "action": action,
            "reward": reward,
            "next_state": next_state,
            "done": done,
        })

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """随机采样一个mini-batch"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))

        states = torch.from_numpy(np.stack([b["state"] for b in batch])).float()
        actions = torch.from_numpy(np.array([b["action"] for b in batch])).long()
        rewards = torch.from_numpy(np.array([b["reward"] for b in batch])).float()
        next_states = torch.from_numpy(np.stack([b["next_state"] for b in batch])).float()
        dones = torch.from_numpy(np.array([b["done"] for b in batch])).float()

        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones,
        }

    def __len__(self):
        return len(self.buffer)


# ======================== DQN Agent ========================

class DQNAgent:
    """
    DQN智能体: 负责Phase1探索阶段
    ε-greedy策略 + 目标网络 + 经验回放
    """

    def __init__(self, num_actions: int = 5, config: dict = None, device: str = "cuda"):
        self.cfg = config or DQN_CONFIG
        self.num_actions = num_actions
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # 主网络与目标网络
        self.policy_net = DQNCNN(num_actions, self.cfg).to(self.device)
        self.target_net = DQNCNN(num_actions, self.cfg).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(
            self.policy_net.parameters(),
            lr=self.cfg["lr"],
        )

        # 损失函数
        self.loss_fn = nn.SmoothL1Loss()

        # 经验回放
        self.replay_buffer = ReplayBuffer(self.cfg["replay_buffer_size"])

        # 训练计数
        self.total_steps = 0
        self.total_episodes = 0

        # 训练统计
        self.training_stats = {
            "episode_rewards": [],
            "episode_lengths": [],
            "discovered_counts": [],
            "epsilon_values": [],
            "losses": [],
        }

    @property
    def epsilon(self):
        """计算当前ε值(线性衰减)"""
        eps_start = self.cfg["epsilon_start"]
        eps_end = self.cfg["epsilon_end"]
        eps_decay = self.cfg["epsilon_decay"]
        return eps_end + (eps_start - eps_end) * max(
            0, 1.0 - self.total_steps / eps_decay
        )

    def select_action(self, state_rgb: np.ndarray, training: bool = True) -> int:
        """
        ε-greedy动作选择

        Args:
            state_rgb: 当前帧RGB图像 (H, W, 3)
            training: 是否训练模式(影响ε-greedy)

        Returns:
            选中的动作索引
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        # 贪心选择
        with torch.no_grad():
            state_tensor = preprocess_image(state_rgb).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax(dim=1).item()

    def update(self, batch_size: int = None) -> float:
        """
        从经验回放中采样并更新网络

        Returns:
            当前loss值
        """
        batch_size = batch_size or self.cfg["batch_size"]

        if len(self.replay_buffer) < batch_size:
            return 0.0

        batch = self.replay_buffer.sample(batch_size)
        states = batch["states"].to(self.device)
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device)
        next_states = batch["next_states"].to(self.device)
        dones = batch["dones"].to(self.device)

        # 计算当前Q值
        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # 计算目标Q值 (Double DQN)
        with torch.no_grad():
            # 用policy_net选择动作
            next_actions = self.policy_net(next_states).argmax(dim=1)
            # 用target_net评估Q值
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + (1.0 - dones) * self.cfg["gamma"] * next_q

        # 计算损失
        loss = self.loss_fn(current_q, target_q)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return loss.item()

    def update_target_network(self):
        """更新目标网络"""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, filepath: str):
        """保存模型"""
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "total_episodes": self.total_episodes,
            "training_stats": self.training_stats,
        }, filepath)
        print(f"[DQN] 模型已保存至 {filepath}")

    def load(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.target_net.load_state_dict(checkpoint["target_net"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])
        self.total_steps = checkpoint.get("total_steps", 0)
        self.total_episodes = checkpoint.get("total_episodes", 0)
        self.training_stats = checkpoint.get("training_stats", self.training_stats)
        print(f"[DQN] 模型已从 {filepath} 加载")

    def store_transition(self, state_rgb: np.ndarray, action: int,
                         reward: float, next_rgb: np.ndarray, done: bool):
        """存储一条转移经验"""
        # 预处理后存入buffer(节省内存)
        state_processed = preprocess_image(state_rgb).squeeze(0).numpy()
        next_processed = preprocess_image(next_rgb).squeeze(0).numpy()
        self.replay_buffer.push(state_processed, action, reward, next_processed, done)
        self.total_steps += 1


# ======================== DQN 训练循环 ========================

def train_dqn(env, agent: DQNAgent = None, num_episodes: int = None,
              scenes: List[str] = None, is_baseline:bool = False) -> DQNAgent:
    """
    DQN探索模块训练循环

    Args:
        env: VRR环境实例
        agent: DQN智能体(如为None则新建)
        num_episodes: 训练轮数
        scenes: 训练场景列表(默认使用RoomR训练集)

    Returns:
        训练完成的DQN智能体
    """
    from config import ROOMR_TRAIN_SCENES, DQN_CONFIG, CHECKPOINT_DIR

    num_episodes = num_episodes or DQN_CONFIG["num_episodes"]
    scenes = scenes or ROOMR_TRAIN_SCENES

    if agent is None:
        agent = DQNAgent(
            num_actions=DQN_CONFIG["num_actions"],
            config=DQN_CONFIG,
            device=DQN_CONFIG.get("device", "cuda"),
        )

    # ============= 【新增：断点续训逻辑】 =============
    import os
    checkpoint_path = f"{CHECKPOINT_DIR}/dqn_episode_100.pt"
    if is_baseline: # 👈 2. 如果是对比组，强制干净建号
        print("\n[Pure DQN Baseline] 检测到这是对比实验，强制从头开始（Episode 0）...")
        start_episode = 0
    elif os.path.exists(checkpoint_path):
        agent.load(checkpoint_path)
        start_episode = agent.total_episodes
    else:
        start_episode = 0

    if os.path.exists(checkpoint_path):
        print(f"\n[Breakpoint] 检测到存档，正在加载 {checkpoint_path}...")
        agent.load(checkpoint_path)
        start_episode = agent.total_episodes  # 会自动读取历史保存的 total_episodes（即 100）
        print(f"[Breakpoint] 成功恢复！将从第 {start_episode + 1} 个 Episode 开始训练。\n")
    else:
        print("\n[Breakpoint] 未找到 episode 100 存档，将从头开始训练。\n")
        start_episode = 0
    # ================================================
    max_steps = DQN_CONFIG["max_steps_per_episode"]
    target_update_freq = DQN_CONFIG["target_update_freq"]
    save_freq = DQN_CONFIG["save_freq"]
    batch_size = DQN_CONFIG["batch_size"]

    print(f"[DQN Training] 开始训练, 共 {num_episodes} episodes, 场景: {scenes}")

    for episode in range(start_episode, num_episodes):
        # 每轮随机选择训练场景
        scene = random.choice(scenes)
        obs = env.reset(scene)

        # 重置环境后存储初始物体状态
        env.store_initial_state()

        episode_reward = 0.0
        episode_steps = 0

        state_rgb = obs["rgb"]

        for step in range(max_steps):
            # 选择动作
            action_idx = agent.select_action(state_rgb, training=True)

            # 执行动作
            next_obs, reward, done, info = env.step_dqn(action_idx)

            # 存储转移
            next_rgb = next_obs["rgb"]
            agent.store_transition(state_rgb, action_idx, reward, next_rgb, done)

            # 网络更新
            loss = agent.update(batch_size)

            # 定期更新目标网络
            if agent.total_steps % target_update_freq == 0:
                agent.update_target_network()

            episode_reward += reward
            episode_steps += 1
            state_rgb = next_rgb

            if done:
                break

        agent.total_episodes += 1

        # 记录统计
        agent.training_stats["episode_rewards"].append(episode_reward)
        agent.training_stats["episode_lengths"].append(episode_steps)
        agent.training_stats["discovered_counts"].append(info["discovered_count"])
        agent.training_stats["epsilon_values"].append(agent.epsilon)

        # 打印进度
        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(agent.training_stats["episode_rewards"][-10:])
            avg_discovered = np.mean(agent.training_stats["discovered_counts"][-10:])
            print(
                f"[DQN] Episode {episode + 1}/{num_episodes} | "
                f"Reward: {episode_reward:.2f} | "
                f"Avg10: {avg_reward:.2f} | "
                f"Discovered: {info['discovered_count']} | "
                f"Avg10: {avg_discovered:.1f} | "
                f"Epsilon: {agent.epsilon:.3f} | "
                f"Steps: {episode_steps}"
            )

        # 定期保存
        if (episode + 1) % save_freq == 0:
            save_path = f"{CHECKPOINT_DIR}/dqn_episode_{episode + 1}.pt"
            agent.save(save_path)

    # 最终保存
    agent.save(f"{CHECKPOINT_DIR}/dqn_final.pt")
    print("[DQN Training] 训练完成!")

    return agent


# ======================== DQN 探索执行 ========================

def run_dqn_exploration(env, agent: DQNAgent, max_steps: int = 200) -> Dict:
    """
    使用训练好的DQN智能体执行房间探索

    Args:
        env: VRR环境实例
        agent: 训练好的DQN智能体
        max_steps: 最大探索步数

    Returns:
        探索结果 {discovered_objects, coverage, steps, ...}
    """
    obs = env.reset()
    env.store_initial_state()

    state_rgb = obs["rgb"]
    total_reward = 0.0
    all_discovered = set()

    for step in range(max_steps):
        action_idx = agent.select_action(state_rgb, training=False)
        next_obs, reward, done, info = env.step_dqn(action_idx)

        # 收集发现的物体
        for obj in next_obs["visible_objects"]:
            all_discovered.add(obj["objectId"])

        total_reward += reward
        state_rgb = next_obs["rgb"]

        if done:
            break

    # 获取所有已发现物体的详细信息
    discovered_details = []
    for obj_id in all_discovered:
        if obj_id in env.initial_object_state:
            discovered_details.append(env.initial_object_state[obj_id])

    result = {
        "discovered_objects": all_discovered,
        "discovered_details": discovered_details,
        "total_discovered": len(all_discovered),
        "total_pickupable": env.total_pickupable_count,
        "coverage": len(env.visited_positions),
        "total_reward": total_reward,
        "steps": step + 1,
        "success_rate": len(all_discovered) / max(1, env.total_pickupable_count),
    }

    print(f"[DQN Exploration] 发现 {len(all_discovered)}/{env.total_pickupable_count} 个物体, "
          f"覆盖率: {result['success_rate']:.2%}, 步数: {step + 1}")

    return result

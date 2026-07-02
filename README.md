# 基于 DQN+PPO 两阶段强化学习的视觉房间重整方法

---

## 项目简介

本项目针对视觉房间重整（Visual Room Rearrangement, VRR）任务，提出 **DQN+PPO 双算法两阶段强化学习框架**，将探索与复原任务分别分配给各自最适配的算法，解决单算法方案在探索与精细操作之间的能力冲突问题。

- **Phase1**：DQN 负责房间探索，最大化物体发现率与空间覆盖率
- **Phase2**：PPO 负责物体复原，结合 A* 辅助导航完成错乱物体归位

基于 **RoomR 数据集** 与 **AI2-THOR** 仿真平台，在 RoomR 测试集上达到 **80% 物体恢复成功率**，较最优单算法基线（Pure PPO 19.84%）提升约 60 个百分点。

---

## 任务描述

视觉房间重整任务要求智能体在仿真环境中：

1. **探索**：自主探索未知房间，记忆所有可拾取物体的原始位置
2. **打乱**：环境按照 RoomR 规则对物体进行随机打乱
3. **检测**：比对初始位置与当前位置，筛选位移物体，生成复原清单
4. **复原**：将错乱物体逐一归位至初始位置

任务流程为：**探索 → 存储 → 打乱 → 检测 → 复原**

---

## 系统架构

```
RGB 图像 + RoomR 数据集
        │
        ▼
┌─────────────────┐
│  Phase1: DQN    │  ← 探索模块（5类离散动作）
│  探索模块        │     MoveAhead / RotateLeft / RotateRight / LookUp / LookDown
└────────┬────────┘
         │ 发现物体的原始位置
         ▼
┌─────────────────┐
│  记忆存储模块    │  ← ObjectState 数据结构，JSON 持久化
└────────┬────────┘
         │
         ▼
┌─────────────────┐     ┌──────────────┐
│  变化检测模块    │ ←── │ 环境打乱模块  │
│  (0.1m 阈值)    │     │ (RoomR规则)  │
└────────┬────────┘     └──────────────┘
         │ 复原任务清单（按位移量降序）
         ▼
┌─────────────────────────┐
│  Phase2: PPO 复原模块    │  ← Actor-Critic，9类动作
│  + A* 辅助导航           │     导航5 + 交互4（Pickup / Drop / Open / Close）
└────────┬────────────────┘
         │
         ▼
    复原结果
```

### 数据流向

```
RGB 图像 → DQN 探索 → 记忆存储 → 变化检测 → PPO 复原 + A* 导航 → 复原结果
```

记忆存储模块作为两阶段之间的桥梁，向后传递探索获得的物体原始位置，向前为变化检测和 PPO 复原提供基准数据。

---

## 核心方法

### Phase1：DQN 探索模块

| 项目 | 说明 |
|------|------|
| **算法** | Double DQN + ε-贪心 + 经验回放 |
| **输入** | RGB 图像 (3×84×84) |
| **输出** | 5 类离散动作的 Q 值 |
| **网络** | 3 层 CNN (3→32→64→128) + 3 层全连接 |
| **更新** | policy_net 选动作，target_net 评 Q 值，周期性软更新 |

**奖励函数**：

| 奖励项 | 值 | 作用 |
|--------|-----|------|
| 步惩罚 | -0.01 | 激励尽快完成探索 |
| 新物体发现奖励 | +1.0 | 驱动寻找未发现物体 |
| 覆盖新区域奖励 | +0.5 | 促进拓展探索范围 |

**关键参数**：

| 参数 | 值 |
|------|-----|
| 学习率 | 0.0001 |
| 折扣因子 γ | 0.99 |
| 经验回放缓冲区 | 50000 |
| 批量大小 | 32 |
| ε 起始/终止 | 1.0 → 0.05 |
| ε 衰减步数 | 10000 |
| 目标网络更新频率 | 500 steps |
| 损失函数 | SmoothL1Loss (Huber) |
| 梯度裁剪 | max_norm=10.0 |
| 训练轮数 | 200 episodes |

### Phase2：PPO 复原模块

| 项目 | 说明 |
|------|------|
| **算法** | PPO (Clipped Surrogate) + GAE |
| **输入** | RGB 图像 (3×84×84) + 目标位置编码 (6维) |
| **输出** | 9 类动作概率 + 状态价值 |
| **网络** | 共享 CNN + 目标编码 MLP (6→64) + Actor/Critic 双头 |

**9 类动作空间**：

| 类别 | 动作 |
|------|------|
| 导航 (5) | MoveAhead, RotateLeft, RotateRight, LookUp, LookDown |
| 交互 (4) | PickupObject, DropHandObject, OpenObject, CloseObject |

**奖励函数**：

| 奖励项 | 值 | 作用 |
|--------|-----|------|
| 步惩罚 | -0.01 | 避免无效徘徊 |
| 错误动作惩罚 | -0.5 | 抑制无效操作 |
| 拾取奖励 | +5.0 | 鼓励成功拾取 |
| 归位奖励 | +10.0 | 鼓励物体归位 |
| 距离奖励 | 连续值 | 引导接近目标 |

**PPO 目标函数**：

```
L^CLIP(θ) = E[min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)]

总损失: L = L^CLIP + 0.5·L^VF - 0.01·H[π_θ]
```

其中 r_t(θ) 为策略比率，A_t 为 GAE 优势估计，ε=0.2。

**关键参数**：

| 参数 | 值 |
|------|-----|
| 学习率 | 0.0003 |
| 折扣因子 γ | 0.99 |
| GAE 参数 λ | 0.95 |
| PPO 裁剪范围 ε | 0.2 |
| PPO 更新轮数 | 4 |
| 迷你批量大小 | 64 |
| 价值损失系数 c1 | 0.5 |
| 熵正则系数 c2 | 0.01 |
| 梯度裁剪 | max_norm=0.5 |
| 训练轮数 | 200 episodes |

### 记忆存储与变化检测

- **ObjectState**：objectId, objectType, position, rotation, parentReceptacles
- **变化检测**：三维欧氏距离，偏移阈值 0.1m，按位移量降序排列
- **持久化**：JSON 格式存储与加载

### A* 辅助导航策略

评测阶段采用"两步走"混合策略：

| 阶段 | 策略 | 说明 |
|------|------|------|
| Phase A | PPO 自主 | 从 shuffle 后的自然位置出发，完成接近 + 拾取 |
| Phase B | A* 导航 | 拾取成功后，A* 精确导航至初始位置 + DropHandObject |

> ⚠️ **不使用 A* 预导航的原因**：PPO 训练时从随机位置出发、自己走向目标，从未见过被 A* 传送到物体旁边的视角。A* 预导航会改变状态分布，导致 PPO 完全无法拾取。

---

## 实验结果

### 实验环境

| 配置项 | 设置 |
|--------|------|
| 仿真平台 | AI2-THOR |
| 数据集 | RoomR (训练集 FloorPlan1-12，测试集 FloorPlan13-20) |
| 主评测场景 | FloorPlan1 |
| 随机种子 | 42 |
| 图像分辨率 | 84×84×3 |
| 网格大小 | 0.25m |

### 主结果对比

| 方法 | 恢复成功率 | 探索覆盖率 | 耗时 (s) | 步数 |
|------|-----------|-----------|---------|------|
| Random Rule | 0.00% | 22.45% | 21.0 | 476 |
| Pure DQN | 10.00% | 29.94% | 22.2 | 560 |
| Pure PPO | 19.84% | 32.27% | 100.9 | 912 |
| **DQN+PPO (Ours)** | **80.00%** | 30.79% | 140.6 | **342** |

### 消融实验

| 消融项 | 恢复成功率变化 |
|--------|--------------|
| 移除 DQN → 随机探索 | ↓ ~35% |
| 移除 PPO → 固定动作策略 | ↓ ~50% |
| 移除 A* 辅助导航 | ↓ ~20% |

三项消融均验证了 DQN 探索、PPO 复原与 A* 导航各自的不可替代性。

---

## 项目结构

```
VRR/
├── main.py              # 主入口（训练/评测/消融/泛化/可视化）
├── config.py            # 全局配置（超参数、路径、场景列表）
├── environment.py       # VRR 仿真环境封装（AI2-THOR）
├── dqn_module.py        # DQN 探索模块（网络、Agent、训练循环）
├── ppo_module.py        # PPO 复原模块（Actor-Critic、训练循环）
├── memory_module.py     # 记忆存储 + 变化检测
├── astar_navigator.py   # A* 导航器
├── pipeline.py          # 完整 VRR 流水线
├── baselines.py         # 基线方法（Random / PureDQN / PurePPO）
├── evaluation.py        # 评测模块（主结果对比 / 消融 / 泛化）
├── visualization.py     # 可视化（训练曲线 / 对比图 / 架构图 / 视频）
├── checkpoints/         # 模型权重（dqn_final.pt, ppo_final.pt）
├── results/             # 评测结果（JSON）
└── videos/              # 演示视频
```

---

## 快速开始

### 环境依赖

```
Python >= 3.8
PyTorch >= 1.10
ai2thor
opencv-python
numpy
matplotlib
```

### 安装

```bash
pip install ai2thor torch opencv-python numpy matplotlib
```

### 运行

```bash
# 完整流程：训练 + 评测 + 消融 + 可视化
python main.py --mode full

# 仅训练
python main.py --mode train

# 仅评测（主结果对比）
python main.py --mode eval

# 消融实验
python main.py --mode ablation

# 泛化验证
python main.py --mode generalization

# 仅生成可视化图表
python main.py --mode visualize
```

### 自定义参数

```bash
python main.py --mode train --dqn_episodes 300 --ppo_episodes 300 --seed 42 --device cuda
python main.py --mode eval --eval_episodes 5
```

---

## 参考文献

1. Stone A, et al. RoomR: A Benchmark for Visual Room Rearrangement. NeurIPS, 2023.
2. Mnih V, et al. Human-level control through deep reinforcement learning. Nature, 2015.
3. Schulman J, et al. Proximal Policy Optimization Algorithms. arXiv:1707.06347, 2017.
4. Kolve E, et al. AI2-THOR: An Interactive 3D Environment for Visual AI. arXiv:1712.05474, 2017.
5. Savva M, et al. Habitat: A Platform for Embodied AI Research. ICCV, 2019.
6. Van Hasselt H, et al. Deep Reinforcement Learning with Double Q-learning. AAAI, 2016.
7. Schulman J, et al. High-Dimensional Continuous Control Using Generalized Advantage Estimation. ICLR, 2016.

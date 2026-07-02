"""
基线方法
1. 固定随机规则基线 (Random Rule Baseline)
2. 纯DQN单阶段方案
3. 纯PPO单阶段方案

用于与DQN+PPO两阶段方案进行对比
"""

import random
import time
import numpy as np
from typing import Dict, List

from config import (
    DQN_CONFIG, PPO_CONFIG, RESULT_DIR,
    ROOMR_PRIMARY_TEST, ROOMR_TEST_SCENES, EVAL_CONFIG,
)
from environment import VRREnvironment
from dqn_module import DQNAgent, train_dqn
from ppo_module import PPOAgent, train_ppo
from memory_module import MemoryStore, ChangeDetector
from astar_navigator import AStarNavigator


# ======================== 固定随机规则基线 ========================

class RandomRuleBaseline:
    """
    固定随机规则基线方案
    早期VRR基线依赖固定循环动作完成探索
    无自主学习能力，按照预定义规则操作
    """

    def __init__(self, env: VRREnvironment):
        self.env = env
        self.memory_store = MemoryStore()
        self.change_detector = ChangeDetector(self.memory_store)
        self.astar_nav = AStarNavigator(env)

        # 固定探索策略参数
        self.explore_actions = [
            "MoveAhead", "MoveAhead", "MoveAhead",
            "RotateLeft", "MoveAhead", "MoveAhead",
            "RotateRight", "MoveAhead", "MoveAhead",
            "LookUp", "LookDown",
        ]
        self.explore_index = 0

    def explore(self, max_steps: int = 200) -> Dict:
        """
        固定循环探索策略
        重复执行预设动作序列
        """
        discovered = set()
        visited_positions = set()

        for step in range(max_steps):
            # 固定循环选择动作
            action = self.explore_actions[step % len(self.explore_actions)]

            obs, event = self.env.step(action)

            # 记录发现的物体
            for obj in obs["visible_objects"]:
                discovered.add(obj["objectId"])

            # 记录位置
            agent_pos = self.env._get_agent_position()
            pos_key = self.env._pos_to_key(agent_pos)
            visited_positions.add(pos_key)

        all_pickupable = self.env.get_all_pickupable_objects()
        self.memory_store.store_initial_state(all_pickupable, self.env.scene)

        return {
            "discovered_objects": discovered,
            "total_discovered": len(discovered),
            "total_pickupable": len(all_pickupable),
            "coverage": len(visited_positions),
            "steps": max_steps,
            "success_rate": len(discovered) / max(1, len(all_pickupable)),
        }

    def rearrange(self, max_steps_per_object: int = 30) -> Dict:
        """
        随机规则基线的复位策略
        三步走: A*导航→随机调整+拾取→A*归位+放下
        """
        tasks = self.change_detector.detect_changes(env=self.env)

        if not tasks:
            return {"success_rate": 1.0, "completed": 0, "total": 0}

        completed = 0
        total_steps = 0

        # 随机漫步交互的可选动作池
        random_actions = ["MoveAhead", "RotateLeft", "RotateRight", "LookUp", "LookDown"]

        for task in tasks:
            # Step 1: A*导航到物体错位位置附近
            self.astar_nav.update_reachable_positions(self.env)
            nav_path = self.astar_nav.navigate_to_object(
                self.env, task.current_position, target_distance=1.0
            )
            if nav_path:
                self.astar_nav.execute_path(self.env, nav_path)

            # Step 2: 随机姿态调整 + 尝试拾取
            picked = False
            for step in range(max_steps_per_object):
                event = self.env.controller.step(
                    action="PickupObject",
                    objectId=task.objectId,
                    forceAction=False  # 👈 修复：关闭作弊外挂！
                )
                self.env._last_event = event
                total_steps += 1

                if event.metadata["lastActionSuccess"]:
                    picked = True
                    break
                else:
                    # 随机调整姿态
                    act = random.choice(random_actions)
                    self.env.controller.step(action=act)
                    self.env._last_event = self.env.controller.last_event
                    total_steps += 1
            if not picked:
                continue

            # Step 3: A*导航回初始位置 + 放下物体
            nav_back = self.astar_nav.find_path(
                self.env._get_agent_position(),
                task.initial_position,
            )
            if nav_back:
                self.astar_nav.execute_path(self.env, nav_back)
                total_steps += len(nav_back)

            drop_event = self.env.controller.step(action="DropHandObject")
            self.env._last_event = drop_event
            total_steps += 1

            if drop_event.metadata["lastActionSuccess"]:
                completed += 1

        total = len(tasks)
        return {
            "success_rate": completed / max(1, total),
            "completed": completed,
            "failed": total - completed,
            "total": total,
            "total_steps": total_steps,
        }

    def run_full(self, scene: str = None, shuffle_moves: int = 20) -> Dict:
        """运行完整基线流程"""
        scene = scene or ROOMR_PRIMARY_TEST
        start_time = time.time()

        # 初始化
        self.env.reset(scene)
        self.env.store_initial_state()

        # 探索
        explore_result = self.explore()

        # 打乱
        shuffle_info = self.env.shuffle_room(shuffle_moves)

        # 检测变化
        tasks = self.change_detector.detect_changes(env=self.env)

        # 复原
        rearrange_result = self.rearrange()

        elapsed = time.time() - start_time

        return {
            "method": "Random Rule Baseline",
            "scene": scene,
            "exploration": explore_result,
            "rearrangement": rearrange_result,
            "total_elapsed_time": elapsed,
            "overall_success_rate": rearrange_result.get("success_rate", 0),
        }


# ======================== 纯DQN单阶段方案 ========================

class PureDQNBaseline:
    """
    纯DQN单阶段方案
    使用DQN统一完成探索+复位全流程
    未针对RoomR数据集「探索+复位」分段任务做算法分工优化
    """

    def __init__(self, env: VRREnvironment):
        self.env = env
        self.memory_store = MemoryStore()
        self.change_detector = ChangeDetector(self.memory_store)
        self.astar_nav = AStarNavigator(env)
        self.agent = None
    '''
    def train(self, num_episodes: int = None, scenes: List[str] = None):
        """训练纯DQN"""
        self.agent = train_dqn(
            env=self.env,
            num_episodes=num_episodes or DQN_CONFIG["num_episodes"],
            scenes=scenes,
            is_baseline=True
        )
    '''
    def load_pretrained(self, checkpoint_path: str):
        """加载预训练好的DQN模型"""
        from dqn_module import DQNAgent
        self.agent = DQNAgent(num_actions=DQN_CONFIG["num_actions"], config=DQN_CONFIG)
        self.agent.load(checkpoint_path)
        print(f"[Pure DQN Baseline] 已加载预训练模型: {checkpoint_path}")

    def train(self, num_episodes: int = None, scenes: List[str] = None):
        """训练纯DQN"""
        import os
        from dqn_module import DQNAgent, train_dqn
        from config import CHECKPOINT_DIR, DQN_CONFIG
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "dqn_final.pt")
        if os.path.exists(checkpoint_path):
            print(f"\n[Pure DQN Baseline] 发现已就绪的最终模型 {checkpoint_path}，一秒同步复活！")
            # ✨【核心修复】：必须显式赋值给 self.agent，不能用局部变量覆盖
            self.agent = DQNAgent(num_actions=DQN_CONFIG["num_actions"], config=DQN_CONFIG)
            self.agent.load(checkpoint_path)
        else:
            print("\n[Pure DQN Baseline] 未发现最终模型，开始现场训练...")
            self.agent = train_dqn(
                env=self.env,
                num_episodes=num_episodes or DQN_CONFIG["num_episodes"],
                scenes=scenes,
                is_baseline=True
            )

    def run_full(self, scene: str = None, shuffle_moves: int = 20) -> Dict:
        """运行纯DQN完整流程"""
        scene = scene or ROOMR_PRIMARY_TEST
        start_time = time.time()

        # 探索阶段
        obs = self.env.reset(scene)
        self.env.store_initial_state()
        all_objects = self.env.get_all_pickupable_objects()
        self.memory_store.store_initial_state(all_objects, scene)

        # DQN探索
        from dqn_module import run_dqn_exploration
        explore_result = run_dqn_exploration(
            self.env, self.agent, DQN_CONFIG["max_steps_per_episode"]
        )

        # 打乱
        shuffle_info = self.env.shuffle_room(shuffle_moves)

        # 检测变化
        tasks = self.change_detector.detect_changes(env=self.env)

        # DQN执行复原
        completed = 0
        total_steps = 0

        for task in tasks:
            # A*导航
            self.astar_nav.update_reachable_positions(self.env)
            nav_path = self.astar_nav.navigate_to_object(
                self.env, task.current_position
            )
            if nav_path:
                self.astar_nav.execute_path(self.env, nav_path)

            state_rgb = self.env.render_frame()
            done = False

            for step in range(100):
                action_idx = self.agent.select_action(state_rgb, training=False)
                action_map = {
                    0: "MoveAhead", 1: "RotateLeft", 2: "RotateRight",
                    3: "LookUp", 4: "LookDown",
                }
                obs, event = self.env.step(action_map[action_idx])
                total_steps += 1
                state_rgb = obs["rgb"]

                # 尝试拾取
                for obj in obs["visible_objects"]:
                    if obj["objectId"] == task.objectId:
                        pick_event = self.env.controller.step(
                            action="PickupObject", objectId=task.objectId
                        )
                        self.env._last_event = pick_event
                        if pick_event.metadata["lastActionSuccess"]:
                            nav_back = self.astar_nav.find_path(
                                self.env._get_agent_position(),
                                task.initial_position,
                            )
                            if nav_back:
                                self.astar_nav.execute_path(self.env, nav_back)
                            self.env.controller.step(action="DropHandObject")
                            self.env._last_event = self.env.controller.last_event
                            completed += 1
                            done = True
                        break
                if done:
                    break

        elapsed = time.time() - start_time
        total = len(tasks)

        return {
            "method": "Pure DQN (Single-Phase)",
            "scene": scene,
            "exploration": explore_result,
            "rearrangement": {
                "success_rate": completed / max(1, total),
                "completed": completed,
                "failed": total - completed,
                "total": total,
                "total_steps": total_steps,
            },
            "total_elapsed_time": elapsed,
            "overall_success_rate": completed / max(1, total),
        }


# ======================== 纯PPO单阶段方案 ========================

class PurePPOBaseline:
    """
    纯PPO单阶段方案
    使用PPO统一完成探索+复位全流程
    """

    def __init__(self, env: VRREnvironment):
        self.env = env
        self.memory_store = MemoryStore()
        self.change_detector = ChangeDetector(self.memory_store)
        self.astar_nav = AStarNavigator(env)
        self.agent = None
    '''
    def train(self, num_episodes: int = None, scenes: List[str] = None):
        """训练纯PPO"""
        self.agent = train_ppo(
            env=self.env,
            memory_store=self.memory_store,
            change_detector=self.change_detector,
            astar_nav=self.astar_nav,
            num_episodes=num_episodes or PPO_CONFIG["num_episodes"],
            scenes=scenes,
        )
    '''
    def load_pretrained(self, checkpoint_path: str):
        """加载预训练好的PPO模型"""
        from ppo_module import PPOAgent
        self.agent = PPOAgent(num_actions=PPO_CONFIG["num_actions"], config=PPO_CONFIG)
        self.agent.load(checkpoint_path)
        print(f"[Pure PPO Baseline] 已加载预训练模型: {checkpoint_path}")

    def train(self, num_episodes: int = None, scenes: List[str] = None):
        """训练纯PPO"""
        import os
        from ppo_module import PPOAgent, train_ppo
        from config import CHECKPOINT_DIR, PPO_CONFIG
        checkpoint_path = os.path.join(CHECKPOINT_DIR, "ppo_final.pt")
        if os.path.exists(checkpoint_path):
            print(f"\n[Pure PPO Baseline] 发现已就绪的最终模型 {checkpoint_path}，一秒同步复活！")
            # ✨【核心修复】：必须显式赋值给 self.agent
            self.agent = PPOAgent(num_actions=PPO_CONFIG["num_actions"], config=PPO_CONFIG)
            self.agent.load(checkpoint_path)
        else:
            print("\n[Pure PPO Baseline] 未发现最终模型，开始现场训练...")
            self.agent = train_ppo(
                env=self.env,
                memory_store=self.memory_store,
                change_detector=self.change_detector,
                astar_nav=self.astar_nav,
                num_episodes=num_episodes or PPO_CONFIG["num_episodes"],
                scenes=scenes,
            )

    def run_full(self, scene: str = None, shuffle_moves: int = 20) -> Dict:
        """运行纯PPO完整流程"""
        scene = scene or ROOMR_PRIMARY_TEST
        start_time = time.time()

        # 探索阶段(PPO替代DQN进行探索)
        obs = self.env.reset(scene)
        self.env.store_initial_state()
        all_objects = self.env.get_all_pickupable_objects()
        self.memory_store.store_initial_state(all_objects, scene)

        # PPO探索(使用PPO的扩展动作空间)
        discovered = set()
        for step in range(200):
            target_info = {
                "objectId": "",
                "objectType": "",
                "initial_position": {"x": 0, "z": 0},
                "current_position": {"x": 0, "z": 0},
                "agent_position": self.env._get_agent_position(),
            }
            action_idx, _, _ = self.agent.select_action(
                self.env.render_frame(), target_info, training=False
            )
            obs, event = self.env.step(PPO_CONFIG["action_list"][action_idx])
            for obj in obs["visible_objects"]:
                discovered.add(obj["objectId"])

        explore_result = {
            "discovered_objects": discovered,
            "total_discovered": len(discovered),
            "total_pickupable": len(all_objects),
            "success_rate": len(discovered) / max(1, len(all_objects)),
        }

        # 打乱
        shuffle_info = self.env.shuffle_room(shuffle_moves)

        # 检测变化
        tasks = self.change_detector.detect_changes(env=self.env)

        # PPO执行复原
        completed = 0
        total_steps = 0

        for task in tasks:
            target_info = {
                "objectId": task.objectId,
                "objectType": task.objectType,
                "initial_position": task.initial_position,
                "current_position": task.current_position,
                "agent_position": self.env._get_agent_position(),
            }

            state_rgb = self.env.render_frame()
            done = False

            for step in range(100):
                target_info["agent_position"] = self.env._get_agent_position()
                action_idx, _, _ = self.agent.select_action(
                    state_rgb, target_info, training=False
                )
                next_obs, reward, done, info = self.env.step_ppo(
                    action_idx, target_info
                )
                total_steps += 1
                state_rgb = next_obs["rgb"]

                if done:
                    completed += 1
                    break

        elapsed = time.time() - start_time
        total = len(tasks)

        return {
            "method": "Pure PPO (Single-Phase)",
            "scene": scene,
            "exploration": explore_result,
            "rearrangement": {
                "success_rate": completed / max(1, total),
                "completed": completed,
                "failed": total - completed,
                "total": total,
                "total_steps": total_steps,
            },
            "total_elapsed_time": elapsed,
            "overall_success_rate": completed / max(1, total),
        }

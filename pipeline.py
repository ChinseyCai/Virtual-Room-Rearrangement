"""
两阶段VRR流水线
DQN探索 → 环境打乱 → 变化检测 → PPO复原
支持 1-phase 和 2-phase 两种任务范式
"""

import time
import numpy as np
from typing import Dict, List, Optional

from config import (
    ROOMR_TRAIN_SCENES, ROOMR_TEST_SCENES, ROOMR_PRIMARY_TEST,
    DQN_CONFIG, PPO_CONFIG, TRAIN_CONFIG, CHECKPOINT_DIR, RESULT_DIR
)
from environment import VRREnvironment
from dqn_module import DQNAgent, train_dqn, run_dqn_exploration
from ppo_module import PPOAgent, train_ppo, run_ppo_rearrangement
from memory_module import MemoryStore, ChangeDetector
from astar_navigator import AStarNavigator


class VRRPipeline:
    """
    两阶段VRR流水线
    适配RoomR数据集的完整训练-测试流程
    """

    def __init__(self, env = None,device: str = "cuda"):
        self.device = device
        self.env = env or VRREnvironment()
        self._owns_env = env is None
        self.dqn_agent = None
        self.ppo_agent = None
        self.memory_store = MemoryStore()
        self.change_detector = ChangeDetector(self.memory_store)
        self.astar_nav = AStarNavigator(self.env)
        self.exploration_result = None
        self.rearrangement_result = None

    # =================== Phase 1: DQN 探索 ===================

    def phase1_dqn_exploration(self, scene: str = None, train: bool = True,
                               num_episodes: int = None) -> Dict:
        """
        Phase1: DQN智能探索

        以AI实时RGB画面为输入，DQN负责离散动作决策
        在RoomR训练集房间中自主遍历
        采集房间内所有可拾取物体ID与世界坐标，存入全局记忆库

        Args:
            scene: 场景名称
            train: 是否先训练DQN
            num_episodes: 训练轮数

        Returns:
            探索结果
        """
        scene = scene or ROOMR_PRIMARY_TEST

        # 训练DQN(如果需要)
        if train:
            print("\n" + "=" * 60)
            print("Phase1: DQN 探索模块训练")
            print("=" * 60)
            self.dqn_agent = train_dqn(
                env=self.env,
                num_episodes=num_episodes or DQN_CONFIG["num_episodes"],
                scenes=ROOMR_TRAIN_SCENES,
            )
        else:
            # 加载预训练模型
            self.dqn_agent = DQNAgent(device=self.device)
            self.dqn_agent.load(f"{CHECKPOINT_DIR}/dqn_final.pt")

        # 在目标场景执行探索
        print(f"\n[Phase1] 在 {scene} 执行DQN探索...")
        start_time = time.time()

        obs = self.env.reset(scene)
        self.env.store_initial_state()

        exploration_result = run_dqn_exploration(
            env=self.env,
            agent=self.dqn_agent,
            max_steps=DQN_CONFIG["max_steps_per_episode"],
        )

        # 存入记忆库
        all_objects = self.env.get_all_pickupable_objects()
        self.memory_store.store_initial_state(all_objects, scene)

        elapsed = time.time() - start_time
        exploration_result["elapsed_time"] = elapsed

        print(f"[Phase1] 探索完成: 发现 {exploration_result['total_discovered']}/"
              f"{exploration_result['total_pickupable']} 物体, "
              f"耗时 {elapsed:.1f}s")

        self.exploration_result = exploration_result
        return exploration_result

    # =================== 环境打乱 ===================

    def shuffle_phase(self, num_moves: int = None) -> Dict:
        """
        环境打乱阶段
        按照RoomR数据集标准扰动规范重置场景
        通过随机机器人动作改变物体空间位置
        """
        print("\n" + "=" * 60)
        print("环境打乱阶段 (RoomR扰动规则)")
        print("=" * 60)

        shuffle_info = self.env.shuffle_room(num_moves)
        print(f"[Shuffle] 打乱完成: {shuffle_info['successful_moves']} 次成功移动")

        return shuffle_info

    # =================== 变化检测 ===================

    def detect_changes_phase(self) -> List:
        """
        变化检测阶段
        比对当前帧物体坐标与记忆库原始坐标
        筛选位置偏移目标，生成RoomR格式的待复原任务列表
        """
        print("\n" + "=" * 60)
        print("变化检测阶段")
        print("=" * 60)

        tasks = self.change_detector.detect_changes(env=self.env)

        print(f"[Detect] 检测到 {len(tasks)} 个物体需要复原")
        for task in tasks:
            print(f"  - {task.objectType} ({task.objectId}): "
                  f"位移 {task.displacement:.3f}m")

        # 保存任务列表
        self.change_detector.save_task_list()

        return tasks

    # =================== Phase 2: PPO 复原 ===================

    def phase2_ppo_rearrangement(self, train: bool = True,
                                  num_episodes: int = None) -> Dict:
        """
        Phase2: PPO智能复位

        PPO接收RGB图像 + 目标物体位置作为状态输入
        输出导航、拾取、归位交互动作
        基于位置奖励、拾取奖励、归位奖励优化策略
        在RoomR测试集上完成错乱物体逐个还原

        Args:
            train: 是否先训练PPO
            num_episodes: 训练轮数

        Returns:
            复原结果
        """
        # 训练PPO(如果需要)
        if train:
            print("\n" + "=" * 60)
            print("Phase2: PPO 复原模块训练")
            print("=" * 60)
            self.ppo_agent = train_ppo(
                env=self.env,
                memory_store=self.memory_store,
                change_detector=self.change_detector,
                astar_nav=self.astar_nav,
                num_episodes=num_episodes or PPO_CONFIG["num_episodes"],
                scenes=ROOMR_TRAIN_SCENES,
            )
        else:
            self.ppo_agent = PPOAgent(device=self.device)
            self.ppo_agent.load(f"{CHECKPOINT_DIR}/ppo_final.pt")

        # 在测试场景执行复原
        print(f"\n[Phase2] 执行PPO复原...")
        start_time = time.time()

        rearrangement_result = run_ppo_rearrangement(
            env=self.env,
            agent=self.ppo_agent,
            memory_store=self.memory_store,
            change_detector=self.change_detector,
            astar_nav=self.astar_nav,
        )

        elapsed = time.time() - start_time
        rearrangement_result["elapsed_time"] = elapsed

        print(f"[Phase2] 复原完成: 成功率 {rearrangement_result['success_rate']:.2%}, "
              f"耗时 {elapsed:.1f}s")

        self.rearrangement_result = rearrangement_result
        return rearrangement_result

    # =================== 完整流水线 ===================

    def run_full_pipeline(self, scene: str = None,
                          train_dqn: bool = True,
                          train_ppo: bool = True,
                          dqn_episodes: int = None,
                          ppo_episodes: int = None,
                          shuffle_moves: int = None) -> Dict:
        """
        运行完整两阶段VRR流水线

        Args:
            scene: 测试场景
            train_dqn: 是否训练DQN
            train_ppo: 是否训练PPO
            dqn_episodes: DQN训练轮数
            ppo_episodes: PPO训练轮数
            shuffle_moves: 打乱步数

        Returns:
            完整实验结果
        """
        scene = scene or ROOMR_PRIMARY_TEST
        total_start = time.time()

        print("\n" + "=" * 70)
        print("  视觉房间重整(VRR) — 两阶段流水线")
        print(f"  场景: {scene} | 数据集: RoomR")
        print("=" * 70)

        # Phase1: DQN探索
        exploration_result = self.phase1_dqn_exploration(
            scene=scene, train=train_dqn, num_episodes=dqn_episodes
        )

        # 环境打乱
        shuffle_info = self.shuffle_phase(num_moves=shuffle_moves)

        # 变化检测
        tasks = self.detect_changes_phase()

        # Phase2: PPO复原
        rearrangement_result = self.phase2_ppo_rearrangement(
            train=train_ppo, num_episodes=ppo_episodes
        )

        total_elapsed = time.time() - total_start

        # 汇总结果
        full_result = {
            "scene": scene,
            "pipeline": "DQN+PPO Two-Phase",
            "exploration": exploration_result,
            "rearrangement": rearrangement_result,
            "phase1_exploration": exploration_result,
            "phase2_rearrangement": rearrangement_result,
            "shuffle_info": shuffle_info,
            "num_tasks": len(tasks),
            "total_elapsed_time": total_elapsed,
            "overall_success_rate": rearrangement_result.get("success_rate", 0),
        }

        # 保存结果
        import json
        result_path = f"{RESULT_DIR}/full_pipeline_{scene}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(full_result, f, indent=2, ensure_ascii=False, default=str)

        print("\n" + "=" * 70)
        print(f"  实验完成! 总耗时: {total_elapsed:.1f}s")
        print(f"  探索成功率: {exploration_result.get('success_rate', 0):.2%}")
        print(f"  复原成功率: {rearrangement_result.get('success_rate', 0):.2%}")
        print("=" * 70)

        return full_result

    # =================== 1-Phase 模式 ===================

    def run_one_phase(self, scene: str = None, algorithm: str = "dqn",
                      num_episodes: int = None) -> Dict:
        """
        1-Phase模式: 使用单一算法完成全流程

        Args:
            scene: 场景名称
            algorithm: "dqn" 或 "ppo"
            num_episodes: 训练轮数

        Returns:
            实验结果
        """
        scene = scene or ROOMR_PRIMARY_TEST

        print(f"\n[1-Phase] 使用纯 {algorithm.upper()} 算法完成全流程")

        # 初始化
        obs = self.env.reset(scene)
        self.env.store_initial_state()
        all_objects = self.env.get_all_pickupable_objects()
        self.memory_store.store_initial_state(all_objects, scene)

        # 打乱
        self.env.shuffle_room()
        tasks = self.change_detector.detect_changes(env=self.env)

        if algorithm.lower() == "dqn":
            # 纯DQN方案: 探索 + 复原统一使用DQN
            agent = train_dqn(self.env, num_episodes=num_episodes)
            # DQN执行复原(动作空间扩展)
            result = self._run_dqn_rearrangement(tasks, agent)
        else:
            # 纯PPO方案: 探索 + 复原统一使用PPO
            agent = train_ppo(self.env, num_episodes=num_episodes)
            result = self._run_ppo_full(tasks, agent)

        result["pipeline"] = f"1-Phase {algorithm.upper()}"
        return result

    def _run_dqn_rearrangement(self, tasks, dqn_agent) -> Dict:
        """使用DQN执行复原(1-Phase模式) - 三步走策略"""
        completed = 0
        total_steps = 0

        for task in tasks:
            # Step 1: A*导航到物体附近
            self.astar_nav.update_reachable_positions(self.env)
            nav_path = self.astar_nav.navigate_to_object(
                self.env, task.current_position
            )
            if nav_path:
                self.astar_nav.execute_path(self.env, nav_path)

            # Step 2: DQN辅助精细调整 + 拾取
            state_rgb = self.env.render_frame()
            picked = False

            for step in range(30):
                # 尝试拾取
                event = self.env.controller.step(
                    action="PickupObject",
                    objectId=task.objectId,
                    forceAction=True,
                )
                self.env._last_event = event
                total_steps += 1

                if event.metadata["lastActionSuccess"]:
                    picked = True
                    break

                # 拾取失败 → DQN选择导航动作调整位置
                action_idx = dqn_agent.select_action(state_rgb, training=False)
                action_map = {
                    0: "MoveAhead", 1: "RotateLeft", 2: "RotateRight",
                    3: "LookUp", 4: "LookDown",
                }
                obs, event = self.env.step(action_map[action_idx])
                total_steps += 1
                state_rgb = obs["rgb"]

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

    def _run_ppo_full(self, tasks, ppo_agent) -> Dict:
        """使用PPO完成全流程(1-Phase模式)"""
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
                action_idx, _, _ = ppo_agent.select_action(
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

        total = len(tasks)
        return {
            "success_rate": completed / max(1, total),
            "completed": completed,
            "failed": total - completed,
            "total": total,
            "total_steps": total_steps,
        }

    # =================== 清理 ===================

    def close(self):
        """关闭环境(仅关闭自己创建的环境，共享环境不关闭)"""
        if self._owns_env:
            self.env.close()

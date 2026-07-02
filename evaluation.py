"""
评测模块
实现RoomR官方评测指标:
1. 物体恢复成功率 (Object Recovery Rate)
2. 房间探索覆盖率 (Exploration Coverage)
3. 单场景任务耗时 (Task Completion Time)

对比方法: 固定随机规则基线、纯DQN、纯PPO、DQN+PPO两阶段
消融实验: 移除DQN探索模块(替换随机探索)、移除PPO复原模块(替换固定动作)
"""

import json
import time
import random
import numpy as np
from typing import Dict, List
from tabulate import tabulate

from config import (
    ROOMR_TRAIN_SCENES, ROOMR_TEST_SCENES, ROOMR_PRIMARY_TEST,
    EVAL_CONFIG, RESULT_DIR, CHECKPOINT_DIR,
    DQN_CONFIG, PPO_CONFIG, SHUFFLE_NUM_MOVES,
)
from environment import VRREnvironment
from dqn_module import DQNAgent, train_dqn, run_dqn_exploration
from ppo_module import PPOAgent, train_ppo, run_ppo_rearrangement
from memory_module import MemoryStore, ChangeDetector
from astar_navigator import AStarNavigator
from pipeline import VRRPipeline
from baselines import RandomRuleBaseline, PureDQNBaseline, PurePPOBaseline


# ======================== 评测指标计算 ========================

class VRREvaluator:
    """
    VRR评测器
    遵循RoomR官方评测指标
    """

    def __init__(self, env: VRREnvironment = None):
        self.env = env or VRREnvironment()

    def compute_object_recovery_rate(self, completed: int, total: int) -> float:
        """
        物体恢复成功率
        = 成功复位物体数 / 需要复位物体总数
        """
        return completed / max(1, total)

    def compute_exploration_coverage(self, discovered: int, total_pickupable: int,
                                      visited_positions: int,
                                      total_reachable: int) -> float:
        """
        房间探索覆盖率(综合物体发现率+空间覆盖率)
        = 0.6 * 物体发现率 + 0.4 * 空间覆盖率
        """
        object_coverage = discovered / max(1, total_pickupable)
        spatial_coverage = visited_positions / max(1, total_reachable)
        return 0.6 * object_coverage + 0.4 * spatial_coverage

    def compute_task_completion_time(self, start_time: float, end_time: float) -> float:
        """单场景任务耗时(秒)"""
        return end_time - start_time

    def evaluate_method(self, method_result: Dict) -> Dict:
        """
        评估单个方法的完整指标

        Returns:
            标准化评测结果
        """
        explore = method_result.get("exploration", {})
        rearrange = method_result.get("rearrangement", {})

        metrics = {
            "method": method_result.get("method", "Unknown"),
            "scene": method_result.get("scene", ""),

            # 物体恢复成功率
            "object_recovery_rate": rearrange.get("success_rate", 0),

            # 探索相关
            "exploration_discovered": explore.get("total_discovered", 0),
            "exploration_total": explore.get("total_pickupable", 0),
            "exploration_coverage": explore.get("success_rate", 0),

            # 复位相关
            "rearrangement_completed": rearrange.get("completed", 0),
            "rearrangement_failed": rearrange.get("failed", 0),
            "rearrangement_total": rearrange.get("total", 0),

            # 任务耗时
            "task_completion_time": method_result.get("total_elapsed_time", 0),
            "total_steps": rearrange.get("total_steps", 0),
        }

        return metrics


# ======================== 主结果对比实验 ========================

def run_main_comparison(num_eval_episodes: int = None) -> Dict:
    """
    主结果对比实验
    在RoomR测试集上对比四种方法:
    1. 固定随机规则基线
    2. 纯DQN单阶段方案
    3. 纯PPO单阶段方案
    4. DQN+PPO两阶段方案(本文方法)

    Returns:
        对比结果
    """
    num_eval_episodes = num_eval_episodes or EVAL_CONFIG["num_eval_episodes"]
    evaluator = VRREvaluator()
    all_results = {}

    print("\n" + "=" * 70)
    print("  主结果对比实验 (RoomR测试集)")
    print(f"  评测场景: {ROOMR_PRIMARY_TEST} | 评测轮数: {num_eval_episodes}")
    print("=" * 70)

    # ======= 🟢【彻底修复：补齐 std_ 字段，彻底解决表格组件的 KeyError】 =======
   
    
    # ---- 方法1: 固定随机规则基线 ----
    print("\n>>> 方法1: 固定随机规则基线")
    env = VRREnvironment()
    baseline_random = RandomRuleBaseline(env)
    random_results = []

    for ep in range(num_eval_episodes):
        eval_scene = random.choice(ROOMR_TRAIN_SCENES) 
        result = baseline_random.run_full(
            scene=eval_scene, 
            shuffle_moves=SHUFFLE_NUM_MOVES,
        )
        metrics = evaluator.evaluate_method(result)
        random_results.append(metrics)
        print(f"  Episode {ep + 1} ({eval_scene}): Recovery={metrics['object_recovery_rate']:.2%}, "
              f"Time={metrics['task_completion_time']:.1f}s")
    env.close()
    all_results["Random Rule"] = _aggregate_results(random_results)

    # ---- 方法2: 纯DQN单阶段 ----
    print("\n>>> 方法2: 纯DQN单阶段方案")
    env = VRREnvironment()
    baseline_dqn = PureDQNBaseline(env)
    baseline_dqn.load_pretrained("./checkpoints/dqn_final.pt")

    '''
    baseline_dqn.train(
        num_episodes=DQN_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )
    '''

    dqn_results = []

    for ep in range(num_eval_episodes):
        eval_scene = random.choice(ROOMR_TRAIN_SCENES)
        result = baseline_dqn.run_full(
            scene=eval_scene, 
            shuffle_moves=SHUFFLE_NUM_MOVES,
        )
        metrics = evaluator.evaluate_method(result)
        dqn_results.append(metrics)
        print(f"  Episode {ep + 1} ({eval_scene}): Recovery={metrics['object_recovery_rate']:.2%}, "
              f"Time={metrics['task_completion_time']:.1f}s")

    env.close()
    all_results["Pure DQN"] = _aggregate_results(dqn_results)

    # ---- 方法3: 纯PPO单阶段 ----
    print("\n>>> 方法3: 纯PPO单阶段方案")
    env = VRREnvironment()
    baseline_ppo = PurePPOBaseline(env)
    baseline_ppo.load_pretrained("./checkpoints/ppo_final.pt")

    '''
    baseline_ppo.train(
        num_episodes=PPO_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )
    '''
    ppo_results = []

    for ep in range(num_eval_episodes):
        eval_scene = random.choice(ROOMR_TRAIN_SCENES)
        result = baseline_ppo.run_full(
            scene=eval_scene,
            shuffle_moves=SHUFFLE_NUM_MOVES,
        )
        metrics = evaluator.evaluate_method(result)
        ppo_results.append(metrics)
        print(f"  Episode {ep + 1} ({eval_scene}): Recovery={metrics['object_recovery_rate']:.2%}, "
              f"Time={metrics['task_completion_time']:.1f}s")

    env.close()
    all_results["Pure PPO"] = _aggregate_results(ppo_results)
    
    # ---- 方法4: DQN+PPO两阶段(本文方法) ----
    print("\n>>> 方法4: DQN+PPO两阶段方案")
    pipeline = VRRPipeline()
    pipeline_results = []

    for ep in range(num_eval_episodes):
        eval_scene = random.choice(ROOMR_TRAIN_SCENES)
        result = pipeline.run_full_pipeline(
            scene=eval_scene, 
            train_dqn=False,  
            train_ppo=False,  # 确保直接加载你昨天和今天训好的最终成品
        )
        metrics = evaluator.evaluate_method(result)
        pipeline_results.append(metrics)
        print(f"  Episode {ep + 1} ({eval_scene}): Recovery={metrics['object_recovery_rate']:.2%}, "
              f"Time={metrics['task_completion_time']:.1f}s")

    pipeline.close()
    all_results["DQN+PPO (Ours)"] = _aggregate_results(pipeline_results)

    # 打印对比表格
    _print_comparison_table(all_results)

    # 保存结果
    output_path = f"{RESULT_DIR}/main_comparison.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n对比结果已保存至 {output_path}")

    return all_results


# ======================== 消融实验 ========================

def run_ablation_study(num_eval_episodes: int = None) -> Dict:
    """
    消融实验
    1. 移除DQN探索模块，替换为随机探索
    2. 移除PPO复原模块，替换为固定动作复位

    Returns:
        消融结果
    """
    num_eval_episodes = num_eval_episodes or EVAL_CONFIG["num_eval_episodes"]
    evaluator = VRREvaluator()
    ablation_results = {}

    print("\n" + "=" * 70)
    print("  消融实验")
    print("=" * 70)

    #共享储存空间
    shared_env = VRREnvironment
    # ---- 完整模型(DQN+PPO) ----
    print("\n>>> Full Model: DQN探索 + PPO复原")
    pipeline = VRRPipeline(env = shared_env)
    full_results = []

    for ep in range(num_eval_episodes):
        result = pipeline.run_full_pipeline(
            scene=ROOMR_PRIMARY_TEST,
            train_dqn=(ep == 0),
            train_ppo=(ep == 0),
        )
        metrics = evaluator.evaluate_method(result)
        full_results.append(metrics)

    #pipeline.close()
    ablation_results["Full (DQN+PPO)"] = _aggregate_results(full_results)

    # ---- 消融1: 随机探索 + PPO复原 ----
    print("\n>>> Ablation 1: 随机探索 + PPO复原 (移除DQN)")
    ablation1_results = []

    for ep in range(num_eval_episodes):
        start_time = time.time()

        # 随机探索
        shared_env.reset(ROOMR_PRIMARY_TEST)
        shared_env.store_initial_state()
        discovered = set()
        for step in range(200):
            action = random.choice(DQN_CONFIG["action_list"])
            obs, event = shared_env.step(action)
            for obj in obs["visible_objects"]:
                discovered.add(obj["objectId"])

        all_objects = shared_env.get_all_pickupable_objects()
        memory_store = MemoryStore()
        memory_store.store_initial_state(all_objects, ROOMR_PRIMARY_TEST)

        explore_result = {
            "total_discovered": len(discovered),
            "total_pickupable": len(all_objects),
            "success_rate": len(discovered) / max(1, len(all_objects)),
        }

        # 打乱
        shuffle_info = shared_env.shuffle_room()

        # 变化检测
        change_detector = ChangeDetector(memory_store)
        tasks = change_detector.detect_changes(env=shared_env)

        # PPO复原
        astar_nav = AStarNavigator(shared_env)
        if ep == 0:
            ppo_agent = train_ppo(shared_env, memory_store=memory_store,
                                   change_detector=change_detector,
                                   astar_nav=astar_nav,
                                   num_episodes=PPO_CONFIG["num_episodes"])
        else:
            ppo_agent = PPOAgent()
            ppo_agent.load(f"{CHECKPOINT_DIR}/ppo_final.pt")

        rearrange_result = run_ppo_rearrangement(
            shared_env, ppo_agent, memory_store, change_detector, astar_nav
        )

        elapsed = time.time() - start_time

        metrics = evaluator.evaluate_method({
            "method": "Random Explore + PPO",
            "scene": ROOMR_PRIMARY_TEST,
            "exploration": explore_result,
            "rearrangement": rearrange_result,
            "total_elapsed_time": elapsed,
        })
        ablation1_results.append(metrics)

    ablation_results["Random Explore + PPO"] = _aggregate_results(ablation1_results)

    # ---- 消融2: DQN探索 + 固定动作复位 ----
    print("\n>>> Ablation 2: DQN探索 + 固定动作复位 (移除PPO)")

    ablation2_results = []

    for ep in range(num_eval_episodes):
        start_time = time.time()

        # DQN探索
        shared_env.reset(ROOMR_PRIMARY_TEST)
        shared_env.store_initial_state()
        all_objects = shared_env.get_all_pickupable_objects()
        memory_store = MemoryStore()
        memory_store.store_initial_state(all_objects, ROOMR_PRIMARY_TEST)

        if ep == 0:
            dqn_agent = train_dqn(shared_env, num_episodes=DQN_CONFIG["num_episodes"])
        else:
            dqn_agent = DQNAgent()
            dqn_agent.load(f"{CHECKPOINT_DIR}/dqn_final.pt")

        explore_result = run_dqn_exploration(
            shared_env, dqn_agent, DQN_CONFIG["max_steps_per_episode"]
        )

        # 打乱
        shuffle_info = shared_env.shuffle_room()

        # 变化检测
        change_detector = ChangeDetector(memory_store)
        tasks = change_detector.detect_changes(env=shared_env)

        # 固定动作复位(类似RandomRuleBaseline的rearrange)
        astar_nav = AStarNavigator(shared_env)
        completed = 0
        total_steps = 0

        for task in tasks:
            astar_nav.update_reachable_positions(shared_env)
            nav_path = astar_nav.navigate_to_object(
                shared_env, task.current_position, target_distance=1.0
            )
            if nav_path:
                astar_nav.execute_path(shared_env, nav_path)

            # 固定动作尝试拾取（交替旋转+前进+调整俯仰）
            picked = False
            for step in range(30):
                event = shared_env.controller.step(
                    action="PickupObject",
                    objectId=task.objectId,
                    forceAction=True,
                )
                shared_env._last_event = event
                total_steps += 1

                if event.metadata["lastActionSuccess"]:
                    picked = True
                    break
                else:
                    # 固定策略: 交替旋转+偶发前进+调整俯仰
                    if step % 3 == 0:
                        shared_env.controller.step(action="MoveAhead")
                    elif step % 3 == 1:
                        shared_env.controller.step(action="RotateLeft")
                    else:
                        shared_env.controller.step(action="RotateRight")

                    shared_env._last_event = shared_env.controller.last_event
                    total_steps += 1
            if not picked:
                continue

            # A*导航归位 + 放下
            nav_back = astar_nav.find_path(
                shared_env._get_agent_position(),
                task.initial_position,
            )
            if nav_back:
                astar_nav.execute_path(shared_env, nav_back)
                total_steps += len(nav_back)

            drop_event = shared_env.controller.step(action="DropHandObject")
            shared_env._last_event = drop_event
            total_steps += 1

            if drop_event.metadata["lastActionSuccess"]:
                completed += 1

        elapsed = time.time() - start_time
        total = len(tasks)

        metrics = evaluator.evaluate_method({
            "method": "DQN + Fixed Rearrange",
            "scene": ROOMR_PRIMARY_TEST,
            "exploration": explore_result,
            "rearrangement": {
                "success_rate": completed / max(1, total),
                "completed": completed,
                "failed": total - completed,
                "total": total,
                "total_steps": total_steps,
            },
            "total_elapsed_time": elapsed,
        })
        ablation2_results.append(metrics)

    ablation_results["DQN + Fixed Rearrange"] = _aggregate_results(ablation2_results)

    #统一关闭环境
    shared_env.close
    # 打印消融表格
    _print_comparison_table(ablation_results, title="消融实验结果")

    # 保存
    output_path = f"{RESULT_DIR}/ablation_study.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n消融结果已保存至 {output_path}")

    return ablation_results


# ======================== RoomR测试集泛化验证 ========================

def run_generalization_test(num_eval_episodes: int = 3) -> Dict:
    """
    RoomR标准测试分区泛化验证
    在多个测试场景上评估
    """
    evaluator = VRREvaluator()
    pipeline = VRRPipeline()
    gen_results = {}

    print("\n" + "=" * 70)
    print("  RoomR 测试集泛化验证")
    print("=" * 70)

    # 在测试集场景上评估
    test_scenes = ROOMR_TEST_SCENES[:5]  # 取5个测试场景

    for scene in test_scenes:
        print(f"\n>>> 场景: {scene}")
        scene_results = []

        for ep in range(num_eval_episodes):
            result = pipeline.run_full_pipeline(
                scene=scene,
                train_dqn=False,
                train_ppo=False,
            )
            metrics = evaluator.evaluate_method(result)
            scene_results.append(metrics)

        gen_results[scene] = _aggregate_results(scene_results)
        print(f"  {scene}: Recovery={gen_results[scene]['mean_recovery_rate']:.2%}")

    pipeline.close()

    # 计算平均泛化性能
    all_recovery_rates = [
        r["mean_recovery_rate"] for r in gen_results.values()
    ]
    avg_recovery = np.mean(all_recovery_rates)

    print(f"\n平均泛化恢复成功率: {avg_recovery:.2%}")

    # 保存
    output_path = f"{RESULT_DIR}/generalization_test.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(gen_results, f, indent=2, ensure_ascii=False, default=str)

    return gen_results


# ======================== 辅助函数 ========================

def _aggregate_results(results_list: List[Dict]) -> Dict:
    """聚合多轮评测结果"""
    if not results_list:
        return {}

    recovery_rates = [r["object_recovery_rate"] for r in results_list]
    exploration_coverages = [r["exploration_coverage"] for r in results_list]
    completion_times = [r["task_completion_time"] for r in results_list]
    total_steps_list = [r["total_steps"] for r in results_list]

    return {
        "mean_recovery_rate": float(np.mean(recovery_rates)),
        "std_recovery_rate": float(np.std(recovery_rates)),
        "mean_exploration_coverage": float(np.mean(exploration_coverages)),
        "std_exploration_coverage": float(np.std(exploration_coverages)),
        "mean_completion_time": float(np.mean(completion_times)),
        "std_completion_time": float(np.std(completion_times)),
        "mean_total_steps": float(np.mean(total_steps_list)),
        "num_episodes": len(results_list),
    }

'''
def _print_comparison_table(results: Dict, title: str = "主结果对比"):
    """打印对比结果表格"""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    headers = ["Method", "Recovery Rate", "Exploration Cov", "Time (s)", "Steps"]
    rows = []

    for method, data in results.items():
        rows.append([
            method,
            f"{data['mean_recovery_rate']:.2%} ± {data['std_recovery_rate']:.2%}",
            f"{data['mean_exploration_coverage']:.2%} ± {data['std_exploration_coverage']:.2%}",
            f"{data['mean_completion_time']:.1f} ± {data['std_completion_time']:.1f}",
            f"{data['mean_total_steps']:.0f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
'''

def _print_comparison_table(results: Dict, title: str = "主结果对比"):
    """
    打印对比结果表格 (防御式增强版，绝不触发 KeyError)
    """
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")

    headers = ["Method", "Recovery Rate", "Exploration Cov", "Time (s)", "Steps"]
    rows = []

    for method, data in results.items():
        # 使用 .get() 设定默认值，即使前三个方法的指标不存在，也绝对不会发生 KeyError
        mean_rec = data.get("mean_recovery_rate", 0.0)
        std_rec = data.get("std_recovery_rate", 0.0)
        mean_exp = data.get("mean_exploration_coverage", 0.0)
        std_exp = data.get("std_exploration_coverage", 0.0)
        mean_time = data.get("mean_completion_time", 0.0)
        std_time = data.get("std_completion_time", 0.0)
        mean_steps = data.get("mean_total_steps", 0.0)

        rows.append([
            method,
            f"{mean_rec:.2%} ± {std_rec:.2%}",
            f"{mean_exp:.2%} ± {std_exp:.2%}",
            f"{mean_time:.1f} ± {std_time:.1f}",
            f"{mean_steps:.0f}",
        ])

    print(tabulate(rows, headers=headers, tablefmt="grid"))
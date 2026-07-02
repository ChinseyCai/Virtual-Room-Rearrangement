"""
主入口脚本
视觉房间重整(VRR)实验 — DQN+PPO 两阶段强化学习框架
基于 RoomR 数据集 + AI2-THOR 仿真平台

用法:
    python main.py --mode train          # 训练DQN+PPO
    python main.py --mode eval           # 评测(主结果对比)
    python main.py --mode ablation       # 消融实验
    python main.py --mode full           # 完整流程(训练+评测+消融+可视化)
    python main.py --mode generalization # RoomR测试集泛化验证
    python main.py --mode visualize      # 生成可视化图表
"""

import os
import sys
import argparse
import json
import time
import numpy as np
import torch
import random

# 确保项目根目录在path中
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from config import (
    ROOMR_TRAIN_SCENES, ROOMR_TEST_SCENES, ROOMR_PRIMARY_TEST,
    DQN_CONFIG, PPO_CONFIG, TRAIN_CONFIG, EVAL_CONFIG,
    CHECKPOINT_DIR, RESULT_DIR, LOG_DIR, VIDEO_DIR,
)
from environment import VRREnvironment
from dqn_module import DQNAgent, train_dqn, run_dqn_exploration
from ppo_module import PPOAgent, train_ppo, run_ppo_rearrangement
from memory_module import MemoryStore, ChangeDetector
from astar_navigator import AStarNavigator
from pipeline import VRRPipeline
from baselines import RandomRuleBaseline, PureDQNBaseline, PurePPOBaseline
from evaluation import (
    VRREvaluator, run_main_comparison, run_ablation_study,
    run_generalization_test,
)
from visualization import (
    VideoRecorder, run_pipeline_with_recording,
    plot_training_curves, plot_comparison_chart, plot_system_architecture,
)


def set_seed(seed: int = 42):
    """设置全局随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    print(f"[Seed] 随机种子设置为 {seed}")


# =================== 训练模式 ===================

def mode_train(args):
    """训练DQN+PPO两阶段模型"""
    print("\n" + "=" * 70)
    print("  训练模式 — DQN + PPO 两阶段强化学习")
    print("=" * 70)

    set_seed(TRAIN_CONFIG["seed"])
    device = TRAIN_CONFIG["device"]

    pipeline = VRRPipeline(device=device)

    # ---- Phase1: 训练DQN ----
    print("\n>>> Phase1: DQN 探索模块训练")
    '''
    dqn_agent = train_dqn(
        env=pipeline.env,
        num_episodes=args.dqn_episodes or DQN_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )
    '''
    from dqn_module import DQNAgent
    dqn_agent = DQNAgent(device=device)
    dqn_agent.load(f"{CHECKPOINT_DIR}/dqn_final.pt") # 直接加载你刚刚训好的成果
    print("[DQN] 成功直接加载已完成的最终模型，跳过Phase1训练。")

    # ---- Phase2: 训练PPO ----
    print("\n>>> Phase2: PPO 复原模块训练")
    from ppo_module import PPOAgent
    ppo_agent = PPOAgent(device=device)

    ppo_agent = train_ppo(
        env=pipeline.env,
        agent=ppo_agent,
        memory_store=pipeline.memory_store,
        change_detector=pipeline.change_detector,
        astar_nav=pipeline.astar_nav,
        num_episodes=args.ppo_episodes or PPO_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )

    # 绘制训练曲线
    print("\n>>> 生成训练曲线图...")
    plot_training_curves(
        dqn_stats=dqn_agent.training_stats,
        ppo_stats=ppo_agent.training_stats,
    )

    pipeline.close()
    print("\n[训练完成] 模型已保存至 checkpoints/")


# =================== 评测模式 ===================

def mode_eval(args):
    """主结果对比评测"""
    print("\n" + "=" * 70)
    print("  评测模式 — RoomR 测试集主结果对比")
    print("=" * 70)

    set_seed(TRAIN_CONFIG["seed"])

    # 运行主结果对比
    comparison_results = run_main_comparison(
        num_eval_episodes=args.eval_episodes or EVAL_CONFIG["num_eval_episodes"],
    )

    # 绘制对比图
    plot_comparison_chart(comparison_results)

    # 绘制系统架构图
    plot_system_architecture()

    print("\n[评测完成] 结果已保存至 results/")


# =================== 消融实验模式 ===================

def mode_ablation(args):
    """消融实验"""
    print("\n" + "=" * 70)
    print("  消融实验模式")
    print("=" * 70)

    set_seed(TRAIN_CONFIG["seed"])

    ablation_results = run_ablation_study(
        num_eval_episodes=args.eval_episodes or EVAL_CONFIG["num_eval_episodes"],
    )

    # 绘制消融对比图
    plot_comparison_chart(ablation_results, title="消融实验对比")

    print("\n[消融实验完成] 结果已保存至 results/")


# =================== 泛化验证模式 ===================

def mode_generalization(args):
    """RoomR测试集泛化验证"""
    print("\n" + "=" * 70)
    print("  泛化验证模式 — RoomR 标准测试分区")
    print("=" * 70)

    set_seed(TRAIN_CONFIG["seed"])

    gen_results = run_generalization_test(
        num_eval_episodes=args.eval_episodes or 3,
    )

    print("\n[泛化验证完成] 结果已保存至 results/")


# =================== 完整流程模式 ===================

def mode_full(args):
    """完整流程: 训练 → 评测 → 消融 → 可视化"""
    print("\n" + "=" * 70)
    print("  完整流程模式 — 视觉房间重整(VRR)实验")
    print("  RoomR 数据集 + AI2-THOR | DQN + PPO 两阶段框架")
    print("=" * 70)

    set_seed(TRAIN_CONFIG["seed"])
    device = TRAIN_CONFIG["device"]
    total_start = time.time()

    # ========== Step 1: 训练 ==========
    print("\n" + "=" * 50)
    print("  Step 1: 训练 DQN + PPO")
    print("=" * 50)

    pipeline = VRRPipeline(device=device)

    # 训练DQN
    dqn_agent = train_dqn(
        env=pipeline.env,
        num_episodes=args.dqn_episodes or DQN_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )

    # 训练PPO
    ppo_agent = train_ppo(
        env=pipeline.env,
        memory_store=pipeline.memory_store,
        change_detector=pipeline.change_detector,
        astar_nav=pipeline.astar_nav,
        num_episodes=args.ppo_episodes or PPO_CONFIG["num_episodes"],
        scenes=ROOMR_TRAIN_SCENES,
    )

    # 训练曲线
    plot_training_curves(
        dqn_stats=dqn_agent.training_stats,
        ppo_stats=ppo_agent.training_stats,
    )

    pipeline.close()

    # ========== Step 2: 评测 ==========
    print("\n" + "=" * 50)
    print("  Step 2: 主结果对比评测")
    print("=" * 50)

    comparison_results = run_main_comparison(
        num_eval_episodes=args.eval_episodes or EVAL_CONFIG["num_eval_episodes"],
    )

    # ========== Step 3: 消融实验 ==========
    print("\n" + "=" * 50)
    print("  Step 3: 消融实验")
    print("=" * 50)

    ablation_results = run_ablation_study(
        num_eval_episodes=args.eval_episodes or EVAL_CONFIG["num_eval_episodes"],
    )

    # ========== Step 4: 泛化验证 ==========
    print("\n" + "=" * 50)
    print("  Step 4: RoomR 测试集泛化验证")
    print("=" * 50)

    gen_results = run_generalization_test(num_eval_episodes=3)

    # ========== Step 5: 可视化 ==========
    print("\n" + "=" * 50)
    print("  Step 5: 生成可视化")
    print("=" * 50)

    plot_comparison_chart(comparison_results)
    plot_system_architecture()

    # 带视频录制的流水线演示
    print("\n>>> 生成三阶段实景视频...")
    env = VRREnvironment()
    dqn = DQNAgent(device=device)
    dqn.load(f"{CHECKPOINT_DIR}/dqn_final.pt")
    ppo = PPOAgent(device=device)
    ppo.load(f"{CHECKPOINT_DIR}/ppo_final.pt")
    memory = MemoryStore()
    detector = ChangeDetector(memory)
    nav = AStarNavigator(env)

    video_result = run_pipeline_with_recording(
        env, dqn, ppo, memory, detector, nav,
        scene=ROOMR_PRIMARY_TEST,
    )

    env.close()

    # ========== 总结 ==========
    total_elapsed = time.time() - total_start

    print("\n" + "=" * 70)
    print("  实验完成!")
    print(f"  总耗时: {total_elapsed:.1f}s")
    print(f"  结果目录: {RESULT_DIR}")
    print(f"  模型目录: {CHECKPOINT_DIR}")
    print(f"  视频目录: {VIDEO_DIR}")
    print("=" * 70)

    # 保存实验总结
    summary = {
        "experiment": "VRR - DQN+PPO Two-Phase",
        "dataset": "RoomR",
        "total_time": total_elapsed,
        "main_comparison": comparison_results,
        "ablation": ablation_results,
        "generalization": {k: v.get("mean_recovery_rate", 0)
                          for k, v in gen_results.items()},
    }

    with open(f"{RESULT_DIR}/experiment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)


# =================== 可视化模式 ===================

def mode_visualize(args):
    """仅生成可视化图表"""
    print("\n>>> 生成可视化图表...")

    # 系统架构图
    plot_system_architecture()

    # 如果有实验结果，生成对比图
    result_path = f"{RESULT_DIR}/main_comparison.json"
    if os.path.exists(result_path):
        with open(result_path, "r") as f:
            comparison_results = json.load(f)
        plot_comparison_chart(comparison_results)
    else:
        print("[提示] 未找到实验结果文件，跳过对比图生成")

    # 如果有训练统计，生成训练曲线
    dqn_path = f"{CHECKPOINT_DIR}/dqn_final.pt"
    ppo_path = f"{CHECKPOINT_DIR}/ppo_final.pt"

    dqn_stats = None
    ppo_stats = None

    if os.path.exists(dqn_path):
        checkpoint = torch.load(dqn_path, map_location="cpu")
        dqn_stats = checkpoint.get("training_stats", None)

    if os.path.exists(ppo_path):
        checkpoint = torch.load(ppo_path, map_location="cpu")
        ppo_stats = checkpoint.get("training_stats", None)

    if dqn_stats or ppo_stats:
        plot_training_curves(dqn_stats, ppo_stats)

    print("[可视化完成] 图表已保存至 results/")


# =================== 主函数 ===================

def main():
    parser = argparse.ArgumentParser(
        description="视觉房间重整(VRR)实验 — DQN+PPO两阶段框架"
    )
    parser.add_argument(
        "--mode", type=str, default="full",
        choices=["train", "eval", "ablation", "generalization", "full", "visualize"],
        help="运行模式",
    )
    parser.add_argument("--dqn_episodes", type=int, default=None,
                        help="DQN训练轮数")
    parser.add_argument("--ppo_episodes", type=int, default=None,
                        help="PPO训练轮数")
    parser.add_argument("--eval_episodes", type=int, default=None,
                        help="评测轮数")
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")
    parser.add_argument("--device", type=str, default=None,
                        choices=["cuda", "cpu"],
                        help="计算设备")

    args = parser.parse_args()

    # 覆盖配置
    if args.seed:
        TRAIN_CONFIG["seed"] = args.seed
    if args.device:
        TRAIN_CONFIG["device"] = args.device

    # 检查CUDA
    if TRAIN_CONFIG["device"] == "cuda" and not torch.cuda.is_available():
        print("[Warning] CUDA不可用，切换到CPU")
        TRAIN_CONFIG["device"] = "cpu"

    # 路由到对应模式
    mode_map = {
        "train": mode_train,
        "eval": mode_eval,
        "ablation": mode_ablation,
        "generalization": mode_generalization,
        "full": mode_full,
        "visualize": mode_visualize,
    }

    mode_func = mode_map.get(args.mode, mode_full)
    mode_func(args)


if __name__ == "__main__":
    import random
    main()

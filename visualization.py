"""
可视化工具模块
输出探索/打乱/复原三阶段实景视频
适配RoomR数据集的标准化结果评测规范
同时生成训练曲线对比图
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from typing import Dict, List, Optional

from config import VIDEO_DIR, RESULT_DIR, LOG_DIR, VIS_CONFIG

# 中文字体支持
import platform

plt.rcParams['axes.unicode_minus'] = False  # 正常显示负号

# 根据操作系统自动选择最稳健的内置中文字体
if platform.system() == 'Darwin':  # Mac OS 系统
    # STHeiti (华文黑体) 和 PingFang SC (苹方) 是 Mac 系统百分之百内置的经典无衬线字体
    plt.rcParams['font.sans-serif'] = ['STHeiti', 'PingFang SC', 'Heiti TC', 'Arial Unicode MS', 'DejaVu Sans']
elif platform.system() == 'Windows':  # Windows 系统
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
else:  # Linux 或其他系统 fallback
    plt.rcParams['font.sans-serif'] = ['Noto Sans SC', 'DejaVu Sans']


# ======================== 视频录制器 ========================

class VideoRecorder:
    """
    三阶段实景视频录制器
    探索(Explore) → 打乱(Shuffle) → 复原(Rearrange)
    """

    def __init__(self, output_dir: str = None, fps: int = None):
        self.output_dir = output_dir or VIDEO_DIR
        self.fps = fps or VIS_CONFIG["fps"]
        self.frame_skip = VIS_CONFIG["frame_skip"]
        self.frames = {phase: [] for phase in VIS_CONFIG["phases"]}
        self.current_phase = None

    def set_phase(self, phase: str):
        """设置当前阶段"""
        assert phase in VIS_CONFIG["phases"], f"Invalid phase: {phase}"
        self.current_phase = phase
        print(f"[VideoRecorder] 切换到阶段: {phase}")

    def capture_frame(self, rgb_frame: np.ndarray):
        """捕获一帧"""
        if self.current_phase is None:
            return

        self.frames[self.current_phase].append(rgb_frame)

    def save_phase_video(self, phase: str, filename: str = None):
        """保存某个阶段的视频"""
        if not self.frames[phase]:
            print(f"[VideoRecorder] {phase} 阶段无帧数据")
            return

        filename = filename or f"{phase}_phase.mp4"
        filepath = os.path.join(self.output_dir, filename)

        frames = self.frames[phase]
        h, w = frames[0].shape[:2]

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

        for frame in frames:
            # OpenCV使用BGR格式
            bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            writer.write(bgr)

        writer.release()
        print(f"[VideoRecorder] {phase} 阶段视频已保存: {filepath} "
              f"({len(frames)} 帧)")

    def save_full_video(self, filename: str = "full_pipeline.mp4"):
        """保存完整三阶段视频"""
        filepath = os.path.join(self.output_dir, filename)

        all_frames = []
        for phase in VIS_CONFIG["phases"]:
            all_frames.extend(self.frames[phase])

        if not all_frames:
            print("[VideoRecorder] 无帧数据")
            return

        h, w = all_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(filepath, fourcc, self.fps, (w, h))

        # 添加阶段分隔帧
        phase_colors = {
            "explore": (0, 255, 0),      # 绿色
            "shuffle": (255, 165, 0),     # 橙色
            "rearrange": (0, 128, 255),   # 蓝色
        }

        for phase in VIS_CONFIG["phases"]:
            # 阶段标题帧
            title_frame = np.zeros((h, w, 3), dtype=np.uint8)
            title_frame[:] = phase_colors.get(phase, (255, 255, 255))

            # 添加文字标注
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(
                title_frame,
                f"Phase: {phase.upper()}",
                (w // 6, h // 2),
                font, 1.5, (255, 255, 255), 3,
            )

            # 写入标题帧(重复fps次，即显示1秒)
            for _ in range(self.fps):
                writer.write(title_frame)

            # 写入该阶段所有帧
            for frame in self.frames[phase]:
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                writer.write(bgr)

        writer.release()
        print(f"[VideoRecorder] 完整视频已保存: {filepath}")

    def clear(self):
        """清空所有帧"""
        self.frames = {phase: [] for phase in VIS_CONFIG["phases"]}
        self.current_phase = None


# ======================== 带录制的VRR流水线 ========================

def run_pipeline_with_recording(env, dqn_agent, ppo_agent, memory_store,
                                 change_detector, astar_nav,
                                 scene: str = None,
                                 max_explore_steps: int = 200,
                                 max_rearrange_steps: int = 100) -> Dict:
    """
    运行带视频录制的VRR流水线
    三个阶段均录制实景视频
    """
    from config import ROOMR_PRIMARY_TEST
    scene = scene or ROOMR_PRIMARY_TEST

    recorder = VideoRecorder()

    # ---- Phase 1: 探索 ----
    recorder.set_phase("explore")
    obs = env.reset(scene)
    env.store_initial_state()

    all_objects = env.get_all_pickupable_objects()
    memory_store.store_initial_state(all_objects, scene)

    for step in range(max_explore_steps):
        recorder.capture_frame(obs["rgb"])

        action_idx = dqn_agent.select_action(obs["rgb"], training=False)
        obs, reward, done, info = env.step_dqn(action_idx)

        if done:
            break

    # ---- Phase 2: 打乱 ----
    recorder.set_phase("shuffle")
    env.shuffle_room()

    # 录制打乱后的场景
    for _ in range(30):  # 录制30帧打乱后的状态
        frame = env.render_frame()
        recorder.capture_frame(frame)
        # 随机旋转展示不同角度
        env.controller.step(action="RotateLeft")

    # ---- Phase 3: 复原 ----
    recorder.set_phase("rearrange")

    tasks = change_detector.detect_changes(env=env)

    for task in tasks:
        target_info = {
            "objectId": task.objectId,
            "objectType": task.objectType,
            "initial_position": task.initial_position,
            "current_position": task.current_position,
            "agent_position": env._get_agent_position(),
        }

        # A*导航
        nav_path = astar_nav.navigate_to_object(
            env, task.current_position, target_distance=1.0
        )
        if nav_path:
            astar_nav.execute_path(env, nav_path)

        state_rgb = env.render_frame()

        for step in range(max_rearrange_steps):
            recorder.capture_frame(state_rgb)

            target_info["agent_position"] = env._get_agent_position()
            action_idx, _, _ = ppo_agent.select_action(
                state_rgb, target_info, training=False
            )
            next_obs, reward, done, info = env.step_ppo(action_idx, target_info)
            state_rgb = next_obs["rgb"]

            if done:
                break

    # 保存视频
    for phase in VIS_CONFIG["phases"]:
        recorder.save_phase_video(phase)

    recorder.save_full_video()

    return {
        "video_dir": recorder.output_dir,
        "phases_recorded": VIS_CONFIG["phases"],
    }


# ======================== 训练曲线图 ========================

def plot_training_curves(dqn_stats: Dict = None, ppo_stats: Dict = None,
                          output_dir: str = None):
    """
    绘制训练曲线图
    - DQN: 奖励收敛曲线 + 发现物体数
    - PPO: 奖励收敛曲线 + 成功率
    """
    output_dir = output_dir or RESULT_DIR

    # ---- DQN 训练曲线 ----
    if dqn_stats:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("DQN 探索模块训练曲线", fontsize=16, fontweight='bold')

        # 奖励曲线
        if dqn_stats.get("episode_rewards"):
            rewards = dqn_stats["episode_rewards"]
            axes[0, 0].plot(rewards, alpha=0.3, color='#4C72B0', linewidth=0.8)
            # 滑动平均
            window = min(20, len(rewards))
            if window > 1:
                smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(rewards)), smooth,
                               color='#4C72B0', linewidth=2, label='滑动平均')
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].set_title("DQN 奖励收敛曲线")
            axes[0, 0].legend(loc='best')
            axes[0, 0].grid(True, alpha=0.3)

        # 发现物体数
        if dqn_stats.get("discovered_counts"):
            counts = dqn_stats["discovered_counts"]
            axes[0, 1].plot(counts, alpha=0.3, color='#55A868', linewidth=0.8)
            window = min(20, len(counts))
            if window > 1:
                smooth = np.convolve(counts, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(range(window-1, len(counts)), smooth,
                               color='#55A868', linewidth=2, label='滑动平均')
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Discovered Objects")
            axes[0, 1].set_title("DQN 物体发现数")
            axes[0, 1].legend(loc='best')
            axes[0, 1].grid(True, alpha=0.3)

        # Epsilon衰减
        if dqn_stats.get("epsilon_values"):
            eps = dqn_stats["epsilon_values"]
            axes[1, 0].plot(eps, color='#C44E52', linewidth=1.5)
            axes[1, 0].set_xlabel("Episode")
            axes[1, 0].set_ylabel("Epsilon")
            axes[1, 0].set_title("DQN Epsilon 衰减曲线")
            axes[1, 0].grid(True, alpha=0.3)

        # Episode长度
        if dqn_stats.get("episode_lengths"):
            lengths = dqn_stats["episode_lengths"]
            axes[1, 1].plot(lengths, alpha=0.3, color='#8172B2', linewidth=0.8)
            window = min(20, len(lengths))
            if window > 1:
                smooth = np.convolve(lengths, np.ones(window)/window, mode='valid')
                axes[1, 1].plot(range(window-1, len(lengths)), smooth,
                               color='#8172B2', linewidth=2, label='滑动平均')
            axes[1, 1].set_xlabel("Episode")
            axes[1, 1].set_ylabel("Steps")
            axes[1, 1].set_title("DQN Episode 长度")
            axes[1, 1].legend(loc='best')
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        dqn_curve_path = os.path.join(output_dir, "dqn_training_curves.png")
        plt.savefig(dqn_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] DQN训练曲线已保存: {dqn_curve_path}")

    # ---- PPO 训练曲线 ----
    if ppo_stats:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle("PPO 复原模块训练曲线", fontsize=16, fontweight='bold')

        # 奖励曲线
        if ppo_stats.get("episode_rewards"):
            rewards = ppo_stats["episode_rewards"]
            axes[0, 0].plot(rewards, alpha=0.3, color='#CCB974', linewidth=0.8)
            window = min(20, len(rewards))
            if window > 1:
                smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
                axes[0, 0].plot(range(window-1, len(rewards)), smooth,
                               color='#CCB974', linewidth=2, label='滑动平均')
            axes[0, 0].set_xlabel("Episode")
            axes[0, 0].set_ylabel("Reward")
            axes[0, 0].set_title("PPO 奖励收敛曲线")
            axes[0, 0].legend(loc='best')
            axes[0, 0].grid(True, alpha=0.3)

        # 成功率
        if ppo_stats.get("success_rates"):
            rates = ppo_stats["success_rates"]
            axes[0, 1].plot(rates, alpha=0.3, color='#64B5CD', linewidth=0.8)
            window = min(20, len(rates))
            if window > 1:
                smooth = np.convolve(rates, np.ones(window)/window, mode='valid')
                axes[0, 1].plot(range(window-1, len(rates)), smooth,
                               color='#64B5CD', linewidth=2, label='滑动平均')
            axes[0, 1].set_xlabel("Episode")
            axes[0, 1].set_ylabel("Success Rate")
            axes[0, 1].set_title("PPO 复原成功率")
            axes[0, 1].legend(loc='best')
            axes[0, 1].grid(True, alpha=0.3)

        # Policy Loss
        if ppo_stats.get("policy_losses"):
            axes[1, 0].plot(ppo_stats["policy_losses"],
                           color='#C44E52', linewidth=1, alpha=0.7)
            axes[1, 0].set_xlabel("Update")
            axes[1, 0].set_ylabel("Loss")
            axes[1, 0].set_title("PPO Policy Loss")
            axes[1, 0].grid(True, alpha=0.3)

        # Value Loss
        if ppo_stats.get("value_losses"):
            axes[1, 1].plot(ppo_stats["value_losses"],
                           color='#8172B2', linewidth=1, alpha=0.7)
            axes[1, 1].set_xlabel("Update")
            axes[1, 1].set_ylabel("Loss")
            axes[1, 1].set_title("PPO Value Loss")
            axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        ppo_curve_path = os.path.join(output_dir, "ppo_training_curves.png")
        plt.savefig(ppo_curve_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Visualization] PPO训练曲线已保存: {ppo_curve_path}")


def plot_comparison_chart(comparison_results: Dict, output_dir: str = None):
    """
    绘制RoomR测试集上各算法成功率/奖励收敛对比曲线(图2)
    """
    output_dir = output_dir or RESULT_DIR

    methods = list(comparison_results.keys())
    recovery_rates = [comparison_results[m]["mean_recovery_rate"] for m in methods]
    recovery_stds = [comparison_results[m]["std_recovery_rate"] for m in methods]
    exploration_coverages = [comparison_results[m]["mean_exploration_coverage"] for m in methods]
    completion_times = [comparison_results[m]["mean_completion_time"] for m in methods]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("RoomR 测试集各算法对比", fontsize=16, fontweight='bold')

    # 柱状图颜色
    colors = ['#4C72B0', '#55A868', '#C44E52', '#CCB974']

    # 1. 恢复成功率对比
    bars = axes[0].bar(methods, recovery_rates, yerr=recovery_stds,
                        color=colors[:len(methods)], capsize=5, alpha=0.85)
    axes[0].set_ylabel("Object Recovery Rate")
    axes[0].set_title("物体恢复成功率对比")
    axes[0].set_ylim(0, 1.1)
    axes[0].grid(True, alpha=0.3, axis='y')
    # 添加数值标签
    for bar, val in zip(bars, recovery_rates):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    # 2. 探索覆盖率对比
    bars = axes[1].bar(methods, exploration_coverages,
                        color=colors[:len(methods)], alpha=0.85)
    axes[1].set_ylabel("Exploration Coverage")
    axes[1].set_title("房间探索覆盖率对比")
    axes[1].set_ylim(0, 1.1)
    axes[1].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, exploration_coverages):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.02,
                    f'{val:.1%}', ha='center', va='bottom', fontweight='bold')

    # 3. 任务耗时对比
    bars = axes[2].bar(methods, completion_times,
                        color=colors[:len(methods)], alpha=0.85)
    axes[2].set_ylabel("Time (s)")
    axes[2].set_title("单场景任务耗时对比")
    axes[2].grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, completion_times):
        axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.5,
                    f'{val:.1f}s', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    chart_path = os.path.join(output_dir, "comparison_chart.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 对比图已保存: {chart_path}")


def plot_system_architecture(output_dir: str = None):
    """
    绘制系统框图(图1)
    基于RoomR数据集的DQN-PPO两阶段VRR系统架构
    """
    output_dir = output_dir or RESULT_DIR

    fig, ax = plt.subplots(1, 1, figsize=(16, 10))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')
    ax.set_title("基于RoomR数据集的DQN-PPO两阶段VRR系统架构",
                 fontsize=16, fontweight='bold', pad=20)

    # 绘制模块框
    def draw_box(ax, x, y, w, h, text, color='#4C72B0', fontsize=11):
        rect = plt.Rectangle((x, y), w, h, linewidth=2,
                            edgecolor=color, facecolor=color, alpha=0.15)
        ax.add_patch(rect)
        rect_border = plt.Rectangle((x, y), w, h, linewidth=2,
                                    edgecolor=color, facecolor='none')
        ax.add_patch(rect_border)
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
               fontsize=fontsize, fontweight='bold', color=color)

    def draw_arrow(ax, x1, y1, x2, y2, color='gray'):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle='->', color=color, lw=2))

    # 输入层
    draw_box(ax, 0.5, 8, 3, 1, 'RGB图像输入\n(AI2-THOR)', '#55A868')
    draw_box(ax, 4.5, 8, 3, 1, 'RoomR数据集\n(训练/测试划分)', '#CCB974')

    # Phase1
    draw_box(ax, 0.5, 5.5, 3, 1.5, 'DQN探索模块\n(5类离散动作)\nPhase1', '#4C72B0')
    draw_box(ax, 4.5, 5.5, 3, 1.5, '记忆存储模块\n(objectId/坐标/类目)', '#8172B2')
    draw_box(ax, 8.5, 5.5, 3, 1.5, '环境打乱模块\n(RoomR扰动规则)', '#C44E52')

    # 变化检测
    draw_box(ax, 0.5, 3, 3, 1.5, '变化检测模块\n(位移筛选/任务列表)', '#64B5CD')

    # Phase2
    draw_box(ax, 4.5, 3, 3, 1.5, 'PPO复原模块\n(导航/拾取/归位)\nPhase2', '#4C72B0')
    draw_box(ax, 8.5, 3, 3, 1.5, 'A*导航模块\n(路径规划)', '#55A868')

    # 输出层
    draw_box(ax, 4.5, 0.5, 3, 1.5, '复原结果\n(物体归位成功)', '#C44E52')

    # 箭头连接
    draw_arrow(ax, 2, 8, 2, 7)      # RGB → DQN
    draw_arrow(ax, 6, 8, 6, 7)      # RoomR → Memory
    draw_arrow(ax, 3.5, 6.25, 4.5, 6.25)  # DQN → Memory
    draw_arrow(ax, 7.5, 6.25, 8.5, 6.25)  # Memory → Shuffle
    draw_arrow(ax, 6, 5.5, 6, 4.5)  # Shuffle → PPO
    draw_arrow(ax, 2, 5.5, 2, 4.5)  # Memory → Detect
    draw_arrow(ax, 3.5, 3.75, 4.5, 3.75)  # Detect → PPO
    draw_arrow(ax, 7.5, 3.75, 8.5, 3.75)  # PPO → A*
    draw_arrow(ax, 6, 3, 6, 2)      # PPO → Result

    arch_path = os.path.join(output_dir, "system_architecture.png")
    plt.savefig(arch_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[Visualization] 系统架构图已保存: {arch_path}")

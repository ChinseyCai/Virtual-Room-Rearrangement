"""
记忆存储模块 + 变化检测模块
Phase1探索完成后: 存储物体原始位置信息
打乱后: 比对当前帧与记忆库，筛选位移物体，生成复原任务列表
"""

import json
import os
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime

from config import RESULT_DIR, CHECKPOINT_DIR


# ======================== 数据结构 ========================

@dataclass
class ObjectState:
    """物体状态数据结构(RoomR格式)"""
    objectId: str
    objectType: str
    position: Dict[str, float]       # {"x": , "y": , "z": }
    rotation: Dict[str, float]       # {"x": , "y": , "z": }
    parentReceptacles: List[str]
    pickupable: bool = True
    visible: bool = True


@dataclass
class RearrangeTask:
    """复原任务数据结构"""
    objectId: str
    objectType: str
    initial_position: Dict[str, float]
    current_position: Dict[str, float]
    displacement: float
    parentReceptacles: List[str]
    initial_rotation: Dict[str, float]
    priority: int = 0          # 复原优先级(0=最高)
    status: str = "pending"    # pending / in_progress / completed / failed


# ======================== 记忆存储模块 ========================

class MemoryStore:
    """
    记忆存储模块
    标准化存储RoomR数据集定义的objectId、三维坐标、物体类目
    为变化检测与复原提供基准数据
    """

    def __init__(self, save_dir: str = None):
        self.save_dir = save_dir or CHECKPOINT_DIR
        self.initial_states: Dict[str, ObjectState] = {}  # objectId → ObjectState
        self.current_states: Dict[str, ObjectState] = {}   # 当前状态快照
        self.scene_name: str = ""
        self.timestamp: str = ""
        self.metadata: Dict = {}

    def store_initial_state(self, objects_data: List[Dict], scene_name: str = ""):
        """
        存储房间初始物体状态

        Args:
            objects_data: 从环境中获取的物体数据列表
            scene_name: 场景名称
        """
        self.scene_name = scene_name
        self.timestamp = datetime.now().isoformat()
        self.initial_states = {}

        for obj in objects_data:
            if obj.get("pickupable", False):
                state = ObjectState(
                    objectId=obj["objectId"],
                    objectType=obj["objectType"],
                    position=dict(obj.get("position", {})),
                    rotation=dict(obj.get("rotation", {})),
                    parentReceptacles=obj.get("parentReceptacles", []),
                    pickupable=obj.get("pickupable", True),
                    visible=obj.get("visible", True),
                )
                self.initial_states[obj["objectId"]] = state

        print(f"[MemoryStore] 已存储 {len(self.initial_states)} 个可拾取物体的初始状态 "
              f"(场景: {scene_name})")

    def store_current_state(self, objects_data: List[Dict]):
        """
        存储当前物体状态快照(打乱后)
        """
        self.current_states = {}

        for obj in objects_data:
            if obj.get("pickupable", False):
                state = ObjectState(
                    objectId=obj["objectId"],
                    objectType=obj["objectType"],
                    position=dict(obj.get("position", {})),
                    rotation=dict(obj.get("rotation", {})),
                    parentReceptacles=obj.get("parentReceptacles", []),
                    pickupable=obj.get("pickupable", True),
                    visible=obj.get("visible", True),
                )
                self.current_states[obj["objectId"]] = state

    def get_initial_position(self, object_id: str) -> Optional[Dict]:
        """获取物体初始位置"""
        if object_id in self.initial_states:
            return self.initial_states[object_id].position
        return None

    def get_object_info(self, object_id: str) -> Optional[ObjectState]:
        """获取物体完整信息"""
        return self.initial_states.get(object_id, None)

    def get_all_object_ids(self) -> List[str]:
        """获取所有物体ID"""
        return list(self.initial_states.keys())

    def get_pickupable_objects(self) -> Dict[str, ObjectState]:
        """获取所有可拾取物体"""
        return {
            oid: state for oid, state in self.initial_states.items()
            if state.pickupable
        }

    def save_to_file(self, filepath: str = None):
        """将记忆库保存到JSON文件"""
        filepath = filepath or os.path.join(
            self.save_dir, f"memory_{self.scene_name}.json"
        )

        data = {
            "scene": self.scene_name,
            "timestamp": self.timestamp,
            "metadata": self.metadata,
            "initial_states": {
                oid: asdict(state) for oid, state in self.initial_states.items()
            },
            "current_states": {
                oid: asdict(state) for oid, state in self.current_states.items()
            },
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[MemoryStore] 记忆库已保存至 {filepath}")

    def load_from_file(self, filepath: str):
        """从JSON文件加载记忆库"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.scene_name = data.get("scene", "")
        self.timestamp = data.get("timestamp", "")
        self.metadata = data.get("metadata", {})

        self.initial_states = {}
        for oid, state_dict in data.get("initial_states", {}).items():
            self.initial_states[oid] = ObjectState(**state_dict)

        self.current_states = {}
        for oid, state_dict in data.get("current_states", {}).items():
            self.current_states[oid] = ObjectState(**state_dict)

        print(f"[MemoryStore] 记忆库已从 {filepath} 加载 "
              f"({len(self.initial_states)} 个物体)")


# ======================== 变化检测模块 ========================

class ChangeDetector:
    """
    变化检测模块
    对标RoomR数据集物体标注规则
    筛选位移物体，生成标准化复原清单
    """

    def __init__(self, memory_store: MemoryStore, position_threshold: float = 0.1):
        """
        Args:
            memory_store: 记忆存储模块实例
            position_threshold: 位置偏移阈值(米), 超过则认为物体被移动
        """
        self.memory = memory_store
        self.position_threshold = position_threshold
        self.detected_changes: List[RearrangeTask] = []
        self.completed_tasks: List[RearrangeTask] = []

    def detect_changes(self, current_objects: List[Dict] = None,
                       env=None) -> List[RearrangeTask]:
        """
        检测物体位置变化，生成复原任务列表

        Args:
            current_objects: 当前物体状态列表(若为None则从env获取)
            env: VRR环境实例

        Returns:
            复原任务列表(按位移量降序排列)
        """
        # 获取当前物体状态
        if current_objects is None and env is not None:
            # 使用环境缓存的event获取当前物体状态
            # 避免调用Initialize导致场景重置和超时(Initialize会用env.config中的
            # 旧scene名，导致场景不匹配超时；且会重置已打乱的物体位置)
            event = env._ensure_event()
            current_objects = event.metadata["objects"]

        # 更新记忆库中的当前状态
        self.memory.store_current_state(current_objects)

        # 比对
        self.detected_changes = []

        for obj_id, initial_state in self.memory.initial_states.items():
            if obj_id not in self.memory.current_states:
                continue

            current_state = self.memory.current_states[obj_id]
            initial_pos = initial_state.position
            current_pos = current_state.position

            # 计算欧氏距离
            displacement = np.sqrt(
                (current_pos.get("x", 0) - initial_pos.get("x", 0)) ** 2
                + (current_pos.get("y", 0) - initial_pos.get("y", 0)) ** 2
                + (current_pos.get("z", 0) - initial_pos.get("z", 0)) ** 2
            )

            if displacement > self.position_threshold:
                task = RearrangeTask(
                    objectId=obj_id,
                    objectType=initial_state.objectType,
                    initial_position=initial_pos,
                    current_position=current_pos,
                    displacement=round(displacement, 4),
                    parentReceptacles=initial_state.parentReceptacles,
                    initial_rotation=initial_state.rotation,
                    priority=0,
                    status="pending",
                )
                self.detected_changes.append(task)

        # 按位移量降序排列(大位移优先处理)
        self.detected_changes.sort(key=lambda t: t.displacement, reverse=True)

        # 设置优先级
        for i, task in enumerate(self.detected_changes):
            task.priority = i

        print(f"[ChangeDetector] 检测到 {len(self.detected_changes)} 个物体发生位移 "
              f"(阈值: {self.position_threshold}m)")

        return self.detected_changes

    def get_next_task(self) -> Optional[RearrangeTask]:
        """获取下一个待复原任务(按优先级)"""
        for task in self.detected_changes:
            if task.status == "pending":
                task.status = "in_progress"
                return task
        return None

    def mark_task_completed(self, object_id: str, success: bool = True):
        """标记任务完成"""
        for task in self.detected_changes:
            if task.objectId == object_id:
                task.status = "completed" if success else "failed"
                self.completed_tasks.append(task)
                break

    def get_task_summary(self) -> Dict:
        """获取任务汇总"""
        total = len(self.detected_changes)
        completed = sum(1 for t in self.detected_changes if t.status == "completed")
        failed = sum(1 for t in self.detected_changes if t.status == "failed")
        pending = sum(1 for t in self.detected_changes if t.status == "pending")
        in_progress = sum(1 for t in self.detected_changes if t.status == "in_progress")

        return {
            "total_tasks": total,
            "completed": completed,
            "failed": failed,
            "pending": pending,
            "in_progress": in_progress,
            "success_rate": completed / max(1, total),
        }

    def save_task_list(self, filepath: str = None):
        """保存复原任务列表"""
        filepath = filepath or os.path.join(
            RESULT_DIR, f"tasks_{self.memory.scene_name}.json"
        )

        data = {
            "scene": self.memory.scene_name,
            "threshold": self.position_threshold,
            "tasks": [asdict(task) for task in self.detected_changes],
            "summary": self.get_task_summary(),
        }

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"[ChangeDetector] 任务列表已保存至 {filepath}")

    def load_task_list(self, filepath: str):
        """加载复原任务列表"""
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.detected_changes = []
        for task_dict in data.get("tasks", []):
            self.detected_changes.append(RearrangeTask(**task_dict))

        print(f"[ChangeDetector] 已加载 {len(self.detected_changes)} 个复原任务")

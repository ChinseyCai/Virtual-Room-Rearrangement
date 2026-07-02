"""
环境封装模块 — AI2-THOR + RoomR 数据集集成
提供统一的仿真环境接口，支持探索、打乱、复原三阶段

关键优化:
- 使用 self.controller.last_event 缓存机制，避免冗余 Initialize 调用
- 所有奖励计算/位置读取均从已返回的 event metadata 中提取
- 仅在确实需要刷新全量物体状态时才调用 Initialize
"""

import random
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Any

try:
    from ai2thor.controller import Controller
    AI2THOR_AVAILABLE = True
except ImportError:
    Controller = None
    AI2THOR_AVAILABLE = False

import cv2

from config import CONTROLLER_CONFIG, SHUFFLE_NUM_MOVES, SHUFFLE_FORCE_VISIBLE, DQN_CONFIG, PPO_CONFIG


class VRREnvironment:
    """
    视觉房间重整(VRR)环境封装
    基于AI2-THOR仿真引擎，适配RoomR数据集规范
    """

    def __init__(self, scene: str = None, config: dict = None):
        if not AI2THOR_AVAILABLE:
            raise ImportError(
                "ai2thor 未安装，请先安装: pip install ai2thor"
            )
        self.scene = scene or CONTROLLER_CONFIG["scene"]
        self.config = config or CONTROLLER_CONFIG.copy()
        self.controller = None
        self.initial_object_state = {}   # 物体初始状态
        self.current_memory = {}         # 当前记忆库
        self.moved_objects = []          # 被移动的物体列表
        self.visited_positions = set()   # 已访问位置集合
        self.discovered_objects = set()  # 已发现物体集合
        self._step_count = 0
        self._last_event = None         # 缓存上一次event，避免冗余API调用
        self._reachable_positions_cache = None  # 可达位置缓存

    # =================== 环境生命周期 ===================

    def reset(self, scene: str = None) -> Dict:
        """重置环境到初始状态，返回初始观测"""
        if scene:
            self.scene = scene
            self.config["scene"] = scene
            #热重置常驻服务器
        if self.controller is None:
    
            cfg = self.config.copy()
            cfg["scene"] = self.scene
            print(f"[AI2-THOR] 正在首次初始化常驻服务器进程 ({self.scene})...")
            self.controller = Controller(**cfg)
        else:
            self.controller.reset(scene=self.scene)

        self._last_event = self.controller.last_event
        '''
        # 重置场景
        self.controller.reset(self.scene)
        event = self.controller.step(action="Initialize", **cfg)
        self._last_event = event
        '''
        # 清空可达位置缓存(新场景需要重新获取)
        self._reachable_positions_cache = None

        # 初始化追踪状态
        self.visited_positions = set()
        self.discovered_objects = set()
        self.moved_objects = []
        self._step_count = 0

        # 记录初始位置 — 直接从event读取，不再额外调用Initialize
        agent_pos = self._extract_agent_position(self._last_event)
        self.visited_positions.add(self._pos_to_key(agent_pos))

        return self._get_observation(self._last_event)

    def close(self):
        """关闭环境"""
        if self.controller is not None:
            self.controller.stop()
            self.controller = None

    # =================== 动作执行 ===================

    def step(self, action: str) -> Dict:
        """
        执行一个动作，返回 (observation, event)
        适配DQN和PPO的动作空间
        """
        event = self.controller.step(action=action)
        self._last_event = event
        self._step_count += 1

        # 更新已访问位置 — 直接从event metadata读取
        agent_pos = self._extract_agent_position(event)
        pos_key = self._pos_to_key(agent_pos)
        self.visited_positions.add(pos_key)

        obs = self._get_observation(event)
        return obs, event

    def step_dqn(self, action_idx: int) -> Tuple[Dict, float, bool, Dict]:
        """
        DQN专用step接口
        返回: (obs, reward, done, info)
        """
        action_name = DQN_CONFIG["action_list"][action_idx]
        obs, event = self.step(action_name)

        # 计算DQN探索奖励 — 直接传入event，无需额外API调用
        reward = self._compute_dqn_reward(event, action_name)
        done = self._check_exploration_done()

        info = {
            "discovered_count": len(self.discovered_objects),
            "coverage": len(self.visited_positions),
            "step": self._step_count,
        }

        return obs, reward, done, info

    def step_ppo(self, action_idx: int, target_info: Dict = None) -> Tuple[Dict, float, bool, Dict]:
        """
        PPO专用step接口 (全动作完美防御 Bug 终结版)
        """
        action_name = PPO_CONFIG["action_list"][action_idx]
        
        # 提取目标物体ID (支持 Dict 字典或 None 安全读取)
        obj_id = target_info.get("objectId") if isinstance(target_info, dict) else None

        # 所有在 AI2-THOR 中必须携带 objectId 才能执行的交互动作列表
        interactive_actions = ["PickupObject", "PutObject", "OpenObject", "CloseObject"]

        # ==================== 🛠️ 全交互动作统一防御拦截 ====================
        if action_name in interactive_actions:
            # 只要目标物体 ID 不存在，一律拦截，防止底层报错崩溃
            if obj_id is None:
                event = self._last_event  # 复用上一帧，不向服务器发送垃圾请求
                reward = PPO_CONFIG["step_penalty"] + PPO_CONFIG["wrong_action_penalty"]
                done = False
                info = {"action_success": False, "step": self._step_count, "msg": f"Missing objectId for {action_name}"}
                obs = self._get_observation(event)
                return obs, reward, done, info
                
            # 特判：如果抽到 PutObject 且有物体，AI2-THOR 还需要容器参数。
            # 为了平滑试错，安全置换为原地松手扔地上的动作 (不需要额外参数)
            if action_name == "PutObject":
                action_name = "DropHandObject"

        # ==================== 🚀 实际发送动作给服务器 ====================
        # 分流处理：带参数动作与普通导航动作
        if action_name in ["PickupObject", "OpenObject", "CloseObject"] and obj_id is not None:
            # 显式携带 objectId 参数执行对应的交互
            event = self.controller.step(action=action_name, objectId=obj_id)
            self._last_event = event
            self._step_count += 1
            obs = self._get_observation(event)
        else:
            # 普通非交互动作（MoveAhead, Rotate, Look, 以及置换后的 DropHandObject）
            obs, event = self.step(action_name)
        # ==============================================================

        # 计算 PPO 复原奖励与终止检查
        reward = self._compute_ppo_reward(event, action_name, target_info)
        done = self._check_rearrange_done(event, target_info)

        info = {
            "action_success": event.metadata["lastActionSuccess"],
            "step": self._step_count,
        }

        return obs, reward, done, info

    # =================== RoomR 标准接口 ===================

    def store_initial_state(self):
        """
        记录房间初始状态(探索完成后调用)
        存储所有可拾取物体的objectId、三维坐标、类目
        """
        # 这里确实需要Initialize来获取完整物体列表
        event = self._ensure_event()
        self._last_event = event
        objects = event.metadata["objects"]

        self.initial_object_state = {}
        for obj in objects:
            if obj["pickupable"]:
                self.initial_object_state[obj["objectId"]] = {
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"],
                    "position": dict(obj["position"]),
                    "rotation": dict(obj["rotation"]),
                    "parentReceptacles": obj.get("parentReceptacles", []),
                }

        return self.initial_object_state

    def shuffle_room(self, num_moves: int = None) -> Dict:
        """
        按照RoomR数据集标准扰动规范打乱场景
        通过随机机器人动作改变物体空间位置
        """
        num_moves = num_moves or SHUFFLE_NUM_MOVES

        # 获取所有可拾取物体 — 用last_event或Initialize
        event = self._ensure_event()
        pickupable_objects = [
            obj for obj in event.metadata["objects"]
            if obj["pickupable"] and (not SHUFFLE_FORCE_VISIBLE or obj["visible"])
        ]

        if not pickupable_objects:
            pickupable_objects = [
                obj for obj in event.metadata["objects"] if obj["pickupable"]
            ]

        self.moved_objects = []
        successful_moves = 0

        for _ in range(num_moves):
            import time
            time.sleep(0.04)
            if not pickupable_objects:
                break

            # 随机选择一个可拾取物体
            obj = random.choice(pickupable_objects)
            obj_id = obj["objectId"]

            # 尝试拾取物体
            event = self.controller.step(
                action="PickupObject",
                objectId=obj_id,
                forceAction=True,
            )
            self._last_event = event

            if event.metadata["lastActionSuccess"]:
                # 找一个随机可达位置放下
                reachable_positions = self._get_reachable_positions()
                if reachable_positions:
                    target_pos = random.choice(reachable_positions)
                    # 先导航到目标位置附近
                    event = self.controller.step(
                        action="Teleport",
                        x=target_pos["x"],
                        z=target_pos["z"],
                        y=target_pos.get("y", 0),
                        rotation=random.choice([0, 90, 180, 270]),
                    )
                    self._last_event = event
                    # 放下物体
                    event = self.controller.step(action="DropHandObject")
                    self._last_event = event
                    if event.metadata["lastActionSuccess"]:
                        successful_moves += 1
                        self.moved_objects.append(obj_id)

                # 放回机器人到随机位置
                agent_start = self._get_random_reachable_position()
                if agent_start:
                    event = self.controller.step(
                        action="Teleport",
                        x=agent_start["x"],
                        z=agent_start["z"],
                        y=agent_start.get("y", 0),
                        rotation=agent_start.get("rotation", 0),
                    )
                    self._last_event = event

        # 重置视角
        agent_start = self._get_random_reachable_position()
        if agent_start:
            event = self.controller.step(
                action="Teleport",
                x=agent_start["x"],
                z=agent_start["z"],
                y=agent_start.get("y", 0),
                rotation=agent_start.get("rotation", 0),
            )
            self._last_event = event

        shuffle_info = {
            "total_attempts": num_moves,
            "successful_moves": successful_moves,
            "moved_objects": list(set(self.moved_objects)),
        }

        return shuffle_info

    def detect_changes(self) -> List[Dict]:
        """
        变化检测模块: 比对当前帧物体坐标与记忆库原始坐标
        筛选位置偏移目标，生成RoomR格式的待复原任务列表
        """

        event = self._ensure_event()
        #event = self.controller.last_event
        self._last_event = event
        current_objects = {
            obj["objectId"]: obj
            for obj in event.metadata["objects"]
            if obj["pickupable"]
        }

        rearrangement_list = []
        position_threshold = 0.1  # 位置偏移阈值(米)

        for obj_id, initial_info in self.initial_object_state.items():
            if obj_id in current_objects:
                current_pos = current_objects[obj_id]["position"]
                initial_pos = initial_info["position"]

                # 计算欧氏距离
                distance = np.sqrt(
                    (current_pos["x"] - initial_pos["x"]) ** 2
                    + (current_pos["y"] - initial_pos["y"]) ** 2
                    + (current_pos["z"] - initial_pos["z"]) ** 2
                )

                if distance > position_threshold:
                    rearrangement_list.append({
                        "objectId": obj_id,
                        "objectType": initial_info["objectType"],
                        "initial_position": initial_info["position"],
                        "current_position": dict(current_pos),
                        "displacement": round(distance, 3),
                        "parentReceptacles": initial_info.get("parentReceptacles", []),
                        "initial_rotation": initial_info.get("rotation", {}),
                    })

        self.moved_objects = [item["objectId"] for item in rearrangement_list]
        return rearrangement_list

    # =================== 观测与状态 ===================

    def _get_observation(self, event) -> Dict:
        """从event构建观测字典"""
        # RGB图像
        rgb = event.frame  # (H, W, 3) numpy array

        # 深度图
        depth = event.depth_frame if hasattr(event, "depth_frame") else None

        # 实例分割
        instance_seg = None
        if hasattr(event, "instance_segmentation_frame") and event.instance_segmentation_frame is not None:
            instance_seg = event.instance_segmentation_frame

        # 可见物体信息
        visible_objects = []
        for obj in event.metadata["objects"]:
            if obj["visible"] and obj["pickupable"]:
                visible_objects.append({
                    "objectId": obj["objectId"],
                    "objectType": obj["objectType"],
                    "position": dict(obj["position"]),
                    "distance": obj.get("distance", None),
                })
                self.discovered_objects.add(obj["objectId"])

        # Agent状态 — 直接从event metadata读取
        agent_meta = event.metadata.get("agent", {})
        agent_info = {}
        if agent_meta:
            agent_info = {
                "position": agent_meta.get("position", {}),
                "rotation": agent_meta.get("rotation", {}),
                "cameraHorizon": agent_meta.get("cameraHorizon", 0),
                "isHolding": len(agent_meta.get("inventoryObjects", [])) > 0,
            }
            if "inventoryObjects" in agent_meta and agent_meta["inventoryObjects"]:
                agent_info["heldObjectId"] = agent_meta["inventoryObjects"][0]["objectId"]

        return {
            "rgb": rgb,
            "depth": depth,
            "instance_seg": instance_seg,
            "visible_objects": visible_objects,
            "agent": agent_info,
            "event": event,
        }

    def get_state_for_ppo(self, target_info: Dict) -> Dict:
        """
        构建PPO所需的复合状态: RGB图像 + 目标物体位置
        使用缓存的last_event，不额外调用Initialize
        """
        event = self._ensure_event()
        obs = self._get_observation(event)

        # 添加目标信息到状态中
        obs["target"] = target_info
        return obs

    # =================== 奖励计算 ===================

    def _compute_dqn_reward(self, event, action_name: str) -> float:
        """计算DQN探索奖励 — 完全从event metadata读取，零额外API调用"""
        reward = DQN_CONFIG["step_penalty"]

        # 发现新物体奖励
        visible_pickupable = [
            obj for obj in event.metadata["objects"]
            if obj["visible"] and obj["pickupable"]
        ]
        for obj in visible_pickupable:
            if obj["objectId"] not in self.discovered_objects:
                reward += DQN_CONFIG["new_object_reward"]
                self.discovered_objects.add(obj["objectId"])

        # 覆盖新区域奖励 — 从event直接读取agent位置
        agent_pos = self._extract_agent_position(event)
        pos_key = self._pos_to_key(agent_pos)
        if pos_key not in self.visited_positions:
            reward += DQN_CONFIG["coverage_bonus"]
            self.visited_positions.add(pos_key)

        return reward

    def _compute_ppo_reward(self, event, action_name: str, target_info: Dict = None) -> float:
        """计算PPO复原奖励 — 完全从event metadata读取，零额外API调用"""
        reward = PPO_CONFIG["step_penalty"]

        if not event.metadata["lastActionSuccess"]:
            reward += PPO_CONFIG["wrong_action_penalty"]
            return reward

        if target_info is None:
            return reward

        agent_info = event.metadata.get("agent", {})

        # 拾取奖励
        if action_name == "PickupObject":
            held_objects = agent_info.get("inventoryObjects", [])
            if held_objects and held_objects[0]["objectId"] == target_info["objectId"]:
                reward += PPO_CONFIG["pickup_reward"]

        # 归位奖励 - 检查物体是否回到初始位置附近
        if action_name == "PutObject" or action_name == "DropHandObject":
            current_obj = None
            for obj in event.metadata["objects"]:
                if obj["objectId"] == target_info["objectId"]:
                    current_obj = obj
                    break

            if current_obj is not None:
                initial_pos = target_info.get("initial_position", {})
                current_pos = current_obj["position"]
                dist = np.sqrt(
                    (current_pos["x"] - initial_pos.get("x", 0)) ** 2
                    + (current_pos["z"] - initial_pos.get("z", 0)) ** 2
                )
                if dist < 0.3:
                    reward += PPO_CONFIG["put_reward"]
                else:
                    reward += PPO_CONFIG["approach_reward"] * max(0, 1.0 - dist)

        # 靠近目标奖励 — 从event直接读取agent位置
        if target_info and "initial_position" in target_info:
            agent_pos = self._extract_agent_position(event)
            target_pos = target_info["initial_position"]
            dist_to_target = np.sqrt(
                (agent_pos["x"] - target_pos.get("x", 0)) ** 2
                + (agent_pos["z"] - target_pos.get("z", 0)) ** 2
            )
            reward += PPO_CONFIG["approach_reward"] * max(0, 1.0 - dist_to_target)

        return reward

    # =================== 终止条件 ===================

    def _check_exploration_done(self) -> bool:
        """检查探索是否完成: 所有可拾取物体均被发现"""
        all_pickupable = set(self.initial_object_state.keys())
        if not all_pickupable:
            return False
        return len(self.discovered_objects & all_pickupable) == len(all_pickupable)

    def _check_rearrange_done(self, event, target_info: Dict = None) -> bool:
        """检查当前物体复原是否完成 — 直接从event读取，零额外API调用"""
        if target_info is None:
            return False

        # 直接从传入的event metadata中查找物体位置
        for obj in event.metadata["objects"]:
            if obj["objectId"] == target_info["objectId"]:
                current_pos = obj["position"]
                initial_pos = target_info.get("initial_position", {})
                dist = np.sqrt(
                    (current_pos["x"] - initial_pos.get("x", 0)) ** 2
                    + (current_pos["z"] - initial_pos.get("z", 0)) ** 2
                )
                return dist < 0.3
        return False

    # =================== 辅助方法 ===================

    def _extract_agent_position(self, event) -> Dict:
        """
        从event metadata中提取Agent位置
        不触发任何AI2-THOR API调用
        """
        agent = event.metadata.get("agent", {})
        if agent and "position" in agent:
            return dict(agent["position"])
        return {"x": 0, "y": 0, "z": 0}

    def _get_agent_position(self) -> Dict:
        """
        获取Agent当前网格位置
        优先从缓存的last_event读取，仅在没有缓存时才调用API
        """
        if self._last_event is not None:
            return self._extract_agent_position(self._last_event)
        # fallback: 仅在确实没有缓存时
        return {"x": 0, "y": 0, "z": 0}

    def _ensure_event(self) -> object:
        """确保有一个有效的event可用"""
        if self._last_event is not None:
            return self._last_event
        # 仅在第一次调用时执行Initialize
        event = self.controller.step(action="Initialize", **self.config)
        self._last_event = event
        return event

    def _pos_to_key(self, pos: Dict) -> Tuple:
        """将位置转为可哈希的key(网格化)"""
        grid_size = self.config.get("gridSize", 0.25)
        return (
            round(pos["x"] / grid_size) * grid_size,
            round(pos["z"] / grid_size) * grid_size,
        )

    def _get_reachable_positions(self) -> List[Dict]:
        """获取场景中所有可达位置(带缓存)"""
        if self._reachable_positions_cache is not None:
            return self._reachable_positions_cache
        event = self.controller.step(action="GetReachablePositions")
        self._last_event = event
        if "actionReturn" in event.metadata:
            self._reachable_positions_cache = event.metadata["actionReturn"]
            return self._reachable_positions_cache
        return []

    def _get_random_reachable_position(self) -> Dict:
        """随机获取一个可达位置"""
        positions = self._get_reachable_positions()
        if positions:
            pos = random.choice(positions)
            return {
                "x": pos["x"],
                "y": pos.get("y", 0),
                "z": pos["z"],
                "rotation": random.choice([0, 90, 180, 270]),
            }
        return {"x": 0, "y": 0, "z": 0, "rotation": 0}

    def get_all_pickupable_objects(self) -> List[Dict]:
        """获取场景中所有可拾取物体"""
        event = self._ensure_event()
        return [
            {
                "objectId": obj["objectId"],
                "objectType": obj["objectType"],
                "position": dict(obj["position"]),
                "rotation": dict(obj["rotation"]),
                "visible": obj["visible"],
                "pickupable": obj["pickupable"],
                "parentReceptacles": obj.get("parentReceptacles", []),
            }
            for obj in event.metadata["objects"]
            if obj["pickupable"]
        ]

    def get_reachable_positions_grid(self) -> List[Tuple]:
        """获取可达位置网格(用于A*导航)"""
        positions = self._get_reachable_positions()
        grid_size = self.config.get("gridSize", 0.25)
        return [
            (
                round(p["x"] / grid_size) * grid_size,
                round(p["z"] / grid_size) * grid_size,
            )
            for p in positions
        ]

    def teleport_to(self, x: float, z: float, rotation: float = 0, horizon: float = 0) -> Dict:
        """传送Agent到指定位置"""
        event = self.controller.step(
            action="Teleport",
            x=x,
            y=0,
            z=z,
            rotation=rotation,
            horizon=horizon,
        )
        self._last_event = event
        return self._get_observation(event)

    def render_frame(self) -> np.ndarray:
        """获取当前帧RGB图像 — 优先使用缓存，不额外调用Initialize"""
        if self._last_event is not None:
            return self._last_event.frame
        # 仅在没有缓存时才调用
        event = self.controller.step(action="Initialize", **self.config)
        self._last_event = event
        return event.frame

    @property
    def total_pickupable_count(self) -> int:
        """可拾取物体总数"""
        return len(self.initial_object_state)

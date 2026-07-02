"""
A* 导航模块
在AI2-THOR网格化空间中实现A*寻路算法
用于PPO复原阶段导航Agent到目标物体附近
"""

import heapq
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Set

from config import ASTAR_CONFIG


class AStarNavigator:
    """
    A*寻路导航器
    基于AI2-THOR的可达位置网格实现最优路径规划
    """

    def __init__(self, env=None, config: dict = None):
        self.env = env
        self.config = config or ASTAR_CONFIG
        self.grid_size = self.config["grid_size"]
        self.reachable_positions: Set[Tuple] = set()
        self.position_cache_valid = False

    def update_reachable_positions(self, env=None):
        """
        从环境中获取并缓存可达位置网格
        """
        if env:
            self.env = env

        if self.env is None:
            raise ValueError("Environment not set for A* navigator")

        raw_positions = self.env._get_reachable_positions()
        self.reachable_positions = set()
        for pos in raw_positions:
            grid_key = self._to_grid(pos["x"], pos["z"])
            self.reachable_positions.add(grid_key)

        self.position_cache_valid = True
        print(f"[AStarNavigator] 已缓存 {len(self.reachable_positions)} 个可达网格位置")

    def find_path(
        self,
        start: Dict,
        goal: Dict,
        allow_diagonal: bool = False,
    ) -> List[Dict]:
        """
        A*寻路主接口

        Args:
            start: 起始位置 {"x": float, "z": float}
            goal: 目标位置 {"x": float, "z": float}
            allow_diagonal: 是否允许对角线移动

        Returns:
            路径列表 [{"x": float, "z": float}, ...] 或空列表(不可达)
        """
        if not self.position_cache_valid:
            self.update_reachable_positions()

        start_grid = self._to_grid(start["x"], start["z"])
        goal_grid = self._to_grid(goal["x"], goal["z"])

        # 如果起点或终点不在可达网格中，找最近的可达点
        if start_grid not in self.reachable_positions:
            start_grid = self._find_nearest_reachable(start_grid)
            if start_grid is None:
                print("[AStarNavigator] 起点不可达且无最近可达点")
                return []

        if goal_grid not in self.reachable_positions:
            goal_grid = self._find_nearest_reachable(goal_grid)
            if goal_grid is None:
                print("[AStarNavigator] 终点不可达且无最近可达点")
                return []

        # A*核心搜索
        path_grids = self._astar_search(start_grid, goal_grid, allow_diagonal)

        if not path_grids:
            return []

        # 将网格坐标转换为世界坐标
        path_world = [
            {"x": g[0], "z": g[1]}
            for g in path_grids
        ]

        return path_world

    def navigate_to_object(
        self,
        env,
        target_position: Dict,
        target_distance: float = 1.0,
    ) -> List[Dict]:
        """
        导航到目标物体附近(不直接到达物体位置)

        Args:
            env: VRR环境实例
            target_position: 目标物体位置 {"x": float, "z": float}
            target_distance: 目标到达距离(米)

        Returns:
            导航路径
        """
        self.update_reachable_positions(env)

        # 获取Agent当前位置
        agent_pos = env._get_agent_position()

        # 寻找距离目标target_distance处的最近可达位置
        approach_positions = self._find_approach_positions(
            target_position, target_distance
        )

        if not approach_positions:
            # 如果没有合适的接近位置，直接导航到最近可达点
            return self.find_path(agent_pos, target_position)

        # 选择离Agent最近的接近位置
        best_approach = min(
            approach_positions,
            key=lambda p: self._heuristic(
                self._to_grid(agent_pos["x"], agent_pos["z"]),
                self._to_grid(p["x"], p["z"]),
            ),
        )

        return self.find_path(agent_pos, best_approach)

    def execute_path(self, env, path: List[Dict]) -> bool:
        """
        在环境中执行导航路径
        逐点Teleport移动

        Args:
            env: VRR环境实例
            path: 路径列表

        Returns:
            是否成功到达终点
        """
        if not path:
            return False

        for i, waypoint in enumerate(path):
            event = env.controller.step(
                action="Teleport",
                x=waypoint["x"],
                y=0,
                z=waypoint["z"],
                rotation=0,
            )
            env._last_event = event

        return True

    def execute_path_with_actions(self, env, path: List[Dict]) -> bool:
        """
        使用MoveAhead/Rotate动作执行导航路径(更真实)
        逐步移动并旋转到下一个航点方向

        Args:
            env: VRR环境实例
            path: 路径列表

        Returns:
            是否成功到达终点
        """
        if not path:
            return False

        for waypoint in path:
            agent_pos = env._get_agent_position()
            agent_rot = env.controller.last_event.metadata.get("agent", {}).get("rotation", {})

            # 计算目标方向
            dx = waypoint["x"] - agent_pos["x"]
            dz = waypoint["z"] - agent_pos["z"]
            target_rotation = math.degrees(math.atan2(dx, dz)) % 360

            # 旋转到目标方向
            current_rotation = agent_rot.get("y", 0) % 360
            rotation_diff = (target_rotation - current_rotation) % 360

            # 选择最短旋转方向
            if rotation_diff > 180:
                rotation_diff -= 360

            # 执行旋转
            while abs(rotation_diff) > 5:
                if rotation_diff > 0:
                    event = env.controller.step(action="RotateRight")
                    env._last_event = event
                    rotation_diff -= 90
                else:
                    event = env.controller.step(action="RotateLeft")
                    env._last_event = event
                    rotation_diff += 90

            # 前进到航点
            dist = math.sqrt(dx ** 2 + dz ** 2)
            steps = max(1, int(dist / env.config.get("gridSize", 0.25)))
            for _ in range(steps):
                event = env.controller.step(action="MoveAhead")
                env._last_event = event
                if not event.metadata["lastActionSuccess"]:
                    break

        return True

    # =================== A* 核心算法 ===================

    def _astar_search(
        self,
        start: Tuple,
        goal: Tuple,
        allow_diagonal: bool = False,
    ) -> List[Tuple]:
        """
        A*搜索算法核心实现

        使用曼哈顿距离/欧氏距离作为启发函数
        f(n) = g(n) + h(n)
        """
        # 优先队列: (f_score, counter, node)
        counter = 0
        open_set = [(0, counter, start)]

        # 记录来源
        came_from: Dict[Tuple, Tuple] = {}

        # g_score: 从起点到当前节点的实际代价
        g_score = {start: 0}

        # f_score: g_score + heuristic
        f_score = {start: self._heuristic(start, goal)}

        # 已访问集合
        closed_set: Set[Tuple] = set()

        while open_set:
            _, _, current = heapq.heappop(open_set)

            if current == goal:
                # 重建路径
                return self._reconstruct_path(came_from, current)

            if current in closed_set:
                continue
            closed_set.add(current)

            # 扩展邻居节点
            for neighbor in self._get_neighbors(current, allow_diagonal):
                if neighbor in closed_set:
                    continue

                if neighbor not in self.reachable_positions:
                    continue

                # 计算移动代价
                move_cost = self._move_cost(current, neighbor)
                tentative_g = g_score[current] + move_cost

                if tentative_g < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)

                    counter += 1
                    heapq.heappush(open_set, (f_score[neighbor], counter, neighbor))

        # 没有找到路径
        return []

    def _get_neighbors(self, pos: Tuple, allow_diagonal: bool = False) -> List[Tuple]:
        """获取网格邻居"""
        x, z = pos
        gs = self.grid_size

        # 四方向移动
        neighbors = [
            (x + gs, z),      # 前
            (x - gs, z),      # 后
            (x, z + gs),      # 右
            (x, z - gs),      # 左
        ]

        if allow_diagonal:
            neighbors.extend([
                (x + gs, z + gs),  # 右前
                (x + gs, z - gs),  # 左前
                (x - gs, z + gs),  # 右后
                (x - gs, z - gs),  # 左后
            ])

        return neighbors

    def _heuristic(self, a: Tuple, b: Tuple) -> float:
        """
        启发函数: 使用欧氏距离 × 权重
        """
        w = self.config["heuristic_weight"]
        return w * math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

    def _move_cost(self, a: Tuple, b: Tuple) -> float:
        """计算两个相邻节点间的移动代价"""
        dx = abs(a[0] - b[0])
        dz = abs(a[1] - b[1])
        if dx > 0 and dz > 0:
            # 对角线移动代价: sqrt(2) * grid_size
            return math.sqrt(2) * self.grid_size
        return self.grid_size

    # =================== 辅助方法 ===================

    def _to_grid(self, x: float, z: float) -> Tuple:
        """世界坐标 → 网格坐标"""
        gs = self.grid_size
        return (round(x / gs) * gs, round(z / gs) * gs)

    def _reconstruct_path(self, came_from: Dict, current: Tuple) -> List[Tuple]:
        """从came_from字典重建路径"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    def _find_nearest_reachable(self, pos: Tuple) -> Optional[Tuple]:
        """找离pos最近的可达网格点"""
        min_dist = float("inf")
        nearest = None
        for rp in self.reachable_positions:
            d = abs(rp[0] - pos[0]) + abs(rp[1] - pos[1])
            if d < min_dist:
                min_dist = d
                nearest = rp
        return nearest

    def _find_approach_positions(
        self, target: Dict, distance: float
    ) -> List[Dict]:
        """
        寻找目标位置周围指定距离内的可达位置
        用于"接近但不直接到达"的导航场景
        """
        target_grid = self._to_grid(target["x"], target["z"])
        approach_positions = []

        for rp in self.reachable_positions:
            dist = math.sqrt(
                (rp[0] - target_grid[0]) ** 2 + (rp[1] - target_grid[1]) ** 2
            )
            if 0.5 < dist < distance + 0.5:
                approach_positions.append({"x": rp[0], "z": rp[1]})

        return approach_positions

    def get_path_length(self, path: List[Dict]) -> float:
        """计算路径总长度"""
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            dx = path[i]["x"] - path[i - 1]["x"]
            dz = path[i]["z"] - path[i - 1]["z"]
            total += math.sqrt(dx ** 2 + dz ** 2)
        return total

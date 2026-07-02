"""
工具函数模块
图像处理、数据转换等通用工具
"""

import cv2
import numpy as np
import torch
from typing import Dict, Tuple, List


def resize_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    """缩放图像到目标尺寸"""
    return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """归一化图像到[0,1]"""
    return image.astype(np.float32) / 255.0


def rgb_to_tensor(rgb: np.ndarray, target_size: Tuple[int, int] = (84, 84)) -> torch.Tensor:
    """
    RGB图像 → PyTorch Tensor
    (H, W, 3) uint8 → (1, 3, H', W') float32
    """
    resized = resize_image(rgb, target_size)
    normalized = normalize_image(resized)
    # HWC → CHW
    transposed = np.transpose(normalized, (2, 0, 1))
    return torch.from_numpy(transposed).unsqueeze(0)


def compute_distance(pos1: Dict, pos2: Dict) -> float:
    """计算两个3D位置之间的欧氏距离"""
    return np.sqrt(
        (pos1.get("x", 0) - pos2.get("x", 0)) ** 2
        + (pos1.get("y", 0) - pos2.get("y", 0)) ** 2
        + (pos1.get("z", 0) - pos2.get("z", 0)) ** 2
    )


def compute_2d_distance(pos1: Dict, pos2: Dict) -> float:
    """计算XZ平面距离"""
    return np.sqrt(
        (pos1.get("x", 0) - pos2.get("x", 0)) ** 2
        + (pos1.get("z", 0) - pos2.get("z", 0)) ** 2
    )


def get_direction_vector(from_pos: Dict, to_pos: Dict) -> Dict:
    """获取从from到to的方向向量"""
    return {
        "x": to_pos.get("x", 0) - from_pos.get("x", 0),
        "y": to_pos.get("y", 0) - from_pos.get("y", 0),
        "z": to_pos.get("z", 0) - from_pos.get("z", 0),
    }


def angle_between_vectors(v1: Dict, v2: Dict) -> float:
    """计算两个2D向量之间的角度(度)"""
    cross = v1["x"] * v2["z"] - v1["z"] * v2["x"]
    dot = v1["x"] * v2["x"] + v1["z"] * v2["z"]
    return np.degrees(np.arctan2(cross, dot))


def position_to_grid(pos: Dict, grid_size: float = 0.25) -> Tuple:
    """世界坐标 → 网格坐标"""
    return (
        round(pos.get("x", 0) / grid_size) * grid_size,
        round(pos.get("z", 0) / grid_size) * grid_size,
    )


def create_segmentation_mask(instance_seg: np.ndarray, object_id_map: Dict) -> np.ndarray:
    """
    从实例分割图创建物体ID掩码
    """
    mask = np.zeros(instance_seg.shape[:2], dtype=np.int32)
    for color, obj_id in object_id_map.items():
        if isinstance(color, tuple):
            match = np.all(instance_seg == np.array(color), axis=-1)
            mask[match] = obj_id
    return mask


def overlay_text_on_frame(frame: np.ndarray, text: str,
                          position: Tuple[int, int] = (10, 30),
                          color: Tuple[int, int, int] = (255, 255, 255),
                          font_scale: float = 1.0) -> np.ndarray:
    """在帧上叠加文字"""
    result = frame.copy()
    cv2.putText(result, text, position, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, color, 2, cv2.LINE_AA)
    return result


class RunningAverage:
    """滑动平均计算器"""

    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.values = []

    def push(self, value: float):
        self.values.append(value)
        if len(self.values) > self.window_size:
            self.values.pop(0)

    def mean(self) -> float:
        return np.mean(self.values) if self.values else 0.0

    def std(self) -> float:
        return np.std(self.values) if self.values else 0.0

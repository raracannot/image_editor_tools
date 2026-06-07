import bpy
import numpy as np
from .base import BaseTool


class EdgeDetectTool(BaseTool):
    tool_id = 'edge_detect'
    label = '边缘检测'

    @staticmethod
    def get_properties():
        return {}

    @staticmethod
    def draw_panel(layout, props):
        layout.label(text="Sobel 算子，无参数")

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_rgb_to_gray, np_sobel_edge
        gray = np_rgb_to_gray(np_array)
        edge = np_sobel_edge(gray)
        result = np.zeros_like(np_array)
        result[:, :, 0] = edge
        result[:, :, 1] = edge
        result[:, :, 2] = edge
        result[:, :, 3] = np_array[:, :, 3]
        return result

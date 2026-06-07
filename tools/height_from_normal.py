import bpy
import numpy as np
from .base import BaseTool


class HeightFromNormalTool(BaseTool):
    tool_id = 'height_from_normal'
    label = '法线还原高度'

    @staticmethod
    def get_properties():
        return {}

    @staticmethod
    def draw_panel(layout, props):
        layout.label(text="FFT Poisson 求解，无参数")

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_normal_to_height_fft
        rgb = np_array[:, :, :3]
        alpha = np_array[:, :, 3].copy()
        height = np_normal_to_height_fft(rgb)
        result = np.zeros_like(np_array)
        result[:, :, 0] = height
        result[:, :, 1] = height
        result[:, :, 2] = height
        result[:, :, 3] = alpha
        return result

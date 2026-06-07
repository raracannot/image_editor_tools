import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class NoiseTool(BaseTool):
    tool_id = 'noise'
    label = '噪点'

    @staticmethod
    def get_properties():
        return {
            'noise_type': bpy.props.EnumProperty(
                name="噪点类型",
                description="噪点生成算法",
                items=[
                    ('GAUSSIAN', "高斯噪点", "正态分布随机噪点"),
                    ('SALT_PEPPER', "椒盐噪点", "随机黑白像素"),
                ],
                default='GAUSSIAN',
                update=_on_param_update,
            ),
            'noise_intensity': bpy.props.FloatProperty(
                name="强度",
                description="噪点强度（高斯=标准差，椒盐=密度）",
                default=0.05,
                min=0.0,
                max=0.3,
                soft_min=0.0,
                soft_max=0.15,
                subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "noise_type")
        layout.prop(props, "noise_intensity", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_add_gaussian_noise, np_add_salt_pepper_noise
        if props.noise_type == 'GAUSSIAN':
            return np_add_gaussian_noise(np_array, 0.0, props.noise_intensity).astype(np.float32, copy=False)
        return np_add_salt_pepper_noise(np_array, props.noise_intensity).astype(np.float32, copy=False)

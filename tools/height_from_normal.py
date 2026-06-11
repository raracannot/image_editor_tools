import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class HeightFromNormalTool(BaseTool):
    tool_id = 'height_from_normal'
    label = '法线还原高度'

    @staticmethod
    def get_properties():
        return {
            'hfn_flip_g': bpy.props.BoolProperty(
                name="翻转 G 通道",
                description="OpenGL (Y+) ⇄ DirectX (Y-) 法线格式切换",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "hfn_flip_g", text="翻转 G 通道")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_normal_to_height_fft
        rgb = np_array[:, :, :3]
        alpha = np_array[:, :, 3].copy()
        height = np_normal_to_height_fft(rgb, flip_g=props.hfn_flip_g)
        result = np.zeros_like(np_array)
        result[:, :, 0] = height
        result[:, :, 1] = height
        result[:, :, 2] = height
        result[:, :, 3] = alpha
        return result

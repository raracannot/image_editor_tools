import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class SharpenTool(BaseTool):
    tool_id = 'sharpen'
    label = 'USM锐化'

    @staticmethod
    def get_properties():
        return {
            'sharpen_sigma': bpy.props.FloatProperty(
                name="模糊半径",
                description="USM 底层模糊的 σ 值",
                default=1.0,
                min=0.1,
                max=10.0,
                soft_min=0.3,
                soft_max=5.0,
                update=_on_param_update,
            ),
            'sharpen_amount': bpy.props.FloatProperty(
                name="锐化强度",
                description="锐化量（1=轻微，2=中等，3+=强锐化）",
                default=1.5,
                min=0.0,
                max=5.0,
                soft_min=0.0,
                soft_max=3.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "sharpen_sigma", slider=True)
        layout.prop(props, "sharpen_amount", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_unsharp_mask
        return np_unsharp_mask(np_array, props.sharpen_sigma, props.sharpen_amount).astype(np.float32, copy=False)

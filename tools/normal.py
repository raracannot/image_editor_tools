import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class NormalTool(BaseTool):
    tool_id = 'normal'
    label = '法线生成'

    @staticmethod
    def get_properties():
        return {
            'normal_strength': bpy.props.FloatProperty(
                name="强度",
                description="法线凹凸强度",
                default=2.0,
                min=0.1,
                max=10.0,
                soft_min=0.5,
                soft_max=5.0,
                update=_on_param_update,
            ),
            'normal_invert': bpy.props.BoolProperty(
                name="反转绿色通道",
                description="反转法线Y方向（切换凹凸方向感知）",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "normal_strength", slider=True)
        layout.prop(props, "normal_invert")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_height_to_normal
        return np_height_to_normal(np_array, props.normal_strength, props.normal_invert)

import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class HighPassTool(BaseTool):
    tool_id = 'high_pass'
    label = '高反差保留'

    @staticmethod
    def get_properties():
        return {
            'high_pass_sigma': bpy.props.FloatProperty(
                name="模糊半径",
                description="控制保留细节的尺度，值越大保留更大尺度的结构",
                default=3.0,
                min=0.1,
                max=50.0,
                soft_min=0.5,
                soft_max=20.0,
                update=_on_param_update,
            ),
            'high_pass_contrast': bpy.props.FloatProperty(
                name="细节强度",
                description="高频细节的加强倍率",
                default=1.0,
                min=0.0,
                max=5.0,
                soft_min=0.0,
                soft_max=3.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "high_pass_sigma", slider=True)
        layout.prop(props, "high_pass_contrast", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_high_pass
        return np_high_pass(np_array, props.high_pass_sigma, props.high_pass_contrast)

import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class InvertTool(BaseTool):
    tool_id = 'invert'
    label = '反相'

    @staticmethod
    def get_properties():
        return {
            'invert_r': bpy.props.FloatProperty(
                name="R 通道", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'invert_g': bpy.props.FloatProperty(
                name="G 通道", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'invert_b': bpy.props.FloatProperty(
                name="B 通道", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "invert_r", text="R 通道", slider=True)
        layout.prop(props, "invert_g", text="G 通道", slider=True)
        layout.prop(props, "invert_b", text="B 通道", slider=True)

    @staticmethod
    def process(np_array, props):
        fr = props.invert_r
        fg = props.invert_g
        fb = props.invert_b
        if fr <= 0 and fg <= 0 and fb <= 0:
            return np_array
        result = np_array.copy()
        if fr > 0:
            result[..., 0] += (1.0 - 2.0 * np_array[..., 0]) * fr
        if fg > 0:
            result[..., 1] += (1.0 - 2.0 * np_array[..., 1]) * fg
        if fb > 0:
            result[..., 2] += (1.0 - 2.0 * np_array[..., 2]) * fb
        return result

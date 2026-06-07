import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class LevelsTool(BaseTool):
    tool_id = 'levels'
    label = '色阶'

    @staticmethod
    def get_properties():
        return {
            'levels_in_black': bpy.props.FloatProperty(
                name="输入黑点", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'levels_in_white': bpy.props.FloatProperty(
                name="输入白点", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'levels_gamma': bpy.props.FloatProperty(
                name="中间调", default=1.0, min=0.1, max=9.99,
                soft_min=0.3, soft_max=3.0,
                update=_on_param_update,
            ),
            'levels_out_black': bpy.props.FloatProperty(
                name="输出黑点", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'levels_out_white': bpy.props.FloatProperty(
                name="输出白点", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        box = layout.box()
        box.label(text="输入色阶", icon='TRACKING_FORWARDS')
        col = box.column(align=True)
        col.prop(props, "levels_in_black", text="黑点", slider=True)
        col.prop(props, "levels_gamma", text="中间调", slider=True)
        col.prop(props, "levels_in_white", text="白点", slider=True)
        box = layout.box()
        box.label(text="输出色阶", icon='TRACKING_BACKWARDS')
        col = box.column(align=True)
        col.prop(props, "levels_out_black", text="黑点", slider=True)
        col.prop(props, "levels_out_white", text="白点", slider=True)

    @staticmethod
    def process(np_array, props):
        rgb = np_array[..., :3].astype(np.float32)
        alpha = np_array[..., 3:4]
        ib = props.levels_in_black
        iw = max(ib + 0.001, props.levels_in_white)
        g = props.levels_gamma
        ob = props.levels_out_black
        ow = props.levels_out_white

        normalized = np.clip((rgb - ib) / (iw - ib), 0.0, 1.0)
        corrected = np.power(normalized, 1.0 / max(0.01, g))
        mapped = corrected * (ow - ob) + ob
        return np.concatenate([np.clip(mapped, 0, 1), alpha], axis=-1)

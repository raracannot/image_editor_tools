import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class DepthTool(BaseTool):
    tool_id = 'depth'
    label = '色深优化'

    @staticmethod
    def get_properties():
        return {
            'depth_mode': bpy.props.EnumProperty(
                name="模式",
                description="色深处理模式",
                items=[
                    ('DEBAND', "去条带", "注入微噪声打破量化断层，为后续调色提供缓冲"),
                    ('DITHER', "仿色", "降位深时注入TPDF三角噪声，掩盖色阶断层"),
                ],
                default='DEBAND',
                update=_on_param_update,
            ),
            'depth_bits': bpy.props.IntProperty(
                name="位深",
                description="去条带=源位深(越低越激进) / 仿色=目标位深",
                default=8,
                min=4,
                max=16,
                soft_min=6,
                soft_max=12,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "depth_mode")
        layout.prop(props, "depth_bits", slider=True)
        if props.depth_mode == 'DEBAND':
            layout.label(text="打破已固化的量化边界", icon='INFO')
        else:
            layout.label(text="注入TPDF噪声掩盖色阶", icon='INFO')

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_deband, np_dither_tpdf
        if props.depth_mode == 'DEBAND':
            return np_deband(np_array, props.depth_bits).astype(np.float32, copy=False)
        return np_dither_tpdf(np_array, props.depth_bits).astype(np.float32, copy=False)

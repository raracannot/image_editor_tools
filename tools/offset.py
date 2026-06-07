import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class OffsetTool(BaseTool):
    tool_id = 'offset'
    label = '位移'

    @staticmethod
    def get_properties():
        return {
            'offset_mode': bpy.props.EnumProperty(
                name="模式",
                description="越界像素处理方式",
                items=[
                    ('WRAP', "折回", "越界像素从另一侧绕回"),
                    ('TRANSPARENT', "透明", "越界像素变为透明"),
                ],
                default='WRAP',
                update=_on_param_update,
            ),
            'offset_h_factor': bpy.props.FloatProperty(
                name="水平",
                description="水平位移比例",
                default=0.0,
                min=-1.0,
                max=1.0,
                soft_min=-1.0,
                soft_max=1.0,
                subtype='FACTOR',
                update=_on_param_update,
            ),
            'offset_v_factor': bpy.props.FloatProperty(
                name="垂直",
                description="垂直位移比例",
                default=0.0,
                min=-1.0,
                max=1.0,
                soft_min=-1.0,
                soft_max=1.0,
                subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "offset_mode")
        layout.prop(props, "offset_h_factor", text="水平", slider=True)
        layout.prop(props, "offset_v_factor", text="垂直", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_offset
        h, w = np_array.shape[:2]
        dx = int(round(w * props.offset_h_factor))
        dy = int(round(h * props.offset_v_factor))

        if props.offset_mode == 'WRAP':
            return np_offset(np_array.copy(), dx, dy)

        result = np.zeros_like(np_array)
        if dx >= 0:
            src_x0, src_x1 = 0, w - dx
            dst_x0, dst_x1 = dx, w
        else:
            src_x0, src_x1 = -dx, w
            dst_x0, dst_x1 = 0, w + dx
        if dy >= 0:
            src_y0, src_y1 = 0, h - dy
            dst_y0, dst_y1 = dy, h
        else:
            src_y0, src_y1 = -dy, h
            dst_y0, dst_y1 = 0, h + dy

        if dst_x0 < dst_x1 and dst_y0 < dst_y1:
            result[dst_y0:dst_y1, dst_x0:dst_x1] = np_array[src_y0:src_y1, src_x0:src_x1]
        return result

import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class CrystallizeTool(BaseTool):
    tool_id = 'crystallize'
    label = '晶格化'

    @staticmethod
    def get_properties():
        return {
            'crystallize_count': bpy.props.IntProperty(
                name="晶格数量",
                description="控制随机采样点数量，越多越接近原图",
                default=200,
                min=5,
                max=500000,
                soft_min=5,
                soft_max=30000,
                update=_on_param_update,
            ),
            'crystallize_fast': bpy.props.BoolProperty(
                name="快速模式",
                description="先缩到 1/2 做晶格化再还原，大幅加速",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "crystallize_count", slider=True)
        layout.prop(props, "crystallize_fast")

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_voronoi_crystallize_spatial, np_resize_img
        if props.crystallize_fast:
            h, w = np_array.shape[:2]
            hw, hh = max(1, w // 2), max(1, h // 2)
            small = np_resize_img(np_array, hw, hh)
            result = np_voronoi_crystallize_spatial(small, props.crystallize_count)
            return np_resize_img(result, w, h)
        return np_voronoi_crystallize_spatial(np_array, props.crystallize_count)

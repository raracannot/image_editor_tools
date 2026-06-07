import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class SeamlessTool(BaseTool):
    tool_id = 'seamless'
    label = '贴图无缝化'

    @staticmethod
    def get_properties():
        return {
            'seamless_method': bpy.props.EnumProperty(
                name="方法",
                description="无缝化算法",
                items=[
                    ('BASIC', "基础混合", "线性渐变边缘混合，快速、保持尺寸"),
                    ('ADVANCED', "高级缝合", "Seam Carving + 双频融合，更自然但可能裁切"),
                ],
                default='BASIC',
                update=_on_param_update,
            ),
            'seamless_blend_ratio': bpy.props.FloatProperty(
                name="混合宽度",
                description="边缘混合带占短边的比例",
                default=0.125,
                min=0.01,
                max=0.5,
                soft_min=0.02,
                soft_max=0.25,
                subtype='FACTOR',
                update=_on_param_update,
            ),
            'seamless_overlap': bpy.props.FloatProperty(
                name="重叠宽度",
                description="拼接重叠区域占图像的比例",
                default=0.2,
                min=0.05,
                max=0.5,
                soft_min=0.1,
                soft_max=0.3,
                subtype='FACTOR',
                update=_on_param_update,
            ),
            'seamless_beta': bpy.props.FloatProperty(
                name="梯度惩罚",
                description="边缘结构保护强度（越大越保护边缘）",
                default=5.0,
                min=1.0,
                max=20.0,
                soft_min=2.0,
                soft_max=10.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "seamless_method")
        if props.seamless_method == 'BASIC':
            layout.prop(props, "seamless_blend_ratio", slider=True)
        else:
            layout.prop(props, "seamless_overlap", slider=True)
            layout.prop(props, "seamless_beta", slider=True)
            layout.label(text="注：高级缝合可能裁切图像", icon='INFO')

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_make_seamless_tile, np_make_texture_seamless, np_resize_img
        if props.seamless_method == 'BASIC':
            return np_make_seamless_tile(np_array, props.seamless_blend_ratio)
        result = np_make_texture_seamless(np_array, props.seamless_overlap, props.seamless_beta)
        h, w = np_array.shape[:2]
        rh, rw = result.shape[:2]
        if rh != h or rw != w:
            result = np_resize_img(result, w, h)
        return result

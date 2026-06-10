import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class ColorTransferTool(BaseTool):
    tool_id = 'color_transfer'
    label = '色彩迁徙'

    _cache_key = None
    _cache_transferred = None

    @staticmethod
    def get_properties():
        return {
            'color_transfer_ref': bpy.props.PointerProperty(
                name="参考图",
                description="用于迁移色彩风格的参考图像",
                type=bpy.types.Image,
                update=_on_param_update,
            ),
            'color_transfer_blend': bpy.props.FloatProperty(
                name="混合",
                description="0=原图，1=完全迁移",
                default=1.0,
                min=0.0,
                max=1.0,
                soft_min=0.0,
                soft_max=1.0,
                subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "color_transfer_ref", text="参考图")
        if props.color_transfer_ref is None:
            layout.label(text="请选择一幅参考图", icon='INFO')
        else:
            layout.prop(props, "color_transfer_blend", text="混合", slider=True)

    @staticmethod
    def process(np_array, props):
        ref_img = props.color_transfer_ref
        if ref_img is None:
            ColorTransferTool._cache_key = None
            ColorTransferTool._cache_transferred = None
            return np_array

        blend = props.color_transfer_blend
        if blend <= 0.0:
            return np_array

        h, w = np_array.shape[:2]
        ref_key = (ref_img.name, ref_img.size[0], ref_img.size[1])
        np_key = (
            h, w,
            float(np_array[0, 0, 0]),
            float(np_array[min(h - 1, h // 2), min(w - 1, w // 2), 1]),
            float(np_array[-1, -1, 2]),
        )
        cache_key = (ref_key, np_key)

        if ColorTransferTool._cache_key != cache_key or ColorTransferTool._cache_transferred is None:
            from ..utils.np_img_utils import (
                blimg_2_npimg,
                np_resize_img,
                np_reinhard_color_transfer,
            )
            ref_np = blimg_2_npimg(ref_img)
            rh, rw = ref_np.shape[:2]
            if rh != h or rw != w:
                ref_np = np_resize_img(ref_np, w, h)
            ColorTransferTool._cache_key = cache_key
            ColorTransferTool._cache_transferred = np_reinhard_color_transfer(
                ref_np, np_array, l_weight=0.5, ab_weight=1.0,
            )

        return np.clip(
            np_array * (1.0 - blend) + ColorTransferTool._cache_transferred * blend,
            0.0, 1.0,
        )

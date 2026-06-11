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
            'normal_detail': bpy.props.FloatProperty(
                name="细节",
                description="高频细节保留强度 (0=仅大形, 1=全细节)",
                default=0.5,
                min=0.0,
                max=1.0,
                soft_min=0.0,
                soft_max=1.0,
                subtype='FACTOR',
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
        layout.prop(props, "normal_detail", text="细节", slider=True)
        layout.prop(props, "normal_invert")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_height_to_normal, np_blur_img
        strength = props.normal_strength
        detail = props.normal_detail
        invert = props.normal_invert

        if detail <= 0.0:
            return np_height_to_normal(np_array, strength, invert)

        full = np_height_to_normal(np_array, strength, invert)

        gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114
        if gray.ndim == 3:
            gray = gray[:, :, 0] if gray.shape[-1] == 1 else gray
        blurred_gray = np_blur_img(gray, 1.5, 'edge')
        blurred_h = np.zeros_like(np_array)
        blurred_h[:, :, 0] = blurred_gray
        blurred_h[:, :, 1] = blurred_gray
        blurred_h[:, :, 2] = blurred_gray
        blurred_h[:, :, 3] = np_array[:, :, 3]
        base = np_height_to_normal(blurred_h, strength, invert)

        result = np.zeros_like(full)
        result[:, :, :3] = np.clip(base[:, :, :3] + (full[:, :, :3] - base[:, :, :3]) * detail, 0, 1)
        result[:, :, 3] = full[:, :, 3]
        return result

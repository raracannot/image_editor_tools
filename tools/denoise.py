import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class DenoiseTool(BaseTool):
    tool_id = 'denoise'
    label = '减少杂色'

    @staticmethod
    def get_properties():
        return {
            'denoise_strength': bpy.props.FloatProperty(
                name="强度",
                description="降噪强度",
                default=5.0,
                min=0.0,
                max=10.0,
                soft_min=0.0,
                soft_max=10.0,
                update=_on_param_update,
            ),
            'denoise_detail': bpy.props.IntProperty(
                name="保留细节",
                description="边缘保护程度",
                default=50,
                min=0,
                max=100,
                soft_min=0,
                soft_max=100,
                subtype='PERCENTAGE',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "denoise_strength", text="强度", slider=True)
        layout.prop(props, "denoise_detail", text="保留细节", slider=True)

    @staticmethod
    def process(np_array, props):
        strength = props.denoise_strength
        if strength <= 0.0:
            return np_array

        from ..np_img_utils import np_gaussian_filter, np_rgb_to_gray, np_sobel_edge

        sigma = strength * 0.8
        alpha = np_array[:, :, 3]
        rgb = np_array[:, :, :3]
        blurred = np_gaussian_filter(rgb, sigma)

        detail = props.denoise_detail / 100.0
        if detail > 0.0:
            gray = np_rgb_to_gray(np_array)
            edges = np_sobel_edge(gray)
            mask = 1.0 - np.clip(edges * detail * 3.0, 0.0, 1.0)
            mask_3d = mask[:, :, np.newaxis]
            result_rgb = rgb * mask_3d + blurred * (1.0 - mask_3d)
        else:
            blend = strength / 10.0
            result_rgb = rgb * (1.0 - blend) + blurred * blend

        result = np.zeros_like(np_array)
        result[:, :, :3] = result_rgb
        result[:, :, 3] = alpha
        return np.clip(result, 0.0, 1.0)

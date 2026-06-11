import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class HeightToAOTool(BaseTool):
    tool_id = 'height_to_ao'
    label = '高度→AO'

    @staticmethod
    def get_properties():
        return {
            'htao_radius': bpy.props.FloatProperty(
                name="半径", default=0.05, min=0.005, max=0.5,
                soft_min=0.01, soft_max=0.2, subtype='FACTOR',
                update=_on_param_update, description="采样半径 (相对图像尺寸)",
            ),
            'htao_intensity': bpy.props.FloatProperty(
                name="强度", default=1.0, min=0.0, max=3.0,
                soft_min=0.5, soft_max=2.0,
                update=_on_param_update, description="AO 强度倍数",
            ),
            'htao_samples': bpy.props.IntProperty(
                name="采样数", default=16, min=4, max=64,
                update=_on_param_update, description="半球采样方向数 (越大越精确越慢)",
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "htao_radius", slider=True)
        layout.prop(props, "htao_intensity", slider=True)
        layout.prop(props, "htao_samples")

    @staticmethod
    def process(np_array, props):
        gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114
        h, w = gray.shape[:2]
        radius = max(1, int(min(h, w) * props.htao_radius))
        n_samples = props.htao_samples
        intensity = props.htao_intensity

        ao = np.zeros_like(gray, dtype=np.float32)
        angles = np.linspace(0, 2 * np.pi, n_samples, endpoint=False, dtype=np.float32)
        cos_a = np.cos(angles)
        sin_a = np.sin(angles)

        for r in range(1, radius + 1):
            for i in range(n_samples):
                dx = int(round(r * cos_a[i]))
                dy = int(round(r * sin_a[i]))
                shifted = np.roll(gray, (dy, dx), axis=(0, 1))
                h_diff = shifted - gray
                ao += np.maximum(h_diff, 0.0) / r

        ao = ao / (radius * n_samples)
        ao = np.clip(1.0 - ao * intensity * 2.0, 0.0, 1.0)
        return _gray_to_rgba(np_array, ao)


class CurvatureTool(BaseTool):
    tool_id = 'curvature'
    label = '曲率图'

    @staticmethod
    def get_properties():
        return {
            'curv_input': bpy.props.EnumProperty(
                name="输入类型", items=[
                    ('HEIGHT', "高度图", "从灰度贴图计算"),
                    ('NORMAL', "法线贴图", "从法线贴图计算"),
                ], default='HEIGHT', update=_on_param_update,
            ),
            'curv_radius': bpy.props.IntProperty(
                name="半径", default=3, min=1, max=20,
                update=_on_param_update, description="曲率计算半径 (像素)",
            ),
            'curv_intensity': bpy.props.FloatProperty(
                name="强度", default=1.0, min=0.1, max=5.0,
                soft_min=0.5, soft_max=3.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "curv_input")
        layout.prop(props, "curv_radius")
        layout.prop(props, "curv_intensity", slider=True)

    @staticmethod
    def process(np_array, props):
        if props.curv_input == 'NORMAL':
            gray = _normal_to_height_fast(np_array)
        else:
            gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114

        r = props.curv_radius
        gy, gx = np.gradient(gray)
        gyy, _ = np.gradient(gy)
        _, gxx = np.gradient(gx)

        curv = gxx + gyy
        mean_abs = np.mean(np.abs(curv)) + 1e-6
        curv = np.clip(curv / mean_abs * props.curv_intensity, -1.0, 1.0)
        curv = (curv + 1.0) / 2.0
        return _gray_to_rgba(np_array, curv.astype(np.float32))


class EdgeDilateTool(BaseTool):
    tool_id = 'edge_dilate'
    label = '边缘扩展'

    @staticmethod
    def get_properties():
        return {
            'edil_pixels': bpy.props.IntProperty(
                name="扩展像素", default=2, min=1, max=64,
                update=_on_param_update, description="向外扩展的像素数",
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "edil_pixels")

    @staticmethod
    def process(np_array, props):
        n = props.edil_pixels
        result = np_array.copy()
        for _ in range(n):
            alpha = result[:, :, 3]
            dilated = alpha.copy()
            dilated[1:, :] = np.maximum(dilated[1:, :], alpha[:-1, :])
            dilated[:-1, :] = np.maximum(dilated[:-1, :], alpha[1:, :])
            dilated[:, 1:] = np.maximum(dilated[:, 1:], alpha[:, :-1])
            dilated[:, :-1] = np.maximum(dilated[:, :-1], alpha[:, 1:])
            mask = (dilated > alpha) & (alpha < 1e-5)
            result[mask, :3] = np_array[mask, :3]
            result[:, :, 3] = dilated
        return result


class RoughnessTool(BaseTool):
    tool_id = 'roughness'
    label = '粗糙度估算'

    @staticmethod
    def get_properties():
        return {
            'rough_input': bpy.props.EnumProperty(
                name="输入类型", items=[
                    ('NORMAL', "法线贴图", "从法线高频变化估算"),
                    ('GRAY', "灰度高度", "从灰度梯度估算"),
                ], default='NORMAL', update=_on_param_update,
            ),
            'rough_radius': bpy.props.IntProperty(
                name="采样半径", default=3, min=1, max=20,
                update=_on_param_update, description="局部方差计算半径",
            ),
            'rough_strength': bpy.props.FloatProperty(
                name="强度", default=1.0, min=0.1, max=3.0,
                soft_min=0.5, soft_max=2.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "rough_input")
        layout.prop(props, "rough_radius")
        layout.prop(props, "rough_strength", slider=True)

    @staticmethod
    def process(np_array, props):
        if props.rough_input == 'NORMAL':
            r = np_array[:, :, 0].astype(np.float32) * 2.0 - 1.0
            g = np_array[:, :, 1].astype(np.float32) * 2.0 - 1.0
            mag = np.sqrt(np.gradient(r)[0] ** 2 + np.gradient(r)[1] ** 2 +
                          np.gradient(g)[0] ** 2 + np.gradient(g)[1] ** 2)
        else:
            gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114
            gx, gy = np.gradient(gray)
            mag = np.sqrt(gx ** 2 + gy ** 2)

        rad = props.rough_radius
        from ..utils.np_img_utils import np_blur_img
        roughness = np_blur_img(mag.astype(np.float32), blur_percent=rad * 2.0, mode='edge')
        roughness = np.clip(roughness * props.rough_strength * 2.0, 0.0, 1.0)
        return _gray_to_rgba(np_array, roughness.astype(np.float32))


def _gray_to_rgba(np_array, gray):
    result = np.zeros_like(np_array)
    result[:, :, 0] = gray
    result[:, :, 1] = gray
    result[:, :, 2] = gray
    result[:, :, 3] = np_array[:, :, 3]
    return result


def _normal_to_height_fast(normal_np):
    """从法线快速近似恢复高度 (梯度积分)"""
    nx = normal_np[:, :, 0].astype(np.float32) * 2.0 - 1.0
    ny = normal_np[:, :, 1].astype(np.float32) * 2.0 - 1.0
    h, w = nx.shape
    height = np.zeros((h, w), dtype=np.float32)
    height[:, 1:] = height[:, :-1] + nx[:, :-1]
    height = height - height.min()
    height = height / (height.max() + 1e-6)
    return height

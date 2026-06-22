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
            'htao_engine': bpy.props.EnumProperty(
                name="引擎",
                description="GPU 着色器(快) 或 CPU numpy",
                items=[('GPU', "GPU", "GPU 着色器"), ('CPU', "CPU", "numpy")],
                default='GPU',
                update=_on_param_update,
            ),
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
        layout.prop(props, "htao_engine")
        layout.separator()
        layout.prop(props, "htao_radius", slider=True)
        layout.prop(props, "htao_intensity", slider=True)
        layout.prop(props, "htao_samples")

    @staticmethod
    def process(np_array, props):
        h, w = np_array.shape[:2]
        radius = max(1, int(min(h, w) * props.htao_radius))
        n_samples = props.htao_samples
        intensity = props.htao_intensity

        if getattr(props, 'htao_engine', 'GPU') == 'GPU':
            try:
                from ..utils.gpu_img_utils import gpu_ao_npimg
                return gpu_ao_npimg(np_array, radius, n_samples, intensity)
            except Exception as e:
                print(f"[高度→AO] GPU 失败，回退 CPU: {e}")

        gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114

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
            'curv_engine': bpy.props.EnumProperty(
                name="引擎",
                description="GPU 着色器(快) 或 CPU numpy",
                items=[('GPU', "GPU", "GPU 着色器"), ('CPU', "CPU", "numpy")],
                default='GPU',
                update=_on_param_update,
            ),
            'curv_input': bpy.props.EnumProperty(
                name="输入类型", items=[
                    ('HEIGHT', "高度图", "从灰度贴图计算"),
                    ('NORMAL', "法线贴图", "从法线贴图计算"),
                ], default='HEIGHT', update=_on_param_update,
            ),
            'curv_radius': bpy.props.IntProperty(
                name="半径", default=3, min=1, max=20,
                update=_on_param_update,
                description="曲率尺度(像素)：求导前对高度预模糊的半径，越大越保留宏观曲率、抹去细节",
            ),
            'curv_intensity': bpy.props.FloatProperty(
                name="强度", default=1.0, min=0.1, max=5.0,
                soft_min=0.5, soft_max=3.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "curv_engine")
        layout.separator()
        layout.prop(props, "curv_input")
        layout.prop(props, "curv_radius")
        layout.prop(props, "curv_intensity", slider=True)

    @staticmethod
    def process(np_array, props):
        if props.curv_input == 'NORMAL':
            gray = _normal_to_height_fast(np_array)
        else:
            gray = np_array[:, :, 0] * 0.299 + np_array[:, :, 1] * 0.587 + np_array[:, :, 2] * 0.114

        intensity = props.curv_intensity
        radius = max(1, int(props.curv_radius))

        if getattr(props, 'curv_engine', 'GPU') == 'GPU':
            try:
                from ..utils.gpu_img_utils import gpu_laplacian_npimg, gpu_blur_npimg
                gray_rgba = np.zeros_like(np_array)
                gray_rgba[:, :, 0] = gray
                gray_rgba[:, :, 1] = gray
                gray_rgba[:, :, 2] = gray
                gray_rgba[:, :, 3] = 1.0
                # 按半径预模糊高度 → 控制曲率尺度（大半径=宏观曲率）
                gray_rgba = gpu_blur_npimg(gray_rgba, radius, blur_type='separable', blur_mode='edge')
                curv = gpu_laplacian_npimg(gray_rgba)[:, :, 0]
                mean_abs = np.mean(np.abs(curv)) + 1e-6
                curv = np.clip(curv / mean_abs * intensity, -1.0, 1.0)
                curv = (curv + 1.0) / 2.0
                return _gray_to_rgba(np_array, curv.astype(np.float32))
            except Exception as e:
                print(f"[曲率图] GPU 失败，回退 CPU: {e}")

        # CPU：按半径(像素)预模糊高度，半径越大曲率越宏观
        import math
        from ..utils.np_img_utils import np_blur_img
        h, w = gray.shape[:2]
        base_len = max(1.0, math.hypot(w, h) / 4.0)
        pct = min(100.0, radius * 100.0 / base_len)
        gray = np_blur_img(gray, pct, 'edge')

        # 5 点拉普拉斯(edge 填充)，与 GPU gpu_laplacian_npimg 一致 → 两引擎结果对齐
        gp = np.pad(gray, 1, mode='edge')
        curv = (gp[1:-1, :-2] + gp[1:-1, 2:] + gp[:-2, 1:-1] + gp[2:, 1:-1] - 4.0 * gray)
        mean_abs = np.mean(np.abs(curv)) + 1e-6
        curv = np.clip(curv / mean_abs * intensity, -1.0, 1.0)
        curv = (curv + 1.0) / 2.0
        return _gray_to_rgba(np_array, curv.astype(np.float32))


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

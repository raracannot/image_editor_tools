import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


def _luminance(rgb):
    return rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114


def _rgb_to_rgba(rgb):
    h, w = rgb.shape[:2]
    return np.dstack([rgb, np.ones((h, w, 1), dtype=np.float32)])


class RelightTool(BaseTool):
    tool_id = 'relight'
    label = '均匀光影'

    @staticmethod
    def get_properties():
        return {
            'relight_engine': bpy.props.EnumProperty(
                name="引擎",
                description="GPU 着色器(快) 或 CPU numpy",
                items=[('GPU', "GPU", "GPU 着色器"), ('CPU', "CPU", "numpy")],
                default='GPU',
                update=_on_param_update,
            ),
            'relight_highpass_sigma': bpy.props.FloatProperty(
                name="高反差半径",
                description="控制分离光影的尺度，值越大去除越粗的光影",
                default=10.0, min=0.1, max=100.0,
                soft_min=1.0, soft_max=50.0,
                update=_on_param_update,
            ),
            'relight_base_mode': bpy.props.EnumProperty(
                name="基色模式",
                description="基础光影层的生成方式",
                items=[
                    ('AVERAGE', "平均色", "整图平均色作为均匀基色"),
                    ('BLUR', "强模糊", "大幅模糊保留宏观光影渐变（更快更自然）"),
                    ('CUSTOM', "自定义", "手动指定基色"),
                ],
                default='AVERAGE',
                update=_on_param_update,
            ),
            'relight_blur_radius': bpy.props.FloatProperty(
                name="模糊半径",
                description="基色层模糊强度（强模糊模式）",
                default=20.0, min=0.5, max=100.0,
                soft_min=2.0, soft_max=50.0,
                subtype='PERCENTAGE',
                update=_on_param_update,
            ),
            'relight_base_color': bpy.props.FloatVectorProperty(
                name="基色", subtype='COLOR', size=3,
                default=(0.5, 0.5, 0.5), min=0.0, max=1.0,
                update=_on_param_update,
            ),
            'relight_replacement': bpy.props.EnumProperty(
                name="替换范围",
                description="基色层替换的颜色分量",
                items=[
                    ('FULL', "全部 (L+a+b)", "明度+色度全部替换"),
                    ('CHROMA', "仅色度 (a+b)", "替换色相/饱和度，保留原始明度分布"),
                    ('LUMA', "仅明度 (L)", "替换明度，保留原始色相/饱和度"),
                ],
                default='FULL',
                update=_on_param_update,
            ),
            'relight_blend': bpy.props.FloatProperty(
                name="混合强度", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            ),
            'relight_preserve_lum': bpy.props.BoolProperty(
                name="保持原始明度",
                description="用原始亮度替换结果亮度，避免暗部裁切损失",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "relight_engine")
        layout.separator()
        layout.prop(props, "relight_highpass_sigma", text="高反差半径", slider=True)
        layout.prop(props, "relight_base_mode", text="基色模式")
        if props.relight_base_mode == 'BLUR':
            layout.prop(props, "relight_blur_radius", text="模糊半径", slider=True)
        elif props.relight_base_mode == 'CUSTOM':
            layout.prop(props, "relight_base_color", text="基色")
        layout.prop(props, "relight_replacement", text="替换范围")
        layout.prop(props, "relight_blend", text="混合强度", slider=True)
        layout.prop(props, "relight_preserve_lum", text="保持原始明度")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_gaussian_filter, np_blur_img, np_resize_img
        from ..utils.np_img_utils import np_rgb_to_lab, np_lab_to_rgb

        rgb = np_array[:, :, :3].astype(np.float32)
        alpha = np_array[:, :, 3].copy()
        h, w = rgb.shape[:2]

        sigma = max(0.1, props.relight_highpass_sigma)
        if getattr(props, 'relight_engine', 'GPU') == 'GPU':
            try:
                from ..utils.gpu_img_utils import gpu_gaussian_npimg
                blurred = gpu_gaussian_npimg(_rgb_to_rgba(rgb), sigma, 'edge')[:, :, :3]
            except Exception as e:
                print(f"[均匀光影] GPU 失败，回退 CPU: {e}")
                blurred = np_gaussian_filter(rgb, sigma)
        else:
            blurred = np_gaussian_filter(rgb, sigma)
        detail = rgb - blurred

        base_mode = props.relight_base_mode
        if base_mode == 'AVERAGE':
            avg_color = np.mean(rgb, axis=(0, 1))
            base = np.full((h, w, 3), avg_color, dtype=np.float32)
        elif base_mode == 'BLUR':
            radius = max(0.5, props.relight_blur_radius)
            hw, hh = max(1, w // 2), max(1, h // 2)
            small = np_resize_img(_rgb_to_rgba(rgb), hw, hh)[:, :, :3]
            small = np_blur_img(_rgb_to_rgba(small), radius, 'edge')[:, :, :3]
            base = np_resize_img(_rgb_to_rgba(small), w, h)[:, :, :3]
        else:
            c = np.asarray(props.relight_base_color, dtype=np.float32)
            base = np.full((h, w, 3), c, dtype=np.float32)

        result = base + detail
        result = np.clip(result, 0.0, 1.0)

        repl = props.relight_replacement
        if repl != 'FULL':
            orig_lab = np_rgb_to_lab(_rgb_to_rgba(rgb))
            res_lab = np_rgb_to_lab(_rgb_to_rgba(result))
            if repl == 'CHROMA':
                combined = np.stack([orig_lab[..., 0], res_lab[..., 1], res_lab[..., 2]], axis=-1)
            else:
                combined = np.stack([res_lab[..., 0], orig_lab[..., 1], orig_lab[..., 2]], axis=-1)
            result = np_lab_to_rgb(combined)

        if props.relight_preserve_lum:
            orig_lum = _luminance(rgb)
            res_lum = _luminance(result)
            scale = orig_lum / np.maximum(res_lum, 1e-6)
            result = np.clip(result * scale[..., np.newaxis], 0.0, 1.0)

        blend = props.relight_blend
        if blend < 1.0:
            result = rgb * (1.0 - blend) + result * blend

        output = np.zeros_like(np_array)
        output[:, :, :3] = np.clip(result, 0.0, 1.0)
        output[:, :, 3] = alpha
        return output

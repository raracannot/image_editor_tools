import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class NoiseTool(BaseTool):
    tool_id = 'noise'
    label = '噪点'

    @staticmethod
    def get_properties():
        return {
            'noise_type': bpy.props.EnumProperty(
                name="噪点类型",
                description="噪点生成算法",
                items=[
                    ('GAUSSIAN', "高斯噪点", "正态分布，最自然的随机噪点"),
                    ('UNIFORM', "均匀噪点", "平坦分布，颗粒感更强更硬"),
                    ('SPECKLE', "斑点噪点", "乘积噪点，暗部干净亮部粗糙"),
                    ('SALT_PEPPER', "椒盐噪点", "随机散布纯黑/纯白像素"),
                ],
                default='GAUSSIAN',
                update=_on_param_update,
            ),
            'noise_intensity': bpy.props.FloatProperty(
                name="强度",
                description="噪点强度",
                default=0.05, min=0.0, max=0.5,
                soft_min=0.0, soft_max=0.15, subtype='FACTOR',
                update=_on_param_update,
            ),
            'noise_color': bpy.props.BoolProperty(
                name="彩色噪点",
                description="RGB 通道独立加噪，产生彩色噪点效果",
                default=False,
                update=_on_param_update,
            ),
            'noise_seed': bpy.props.IntProperty(
                name="随机种子",
                description="固定随机种子以获得可复现的噪点图案 (0=每次随机)",
                default=0, min=0, max=1000,
                update=_on_param_update,
            ),
            'noise_alpha': bpy.props.BoolProperty(
                name="影响 Alpha",
                description="噪点是否同时影响 Alpha 通道",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "noise_type")
        layout.prop(props, "noise_intensity", text="强度", slider=True)
        layout.prop(props, "noise_color", text="彩色噪点")
        layout.prop(props, "noise_seed", text="随机种子")
        layout.prop(props, "noise_alpha", text="影响 Alpha")

    @staticmethod
    def process(np_array, props):
        h, w = np_array.shape[:2]
        rgb = np_array[:, :, :3].astype(np.float32, copy=False)
        alpha_ch = np_array[:, :, 3].copy()
        amount = props.noise_intensity
        color_noise = props.noise_color
        ntype = props.noise_type
        seed_val = props.noise_seed
        rng = np.random.RandomState(seed_val) if seed_val > 0 else np.random

        noise_1ch = np.zeros((h, w), dtype=np.float32)

        if ntype == 'SALT_PEPPER':
            mask = rng.rand(h, w) < amount
            signs = np.where(mask, rng.choice([-1.0, 1.0], (h, w)), 0.0).astype(np.float32)
            if color_noise:
                noise_r = signs.copy()
                noise_g = np.where(mask, rng.choice([-1.0, 1.0], (h, w)), 0.0).astype(np.float32)
                noise_b = np.where(mask, rng.choice([-1.0, 1.0], (h, w)), 0.0).astype(np.float32)
                result_rgb = np.clip(rgb + np.stack([noise_r, noise_g, noise_b], axis=-1), 0.0, 1.0)
            else:
                lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
                noisy_lum = np.clip(lum + signs, 0.0, 1.0)
                result_rgb = np.clip(rgb + (noisy_lum - lum)[:, :, np.newaxis], 0.0, 1.0)
            noise_1ch = signs
        elif ntype == 'SPECKLE':
            if color_noise:
                noise_3ch = rng.normal(0, amount, (h, w, 3)).astype(np.float32)
                result_rgb = np.clip(rgb * (1.0 + noise_3ch), 0.0, 1.0)
            else:
                noise_1ch = rng.normal(0, amount, (h, w)).astype(np.float32)
                result_rgb = np.clip(rgb * (1.0 + noise_1ch[:, :, np.newaxis]), 0.0, 1.0)
        else:
            if color_noise:
                noise_3ch = _gen_noise_3ch(h, w, ntype, amount, rng)
                result_rgb = np.clip(rgb + noise_3ch, 0.0, 1.0)
            else:
                noise_1ch = _gen_noise_1ch(h, w, ntype, amount, rng)
                lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
                noisy_lum = np.clip(lum + noise_1ch, 0.0, 1.0)
                result_rgb = np.clip(rgb + (noisy_lum - lum)[:, :, np.newaxis], 0.0, 1.0)

        result = np.zeros_like(np_array)
        result[:, :, :3] = result_rgb
        result[:, :, 3] = np.clip(alpha_ch + noise_1ch, 0.0, 1.0) if props.noise_alpha else alpha_ch
        return result


def _gen_noise_1ch(h, w, ntype, amount, rng):
    if ntype == 'GAUSSIAN':
        return rng.normal(0, amount, (h, w)).astype(np.float32)
    return rng.uniform(-amount, amount, (h, w)).astype(np.float32)


def _gen_noise_3ch(h, w, ntype, amount, rng):
    if ntype == 'GAUSSIAN':
        return rng.normal(0, amount, (h, w, 3)).astype(np.float32)
    return rng.uniform(-amount, amount, (h, w, 3)).astype(np.float32)

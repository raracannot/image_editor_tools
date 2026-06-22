import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


_STYLES = {
    'BW': {'bg': (0, 0, 0), 'dot': (1, 1, 1), 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': True},
    'WB': {'bg': (1, 1, 1), 'dot': (0, 0, 0), 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': False},
    'CW': {'bg': 'color', 'dot': (1, 1, 1), 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': True},
    'CB': {'bg': 'color', 'dot': (0, 0, 0), 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': False},
    'WC': {'bg': (1, 1, 1), 'dot': 'color', 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': True},
    'BC': {'bg': (0, 0, 0), 'dot': 'color', 'bg_a': 1.0, 'dot_a': 1.0, 'dark_large': False},
    'TC': {'bg': (0, 0, 0), 'dot': 'color', 'bg_a': 0.0, 'dot_a': 1.0, 'dark_large': True},
    'CT': {'bg': 'color', 'dot': (0, 0, 0), 'bg_a': 1.0, 'dot_a': 0.0, 'dark_large': False},
}


class HalftoneTool(BaseTool):
    tool_id = 'halftone'
    label = '色彩半调'

    @staticmethod
    def get_properties():
        return {
            'halftone_radius': bpy.props.IntProperty(
                name="网点半径",
                description="最大网点半径",
                default=8, min=2, max=64, soft_min=3, soft_max=24,
                update=_on_param_update,
            ),
            'halftone_angle': bpy.props.FloatProperty(
                name="角度", description="网屏角度",
                default=45.0, min=0.0, max=180.0, soft_min=0.0, soft_max=180.0,
                update=_on_param_update,
            ),
            'halftone_aa': bpy.props.FloatProperty(
                name="抗锯齿", description="边缘平滑过渡的像素宽度，0为无抗锯齿",
                default=1.5, min=0.0, max=5.0, soft_min=0.0, soft_max=3.0,
                update=_on_param_update,
            ),
            'halftone_style': bpy.props.EnumProperty(
                name="样式",
                description="网点与底色的组合方式",
                items=[
                    ('BW', "黑底白点", "暗处大点"),
                    ('WB', "白底黑点", "亮处大点"),
                    ('CW', "彩底白点", "原图为底，暗处叠加白色网点"),
                    ('CB', "彩底黑点", "原图为底，亮处叠加黑色网点"),
                    ('WC', "白底彩点", "白底上以原图色画点，暗处大点"),
                    ('BC', "黑底彩点", "黑底上以原图色画点，亮处大点"),
                    ('TC', "透明底彩点", "透明底上以原图色画点，暗处大点"),
                    ('CT', "彩底透明点", "原图为底，亮处挖透明孔"),
                ],
                default='BW',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "halftone_radius", text="网点半径", slider=True)
        layout.prop(props, "halftone_angle", text="角度", slider=True)
        layout.prop(props, "halftone_aa", text="抗锯齿", slider=True)
        layout.prop(props, "halftone_style", text="样式")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_rgb_to_gray

        h, w = np_array.shape[:2]
        gray = np_rgb_to_gray(np_array)
        radius = float(props.halftone_radius)
        angle = props.halftone_angle
        aa = props.halftone_aa
        style = props.halftone_style

        spacing = max(2.0, radius * 2.0)
        rad = np.radians(angle)
        ca, sa = np.cos(rad), np.sin(rad)

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)

        Xr = X * ca + Y * sa
        Yr = -X * sa + Y * ca

        Xc_rot = np.round(Xr / spacing) * spacing
        Yc_rot = np.round(Yr / spacing) * spacing

        Xc = Xc_rot * ca - Yc_rot * sa
        Yc = Xc_rot * sa + Yc_rot * ca

        Xc_idx = np.clip(np.round(Xc).astype(np.int32), 0, w - 1)
        Yc_idx = np.clip(np.round(Yc).astype(np.int32), 0, h - 1)
        gray_center = gray[Yc_idx, Xc_idx]
        rgb_center = np_array[Yc_idx, Xc_idx, :3]

        s = _STYLES[style]

        if s['dark_large']:
            dot_r = radius * (1.0 - gray_center)
        else:
            dot_r = radius * gray_center

        dist = np.sqrt((X - Xc) ** 2 + (Y - Yc) ** 2)

        if aa > 0.001:
            t = np.clip((dist - dot_r) / aa + 0.5, 0.0, 1.0)
            t = t * t * (3.0 - 2.0 * t)
            mask_dot = 1.0 - t
        else:
            mask_dot = np.where(dist <= dot_r, 1.0, 0.0).astype(np.float32)

        bg_rgb = rgb_center if s['bg'] == 'color' else np.array(s['bg'], dtype=np.float32)
        dot_rgb = rgb_center if s['dot'] == 'color' else np.array(s['dot'], dtype=np.float32)

        mask3 = mask_dot[..., np.newaxis]
        bg_a = np.float32(s['bg_a'])
        dot_a = np.float32(s['dot_a'])

        result = np.zeros_like(np_array)
        result[:, :, :3] = bg_rgb * (1.0 - mask3) + dot_rgb * mask3
        result[:, :, 3] = bg_a * (1.0 - mask_dot) + dot_a * mask_dot
        return result

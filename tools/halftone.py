import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class HalftoneTool(BaseTool):
    tool_id = 'halftone'
    label = '色彩半调'

    @staticmethod
    def get_properties():
        return {
            'halftone_radius': bpy.props.IntProperty(
                name="网点半径",
                description="最大网点半径",
                default=8,
                min=2,
                max=64,
                soft_min=3,
                soft_max=24,
                update=_on_param_update,
            ),
            'halftone_angle': bpy.props.FloatProperty(
                name="角度",
                description="网屏角度",
                default=45.0,
                min=0.0,
                max=180.0,
                soft_min=0.0,
                soft_max=180.0,
                subtype='ANGLE',
                update=_on_param_update,
            ),
            'halftone_aa': bpy.props.FloatProperty(
                name="抗锯齿",
                description="边缘平滑过渡的像素宽度，0为无抗锯齿",
                default=1.5,
                min=0.0,
                max=5.0,
                soft_min=0.0,
                soft_max=3.0,
                update=_on_param_update,
            ),
            'halftone_invert': bpy.props.BoolProperty(
                name="反相",
                description="白底黑点 ↔ 黑底白点",
                default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "halftone_radius", text="网点半径", slider=True)
        layout.prop(props, "halftone_angle", text="角度", slider=True)
        layout.prop(props, "halftone_aa", text="抗锯齿", slider=True)
        layout.prop(props, "halftone_invert", text="反相")

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_rgb_to_gray

        h, w = np_array.shape[:2]
        gray = np_rgb_to_gray(np_array)
        radius = float(props.halftone_radius)
        angle = props.halftone_angle
        aa = props.halftone_aa
        invert = props.halftone_invert

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

        dot_r = radius * (1.0 - gray_center) if not invert else radius * gray_center
        dist = np.sqrt((X - Xc) ** 2 + (Y - Yc) ** 2)

        if aa > 0.001:
            # 计算平滑过渡因子 (0.0 到 1.0)
            t = np.clip((dist - dot_r) / aa + 0.5, 0.0, 1.0)
            # Smoothstep 平滑处理
            t = t * t * (3.0 - 2.0 * t)
            dot = t if not invert else 1.0 - t
        else:
            # 无抗锯齿的硬边缘
            dot = np.where(dist <= dot_r, 0.0 if not invert else 1.0, 1.0 if not invert else 0.0)


        result = np.zeros_like(np_array)
        result[:, :, 0] = dot
        result[:, :, 1] = dot
        result[:, :, 2] = dot
        result[:, :, 3] = np_array[:, :, 3]
        return result

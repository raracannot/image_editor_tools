import bpy
import numpy as np
from ..np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
from .base import BaseTool
from . import _on_param_update

COLORS = [
    ('red', "红色", 0, 18, 'NODE_SOCKET_MATERIAL'),
    ('orange', "橙色", 30, 18, 'NODE_SOCKET_OBJECT'),
    ('yellow', "黄色", 60, 18, 'NODE_SOCKET_RGBA'),
    ('green', "绿色", 120, 30, 'NODE_SOCKET_SHADER'),
    ('cyan', "浅绿", 180, 18, 'NODE_SOCKET_GEOMETRY'),
    ('blue', "蓝色", 240, 18, 'NODE_SOCKET_STRING'),
    ('purple', "紫色", 270, 18, 'NODE_SOCKET_ROTATION'),
    ('magenta', "洋红", 300, 18, 'NODE_SOCKET_MATRIX'),
]



def _hue_mask(h, center, half_range, feather_deg):
    delta = (h - center / 360.0 + 0.5) % 1.0 - 0.5
    dist = np.abs(delta)
    half = half_range / 360.0
    feather = feather_deg / 360.0
    
    # 1. 计算基础的线性遮罩 (0.0 到 1.0)
    linear_mask = np.clip((half + feather - dist) / max(feather, 1e-4), 0.0, 1.0)
    
    # 2. 使用 Smoothstep 函数进行平滑处理
    smooth_mask = linear_mask * linear_mask * (3.0 - 2.0 * linear_mask)
    
    return smooth_mask


class MixerTool(BaseTool):
    tool_id = 'mixer'
    label = '混色器'

    @staticmethod
    def get_properties():
        props = {}
        # 新增全局羽化属性
        props["mixer_feather"] = bpy.props.FloatProperty(
            name="全局羽化", default=30.0, min=0.0, max=120.0,
            soft_min=0.0, soft_max=90.0,
            update=_on_param_update,
        )
        
        for key, label, center, half_r, icon_name in COLORS:
            props[f"mixer_hue_{key}"] = bpy.props.FloatProperty(
                name=f"{label}色相", default=0.0, min=-1.0, max=1.0,
                soft_min=-1.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            )
            props[f"mixer_sat_{key}"] = bpy.props.FloatProperty(
                name=f"{label}饱和度", default=0.0, min=-1.0, max=1.0,
                soft_min=-1.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            )
            props[f"mixer_val_{key}"] = bpy.props.FloatProperty(
                name=f"{label}明度", default=0.0, min=-1.0, max=1.0,
                soft_min=-1.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
            )
        return props

    @staticmethod
    def draw_panel(layout, props):
        header = layout.row(align=True)
        header.label(text="")
        header.label(text="色相")
        header.label(text="饱和度")
        header.label(text="明度")

        for key, label, center, half_r, icon_name in COLORS:
            row = layout.row(align=True)
            row.label(text=label, icon=icon_name)
            row.prop(props, f"mixer_hue_{key}", text="")
            row.prop(props, f"mixer_sat_{key}", text="")
            row.prop(props, f"mixer_val_{key}", text="")
        
        layout.separator()
        layout.prop(props, "mixer_feather")
        

    @staticmethod
    def process(np_array, props):
        rgb = np_array[:, :, :3]
        alpha = np_array[:, :, 3]

        h, s, v = np_rgb_to_hsv(rgb)

        h_adjust = np.zeros_like(h)
        s_adjust = np.ones_like(s)
        v_adjust = np.ones_like(v)
        
        feather_val = getattr(props, "mixer_feather", 30.0)
        
        for key, label, center, half_r, icon_name in COLORS:
            hue_val = getattr(props, f"mixer_hue_{key}", 0)
            sat_val = getattr(props, f"mixer_sat_{key}", 0)
            val_val = getattr(props, f"mixer_val_{key}", 0)

            if hue_val == 0.0 and sat_val == 0.0 and val_val == 0.0:
                continue

            mask = _hue_mask(h, center, half_r, feather_val)

            if hue_val != 0.0:
                h_adjust += mask * hue_val * 0.5
            if sat_val != 0.0:
                s_adjust *= 1.0 + mask * sat_val
            if val_val != 0.0:
                v_adjust *= 1.0 + mask * val_val

        h = (h + h_adjust) % 1.0
        s = np.clip(s * s_adjust, 0.0, 1.0)
        v = np.clip(v * v_adjust, 0.0, 1.0)

        result = np.zeros_like(np_array)
        result[:, :, :3] = np_hsv_to_rgb(h, s, v)
        result[:, :, 3] = alpha
        return result

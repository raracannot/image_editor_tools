import bpy
import numpy as np
from ..utils.np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
from .base import BaseTool
from . import _on_param_update


def _color_balance(rgb, cr, mg, yb, preserve_lum):
    lum = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
    shadows = np.clip(1.0 - lum * 2.0, 0.0, 1.0)
    highlights = np.clip((lum - 0.5) * 2.0, 0.0, 1.0)
    midtones = 1.0 - shadows - highlights

    shadow_adj = np.array([cr, mg, yb], dtype=np.float32) * 0.02
    midtone_adj = np.array([cr, mg, yb], dtype=np.float32) * 0.01
    highlight_adj = np.array([cr, mg, yb], dtype=np.float32) * 0.02

    adj = (shadows[..., np.newaxis] * shadow_adj +
           midtones[..., np.newaxis] * midtone_adj +
           highlights[..., np.newaxis] * highlight_adj)

    result = rgb + adj
    if preserve_lum:
        result_lum = result[..., 0] * 0.299 + result[..., 1] * 0.587 + result[..., 2] * 0.114
        result = result * (lum / (result_lum + 1e-6))[..., np.newaxis]
    return np.clip(result, 0.0, 1.0)


class ColorTool(BaseTool):
    tool_id = 'color'
    label = '调色'

    @staticmethod
    def get_properties():
        return {
            'color_exposure': bpy.props.FloatProperty(
                name="曝光度", default=0.0, min=-5.0, max=5.0,
                soft_min=-3.0, soft_max=3.0, update=_on_param_update,
            ),
            'color_offset': bpy.props.FloatProperty(
                name="偏移", default=0.0, min=-0.5, max=0.5,
                soft_min=-0.1, soft_max=0.1, update=_on_param_update,
            ),
            'color_exposure_gamma': bpy.props.FloatProperty(
                name="灰度系数", default=1.0, min=0.1, max=9.99,
                soft_min=0.5, soft_max=2.0, update=_on_param_update,
            ),
            'color_balance_cr': bpy.props.IntProperty(
                name="青色-红色", default=0, min=-100, max=100,
                soft_min=-50, soft_max=50, update=_on_param_update,
            ),
            'color_balance_mg': bpy.props.IntProperty(
                name="洋红-绿色", default=0, min=-100, max=100,
                soft_min=-50, soft_max=50, update=_on_param_update,
            ),
            'color_balance_yb': bpy.props.IntProperty(
                name="黄色-蓝色", default=0, min=-100, max=100,
                soft_min=-50, soft_max=50, update=_on_param_update,
            ),
            'color_balance_preserve_lum': bpy.props.BoolProperty(
                name="保持明度", default=True, update=_on_param_update,
            ),
            'color_hue': bpy.props.FloatProperty(
                name="色相", default=0.0, min=-180.0, max=180.0,
                soft_min=-180.0, soft_max=180.0, update=_on_param_update,
            ),
            'color_hsl_saturation': bpy.props.IntProperty(
                name="饱和度", default=0, min=-100, max=100,
                soft_min=-100, soft_max=100, update=_on_param_update,
            ),
            'color_lightness': bpy.props.IntProperty(
                name="明度", default=0, min=-100, max=100,
                soft_min=-100, soft_max=100, update=_on_param_update,
            ),
            'color_vibrance': bpy.props.IntProperty(
                name="自然饱和度", default=0, min=-100, max=100,
                soft_min=-100, soft_max=100, update=_on_param_update,
            ),
            'color_contrast': bpy.props.FloatProperty(
                name="对比度", default=0.0, min=-0.5, max=0.5,
                soft_min=-0.25, soft_max=0.25, update=_on_param_update,
            ),
            'color_saturation': bpy.props.FloatProperty(
                name="饱和度", default=0.0, min=-0.5, max=0.5,
                soft_min=-0.25, soft_max=0.25, update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        box = layout.box()
        box.label(text="曝光度", icon='EXPERIMENTAL')
        col = box.column(align=True)
        col.prop(props, "color_exposure", text="曝光", slider=True)
        col.prop(props, "color_offset", text="偏移", slider=True)
        col.prop(props, "color_exposure_gamma", text="灰度系数", slider=True)

        box = layout.box()
        box.label(text="色彩平衡", icon='COLOR')
        col = box.column(align=True)
        col.prop(props, "color_balance_cr", text="青色-红色", slider=True)
        col.prop(props, "color_balance_mg", text="洋红-绿色", slider=True)
        col.prop(props, "color_balance_yb", text="黄色-蓝色", slider=True)
        col.prop(props, "color_balance_preserve_lum", text="保持明度")

        box = layout.box()
        box.label(text="色相 / 饱和度", icon='COLOR')
        col = box.column(align=True)
        col.prop(props, "color_hue", text="色相", slider=True)
        col.prop(props, "color_hsl_saturation", text="饱和度", slider=True)
        col.prop(props, "color_lightness", text="明度", slider=True)

        box = layout.box()
        box.label(text="增强", icon='COLOR')
        col = box.column(align=True)
        col.prop(props, "color_vibrance", text="自然饱和度", slider=True)
        col.prop(props, "color_contrast", text="对比度", slider=True)
        col.prop(props, "color_saturation", text="饱和度", slider=True)

    @staticmethod
    def process(np_array, props):
        rgb = np_array[:, :, :3]
        alpha = np_array[:, :, 3]
        exp = props.color_exposure
        offset = props.color_offset
        gamma_val = props.color_exposure_gamma
        cr = props.color_balance_cr
        mg = props.color_balance_mg
        yb = props.color_balance_yb
        preserve_lum = props.color_balance_preserve_lum
        hue = props.color_hue
        hsl_sat = props.color_hsl_saturation
        lightness = props.color_lightness
        vibrance = props.color_vibrance
        contrast = props.color_contrast
        saturation = props.color_saturation

        if exp != 0.0 or offset != 0.0 or gamma_val != 1.0:
            rgb = rgb * (2.0 ** exp) + offset
            if gamma_val != 1.0:
                rgb = np.power(np.clip(rgb, 0.0, 1.0), 1.0 / gamma_val)

        if cr != 0 or mg != 0 or yb != 0:
            rgb = _color_balance(rgb, cr, mg, yb, preserve_lum)

        if hue != 0.0 or hsl_sat != 0 or lightness != 0:
            h, s, v = np_rgb_to_hsv(rgb)
            h = (h + hue / 360.0) % 1.0
            s = np.clip(s * (1.0 + hsl_sat / 100.0), 0.0, 1.0)
            v = np.clip(v * (1.0 + lightness / 100.0), 0.0, 1.0)
            rgb = np_hsv_to_rgb(h, s, v)

        if vibrance != 0:
            gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
            max_c = np.max(rgb, axis=-1)
            min_c = np.min(rgb, axis=-1)
            sat = np.where(max_c > 1e-6, (max_c - min_c) / np.maximum(max_c, 1e-10), 0.0)
            boost = (vibrance / 100.0) * (1.0 - sat)
            rgb = rgb + (rgb - gray[..., np.newaxis]) * boost[..., np.newaxis]

        if contrast != 0.0:
            rgb = (rgb - 0.5) * (1.0 + contrast * 2.0) + 0.5

        if saturation != 0.0:
            gray = rgb[..., 0] * 0.299 + rgb[..., 1] * 0.587 + rgb[..., 2] * 0.114
            rgb = gray[..., np.newaxis] + (rgb - gray[..., np.newaxis]) * (1.0 + saturation * 2.0)

        result = np.zeros_like(np_array)
        result[:, :, :3] = np.clip(rgb, 0.0, 1.0)
        result[:, :, 3] = alpha
        return result

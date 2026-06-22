import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update
from ..utils.blend_modes import BLEND_MODE_ITEMS


class CompositeTool(BaseTool):
    tool_id = 'composite'
    label = '图像合成'

    _cache_key = None
    _cache_composited = None

    @staticmethod
    def get_properties():
        return {
            'composite_fg': bpy.props.PointerProperty(
                name="前景图",
                description="用于叠加合成的前景图像",
                type=bpy.types.Image,
                update=_on_param_update,
            ),
            'composite_mode': bpy.props.EnumProperty(
                name="模式",
                description="混合模式",
                items=BLEND_MODE_ITEMS,
                default='MIX',
                update=_on_param_update,
            ),
            'composite_opacity': bpy.props.FloatProperty(
                name="不透明度",
                description="前景不透明度",
                default=1.0,
                min=0.0,
                max=1.0,
                soft_min=0.0,
                soft_max=1.0,
                subtype='FACTOR',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        row = layout.row(align=True)
        row.prop(props, "composite_fg", text="前景图")
        op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
        op.target_prop = "composite_fg"
        if props.composite_fg is None:
            layout.label(text="请选择一幅前景图", icon='INFO')
        else:
            layout.prop(props, "composite_mode", text="模式")
            layout.prop(props, "composite_opacity", text="不透明度", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..utils.blend_modes import apply_blend_mode
        fg_img = props.composite_fg
        if fg_img is None:
            CompositeTool._cache_key = None
            CompositeTool._cache_composited = None
            return np_array

        opacity = props.composite_opacity
        if opacity <= 0.0:
            return np_array

        h, w = np_array.shape[:2]
        fg_key = (fg_img.name, fg_img.size[0], fg_img.size[1])
        bg_key = (
            h, w,
            float(np_array[0, 0, 0]),
            float(np_array[min(h - 1, h // 2), min(w - 1, w // 2), 1]),
            float(np_array[-1, -1, 2]),
        )
        mode = props.composite_mode
        cache_key = (fg_key, bg_key, mode)

        bg_rgb = np_array[:, :, :3]
        bg_a = np_array[:, :, 3]

        if CompositeTool._cache_key != cache_key or CompositeTool._cache_composited is None:
            from ..utils.np_img_utils import blimg_2_npimg, np_resize_img
            fg_np = blimg_2_npimg(fg_img)
            fh, fw = fg_np.shape[:2]
            if fh != h or fw != w:
                fg_np = np_resize_img(fg_np, w, h)

            fg_rgb = fg_np[:, :, :3]
            fg_a = fg_np[:, :, 3]

            blended = apply_blend_mode(bg_rgb, fg_rgb, mode)
            CompositeTool._cache_key = cache_key
            CompositeTool._cache_composited = (blended, fg_a)

        blended, fg_a = CompositeTool._cache_composited
        alpha = fg_a * opacity
        out_alpha = alpha + bg_a * (1.0 - alpha)
        out_alpha_safe = np.where(out_alpha < 1e-5, 1.0, out_alpha)
        out_rgb = (blended * alpha[..., np.newaxis] + bg_rgb * bg_a[..., np.newaxis] * (1.0 - alpha)[..., np.newaxis]) / out_alpha_safe[..., np.newaxis]

        result = np.zeros_like(np_array)
        result[:, :, :3] = out_rgb
        result[:, :, 3] = np.where(out_alpha > 0, out_alpha, 0)
        return result

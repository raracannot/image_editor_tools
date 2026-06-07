import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


_CHANNELS = ['R', 'G', 'B']


class ChannelMixerTool(BaseTool):
    tool_id = 'channel_mixer'
    label = '通道混合器'

    @staticmethod
    def get_properties():
        props = {
            'cm_monochrome': bpy.props.BoolProperty(
                name="单色", description="输出灰度图",
                default=False, update=_on_param_update,
            ),
        }
        defaults = {'R': (100, 0, 0), 'G': (0, 100, 0), 'B': (0, 0, 100)}
        for out_ch in _CHANNELS:
            for i, src_ch in enumerate(_CHANNELS):
                props[f'cm_{out_ch}_{src_ch}'] = bpy.props.IntProperty(
                    name=f"{src_ch}→{out_ch}", default=defaults[out_ch][i],
                    min=-200, max=200, soft_min=-200, soft_max=200,
                    update=_on_param_update,
                )
            props[f'cm_{out_ch}_constant'] = bpy.props.IntProperty(
                name=f"常量({out_ch})", default=0,
                min=-200, max=200, soft_min=-200, soft_max=200,
                update=_on_param_update,
            )
        return props

    @staticmethod
    def draw_panel(layout, props):
        row = layout.row()
        row.prop(props, "cm_monochrome", text="单色")

        if props.cm_monochrome:
            row = layout.row(align=True)
            row.label(text="")
            row.label(text="R"); row.label(text="G"); row.label(text="B")
            row.label(text="常数")
            for src_ch in _CHANNELS:
                row = layout.row(align=True)
                row.label(text=src_ch)
                row.prop(props, f"cm_R_{src_ch}", text="")
                row.prop(props, f"cm_G_{src_ch}", text="")
                row.prop(props, f"cm_B_{src_ch}", text="")
                row.prop(props, f"cm_{src_ch}_constant", text="")
        else:
            for out_ch in _CHANNELS:
                row = layout.row(align=True)
                row.label(text="")
                row.label(text="R"); row.label(text="G"); row.label(text="B")
                row.label(text="常数")
                row = layout.row(align=True)
                row.label(text=f"输出{out_ch}")
                for src_ch in _CHANNELS:
                    row.prop(props, f"cm_{out_ch}_{src_ch}", text="")
                row.prop(props, f"cm_{out_ch}_constant", text="")

    @staticmethod
    def process(np_array, props):
        rgb = np_array[..., :3].astype(np.float32)
        alpha = np_array[..., 3:4]

        if props.cm_monochrome:
            result = np.zeros_like(rgb[..., :1])
            for i, src_ch in enumerate(_CHANNELS):
                w = getattr(props, f'cm_R_{src_ch}') / 100.0
                result[:, :, 0] += rgb[:, :, i] * w
            c = props.cm_R_constant / 100.0 + props.cm_G_constant / 100.0 + props.cm_B_constant / 100.0
            result[:, :, 0] += c / 3.0
            result = np.repeat(np.clip(result, 0, 1), 3, axis=-1)
        else:
            result = np.zeros_like(rgb)
            for oi, out_ch in enumerate(_CHANNELS):
                for si, src_ch in enumerate(_CHANNELS):
                    w = getattr(props, f'cm_{out_ch}_{src_ch}') / 100.0
                    result[:, :, oi] += rgb[:, :, si] * w
                c = getattr(props, f'cm_{out_ch}_constant') / 100.0
                result[:, :, oi] += c
            result = np.clip(result, 0, 1)

        return np.concatenate([result, alpha], axis=-1)

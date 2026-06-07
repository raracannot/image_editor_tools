import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class EmbossTool(BaseTool):
    tool_id = 'emboss'
    label = '浮雕'

    @staticmethod
    def get_properties():
        return {
            'emboss_amount': bpy.props.FloatProperty(
                name="强度",
                description="浮雕凹凸强度",
                default=1.0,
                min=0.1,
                max=5.0,
                soft_min=0.5,
                soft_max=3.0,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "emboss_amount", text="强度", slider=True)

    @staticmethod
    def process(np_array, props):
        amount = props.emboss_amount

        gray = (
            np_array[:, :, 0] * 0.299
            + np_array[:, :, 1] * 0.587
            + np_array[:, :, 2] * 0.114
        )

        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
        gray_pad = np.pad(gray, 1, mode='edge')
        windows = np.lib.stride_tricks.sliding_window_view(gray_pad, (3, 3))
        emboss = np.sum(windows * kernel, axis=(2, 3))
        emboss = np.clip(emboss * amount / 4.0 + 0.5, 0.0, 1.0)

        result = np.zeros_like(np_array)
        result[:, :, 0] = emboss
        result[:, :, 1] = emboss
        result[:, :, 2] = emboss
        result[:, :, 3] = np_array[:, :, 3]
        return result

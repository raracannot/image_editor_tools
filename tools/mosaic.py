import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class MosaicTool(BaseTool):
    tool_id = 'mosaic'
    label = '马赛克'

    @staticmethod
    def get_properties():
        return {
            'mosaic_cell_size': bpy.props.IntProperty(
                name="格子大小",
                description="像素块尺寸",
                default=8,
                min=2,
                max=128,
                soft_min=3,
                soft_max=32,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "mosaic_cell_size", text="格子大小", slider=True)

    @staticmethod
    def process(np_array, props):
        cell = max(1, int(props.mosaic_cell_size))
        h, w = np_array.shape[:2]
        new_h, new_w = h // cell, w // cell

        if new_h < 1 or new_w < 1:
            return np_array

        trim_h, trim_w = new_h * cell, new_w * cell
        small = np_array[:trim_h, :trim_w].reshape(new_h, cell, new_w, cell, 4).mean(axis=(1, 3))
        result = np.repeat(np.repeat(small, cell, axis=0), cell, axis=1)

        if result.shape[0] < h or result.shape[1] < w:
            padded = np.zeros_like(np_array)
            rh, rw = result.shape[:2]
            padded[:rh, :rw] = result
            if rh < h:
                padded[rh:, :rw] = result[-1:, :, :]
            if rw < w:
                padded[:rh, rw:] = result[:, -1:, :]
            if rh < h and rw < w:
                padded[rh:, rw:] = result[-1:, -1:, :]
            result = padded

        return result

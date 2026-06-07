import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class PosterizeTool(BaseTool):
    tool_id = 'posterize'
    label = '色调分离'

    @staticmethod
    def get_properties():
        return {
            'posterize_levels': bpy.props.IntProperty(
                name="色阶数",
                description="量化层级数量",
                default=8,
                min=3,
                max=32,
                soft_min=4,
                soft_max=16,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "posterize_levels", text="色阶数", slider=True)

    @staticmethod
    def process(np_array, props):
        from ..np_img_utils import np_posterize
        rgb = np_posterize(np_array[:, :, :3], props.posterize_levels)
        result = np.zeros_like(np_array)
        result[:, :, :3] = rgb
        result[:, :, 3] = np_array[:, :, 3]
        return result

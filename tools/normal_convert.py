import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class NormalConvertTool(BaseTool):
    tool_id = 'normal_convert'
    label = '法线转换'

    @staticmethod
    def get_properties():
        return {
            'nc_mode': bpy.props.EnumProperty(
                name="转换模式",
                items=[
                    ('DX_TO_GL', "DX→OpenGL (翻转G)", "翻转G通道: DX→OpenGL 法线约定转换"),
                    ('2TO3', "2通→3通 (重建B)", "从R/G通道数学重建缺失的B通道"),
                    ('3TO2', "3通→2通 (清除B)", "清除B通道数据, 转为2通道格式"),
                    ('LINEAR_TO_SRGB', "线性→sRGB", "强制将线性数据转为sRGB (修复色彩空间标签错误)"),
                    ('SRGB_TO_LINEAR', "sRGB→线性", "强制将sRGB数据转为线性 (修复色彩空间标签错误)"),
                ],
                default='DX_TO_GL',
                update=_on_param_update,
            ),
            'nc_normalize': bpy.props.BoolProperty(
                name="自动归一化",
                description="转换后重新归一化法线向量至单位长度",
                default=True,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "nc_mode", text="转换模式")
        layout.prop(props, "nc_normalize", text="自动归一化")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_linear_to_srgb, np_srgb_to_linear

        rgb = np_array[:, :, :3].astype(np.float32, copy=True)
        alpha = np_array[:, :, 3].copy()

        mode = props.nc_mode

        if mode == 'DX_TO_GL':
            rgb[:, :, 1] = 1.0 - rgb[:, :, 1]

        elif mode == '2TO3':
            x = rgb[:, :, 0] * 2.0 - 1.0
            y = rgb[:, :, 1] * 2.0 - 1.0
            z = np.sqrt(np.maximum(1.0 - x * x - y * y, 0.0))
            rgb[:, :, 2] = (z + 1.0) * 0.5

        elif mode == '3TO2':
            rgb[:, :, 2] = 0.0

        elif mode == 'LINEAR_TO_SRGB':
            rgba = np.dstack([rgb, np.ones_like(alpha)])
            rgba = np_linear_to_srgb(rgba)
            rgb = rgba[:, :, :3]

        elif mode == 'SRGB_TO_LINEAR':
            rgba = np.dstack([rgb, np.ones_like(alpha)])
            rgba = np_srgb_to_linear(rgba)
            rgb = rgba[:, :, :3]

        if props.nc_normalize:
            x = rgb[:, :, 0] * 2.0 - 1.0
            y = rgb[:, :, 1] * 2.0 - 1.0
            z = rgb[:, :, 2] * 2.0 - 1.0
            length = np.sqrt(x * x + y * y + z * z)
            length = np.maximum(length, 1e-10)
            x /= length
            y /= length
            z /= length
            rgb[:, :, 0] = (x + 1.0) * 0.5
            rgb[:, :, 1] = (y + 1.0) * 0.5
            rgb[:, :, 2] = (z + 1.0) * 0.5

        result = np.zeros_like(np_array)
        result[:, :, :3] = np.clip(rgb, 0.0, 1.0)
        result[:, :, 3] = alpha
        return result

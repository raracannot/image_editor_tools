import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class BlurTool(BaseTool):
    tool_id = 'blur'
    label = '模糊'

    @staticmethod
    def get_properties():
        return {
            'blur_backend': bpy.props.EnumProperty(
                name="模糊引擎",
                description="GPU 着色器(快、半径无关) 或 CPU numpy",
                items=[
                    ('GPU', "GPU", "GPU 着色器模糊"),
                    ('CPU', "CPU", "numpy 模糊"),
                ],
                default='GPU',
                update=_on_param_update,
            ),
            'blur_radius': bpy.props.FloatProperty(
                name="模糊半径",
                description="模糊强度 (0-100)",
                default=5.0,
                min=0.1,
                max=100.0,
                soft_min=0.1,
                soft_max=50.0,
                subtype='PERCENTAGE',
                update=_on_param_update,
            ),
            'blur_mode': bpy.props.EnumProperty(
                name="模糊模式",
                description="模糊算法",
                items=[
                    ('BOX', "盒式模糊", "基于积分图的快速盒式模糊"),
                    ('GAUSSIAN', "高斯模糊", "分离卷积高斯模糊，更平滑"),
                    ('STACKED_BOX', "堆叠盒式", "多次盒式叠加，逼近高斯且更快"),
                    ('KAWASE', "Kawase 模糊", "多轮小半径迭代，柔光发光效果"),
                    ('DIRECTIONAL', "方向模糊", "旋转至指定角度做水平模糊后旋回"),
                ],
                default='BOX',
                update=_on_param_update,
            ),
            'blur_direction_angle': bpy.props.FloatProperty(
                name="模糊角度",
                description="方向模糊的角度 (-180~180, 0=水平)",
                default=0.0,
                min=-180.0,
                max=180.0,
                soft_min=-180.0,
                soft_max=180.0,
                update=_on_param_update,
            ),
            'blur_border_mode': bpy.props.EnumProperty(
                name="边界填充",
                description="图像边缘外像素的填充方式",
                items=[
                    ('edge', "边缘重复", "重复边缘像素"),
                    ('wrap', "循环平铺", "对侧像素循环填充"),
                    ('constant', "智能填充", "亮图填白暗图填均值"),
                ],
                default='edge',
                update=_on_param_update,
            ),
            'blur_fast': bpy.props.BoolProperty(
                name="快速模糊",
                description="先缩到 1/2 做模糊再还原，大幅加速",
                default=False,
                update=_on_param_update,
            ),
            'gpu_blur_radius': bpy.props.FloatProperty(
                name="模糊强度",
                description="按图像对角线比例 (0-100)，预览/应用一致",
                default=10.0,
                min=0.0,
                max=100.0,
                soft_min=0.0,
                soft_max=50.0,
                subtype='PERCENTAGE',
                update=_on_param_update,
            ),
            'gpu_blur_mode': bpy.props.EnumProperty(
                name="模糊类型",
                description="GPU 模糊算法",
                items=[
                    ('BOX', "盒式 (可分离)", "O(r) 水平+垂直两趟，快"),
                    ('GAUSSIAN', "高斯 (可分离)", "O(r) 平滑高斯"),
                    ('DOWNSAMPLE', "降采样柔光", "降采样金字塔，最快、柔和"),
                    ('KAWASE', "Kawase 发光", "多通道对角采样，柔光发光"),
                    ('RADIAL', "径向", "中心放射模糊"),
                    ('MOTION', "方向", "沿指定角度的运动模糊"),
                ],
                default='BOX',
                update=_on_param_update,
            ),
            'gpu_blur_angle': bpy.props.FloatProperty(
                name="方向角度",
                description="方向模糊角度 (-180~180, 0=水平)",
                default=0.0,
                min=-180.0,
                max=180.0,
                update=_on_param_update,
            ),
            'gpu_blur_border': bpy.props.EnumProperty(
                name="边界填充",
                description="图像边缘外像素的填充方式（所有 GPU 模式生效）",
                items=[
                    ('edge', "边缘延伸", "重复边缘像素"),
                    ('wrap', "循环平铺", "对侧像素循环填充"),
                ],
                default='edge',
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "blur_backend")
        layout.separator()
        if props.blur_backend == 'GPU':
            layout.prop(props, "gpu_blur_radius", slider=True)
            layout.prop(props, "gpu_blur_mode")
            if props.gpu_blur_mode == 'MOTION':
                layout.prop(props, "gpu_blur_angle", slider=True)
            layout.prop(props, "gpu_blur_border")
        else:
            layout.prop(props, "blur_radius", slider=True)
            layout.prop(props, "blur_mode")
            if props.blur_mode == 'DIRECTIONAL':
                layout.prop(props, "blur_direction_angle", slider=True)
            layout.prop(props, "blur_border_mode")
            layout.prop(props, "blur_fast")

    @staticmethod
    def process(np_array, props):
        if getattr(props, 'blur_backend', 'GPU') == 'GPU':
            return BlurTool._process_gpu(np_array, props)
        if props.blur_fast:
            return BlurTool._process_fast(np_array, props)
        return BlurTool._process_normal(np_array, props)

    @staticmethod
    def _process_gpu(np_array, props):
        import math
        pct = props.gpu_blur_radius
        if pct <= 0:
            return np_array
        h, w = np_array.shape[:2]
        base = max(1.0, math.hypot(w, h) / 4.0)
        radius = max(1, min(int(round(base * pct / 100.0)), int(base)))
        mode_map = {
            'BOX': 'separable', 'GAUSSIAN': 'gaussian_separable',
            'DOWNSAMPLE': 'downsample', 'KAWASE': 'kawase',
            'RADIAL': 'radial', 'MOTION': 'motion',
        }
        blur_type = mode_map.get(props.gpu_blur_mode, 'separable')
        border = props.gpu_blur_border
        angle = math.radians(props.gpu_blur_angle)
        try:
            from ..utils.gpu_img_utils import gpu_blur_npimg
            return gpu_blur_npimg(np_array, radius, blur_type=blur_type,
                                  blur_mode=border, angle=angle)
        except Exception as e:
            print(f"[模糊] GPU 失败，回退 CPU: {e}")
            from ..utils.np_img_utils import np_blur_img
            return np_blur_img(np_array, pct, border if border in ('edge', 'wrap') else 'edge')

    @staticmethod
    def _process_normal(np_array, props):
        from ..utils.np_img_utils import np_blur_img, np_gaussian_filter
        from ..utils.np_img_utils import np_stacked_box_blur, np_kawase_blur, np_directional_blur
        mode = props.blur_mode
        if mode == 'BOX':
            return np_blur_img(np_array, props.blur_radius, props.blur_border_mode)
        elif mode == 'GAUSSIAN':
            sigma = max(0.1, props.blur_radius / 10.0)
            return np_gaussian_filter(np_array, sigma, props.blur_border_mode).astype(np.float32, copy=False)
        elif mode == 'STACKED_BOX':
            return np_stacked_box_blur(np_array, props.blur_radius, props.blur_border_mode)
        elif mode == 'KAWASE':
            return np_kawase_blur(np_array, props.blur_radius, props.blur_border_mode)
        elif mode == 'DIRECTIONAL':
            return np_directional_blur(np_array, props.blur_direction_angle, props.blur_radius, props.blur_border_mode)
        return np_array

    @staticmethod
    def _process_fast(np_array, props):
        from ..utils.np_img_utils import np_resize_img, np_blur_img, np_gaussian_filter
        from ..utils.np_img_utils import np_stacked_box_blur, np_kawase_blur, np_directional_blur
        h, w = np_array.shape[:2]
        half_w, half_h = max(1, w // 2), max(1, h // 2)
        small = np_resize_img(np_array, half_w, half_h)
        fast_r = props.blur_radius * 0.5
        mode = props.blur_mode
        bm = props.blur_border_mode
        if mode == 'BOX':
            small = np_blur_img(small, fast_r, bm)
        elif mode == 'GAUSSIAN':
            sigma = max(0.1, fast_r / 10.0)
            small = np_gaussian_filter(small, sigma, bm).astype(np.float32, copy=False)
        elif mode == 'STACKED_BOX':
            small = np_stacked_box_blur(small, fast_r, bm)
        elif mode == 'KAWASE':
            small = np_kawase_blur(small, fast_r, bm)
        elif mode == 'DIRECTIONAL':
            small = np_directional_blur(small, props.blur_direction_angle, fast_r, bm)
        return np_resize_img(small, w, h)

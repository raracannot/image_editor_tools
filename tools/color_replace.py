import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update


class ColorReplaceTool(BaseTool):
    tool_id = 'color_replace'
    label = '颜色替换'

    @staticmethod
    def get_properties():
        return {
            'crep_mode': bpy.props.EnumProperty(
                name="算法",
                items=[
                    ('HUE_SHIFT', "HSL 色相替换", "保留明暗纹理，仅替换色相 (产品换色首选)"),
                    ('CHROMA_KEY', "色度键", "匹配目标 RGB 的像素直接替换为新色"),
                    ('COLOR_RANGE', "颜色范围", "PS 风格：容差范围内渐变混合"),
                ],
                default='HUE_SHIFT',
                update=_on_param_update,
            ),
            'crep_target': bpy.props.FloatVectorProperty(
                name="目标色", subtype='COLOR', size=3,
                default=(1.0, 0.0, 0.0), min=0.0, max=1.0,
                update=_on_param_update,
            ),
            'crep_replace': bpy.props.FloatVectorProperty(
                name="替换为", subtype='COLOR', size=3,
                default=(0.0, 0.0, 1.0), min=0.0, max=1.0,
                update=_on_param_update,
            ),
            'crep_tolerance': bpy.props.FloatProperty(
                name="容差", default=0.3, min=0.0, max=1.0,
                soft_min=0.01, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
                description="0=精确匹配, 1=最大范围",
            ),
            'crep_fuzziness': bpy.props.FloatProperty(
                name="羽化", default=0.1, min=0.0, max=1.0,
                soft_min=0.0, soft_max=0.5, subtype='FACTOR',
                update=_on_param_update,
                description="边缘过渡柔化程度",
            ),
            'crep_preserve_light': bpy.props.BoolProperty(
                name="保持明度", default=True,
                update=_on_param_update,
                description="替换颜色时保留原图像的明暗信息",
            ),
            'crep_mask_img': bpy.props.PointerProperty(
                name="蒙版图", type=bpy.types.Image,
                update=_on_param_update,
                description="可选灰度蒙版: 白色=替换, 黑色=保留",
            ),
            'crep_mask_invert': bpy.props.BoolProperty(
                name="反转蒙版", default=False,
                update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "crep_mode")
        layout.separator()
        layout.prop(props, "crep_target", text="目标色")
        layout.prop(props, "crep_replace", text="替换为")
        layout.prop(props, "crep_tolerance", text="容差", slider=True)
        layout.prop(props, "crep_fuzziness", text="羽化", slider=True)
        if props.crep_mode == 'HUE_SHIFT':
            layout.prop(props, "crep_preserve_light")

        layout.separator()
        row = layout.row(align=True)
        row.prop(props, "crep_mask_img", text="蒙版")
        op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
        op.target_prop = "crep_mask_img"
        if props.crep_mask_img is not None:
            layout.prop(props, "crep_mask_invert")

    @staticmethod
    def process(np_array, props):
        mask = _get_mask(np_array, props)
        target = np.array(props.crep_target, dtype=np.float32)
        replace = np.array(props.crep_replace, dtype=np.float32)
        tol = max(0.001, props.crep_tolerance)
        fuzz = max(0.0, min(tol * 0.5, props.crep_fuzziness))

        mode = props.crep_mode
        if mode == 'HUE_SHIFT':
            return _blend(np_array, _replace_hue(np_array, target, replace, tol, props.crep_preserve_light), mask)
        elif mode == 'CHROMA_KEY':
            return _blend(np_array, _chroma_key(np_array, target, replace, tol), mask)
        else:
            return _blend(np_array, _color_range_replace(np_array, target, replace, tol, fuzz), mask)


def _get_mask(np_array, props):
    img = props.crep_mask_img
    if img is None:
        return np.ones(np_array.shape[:2], dtype=np.float32)
    from ..utils.np_img_utils import blimg_2_npimg, np_resize_img
    m = blimg_2_npimg(img)
    mh, mw = m.shape[:2]
    h, w = np_array.shape[:2]
    if mh != h or mw != w:
        m = np_resize_img(m, w, h)
    gray = m[:, :, 0] * 0.299 + m[:, :, 1] * 0.587 + m[:, :, 2] * 0.114
    if props.crep_mask_invert:
        gray = 1.0 - gray
    return np.clip(gray, 0.0, 1.0)


def _blend(original, replaced, mask):
    alpha = mask[..., np.newaxis]
    return original * (1.0 - alpha) + replaced * alpha


def _hue_distance(h1, h2):
    d = abs(h1 - h2)
    return np.minimum(d, 1.0 - d)


def _replace_hue(np_array, target_rgb, replace_rgb, tolerance, preserve_light):
    from ..utils.np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
    rgb = np_array[:, :, :3]
    h, s, v = np_rgb_to_hsv(rgb)

    target_h, target_s, _ = np_rgb_to_hsv(target_rgb.reshape(1, 1, 3))
    target_h, target_s = target_h[0, 0], target_s[0, 0]

    replace_h, replace_s, replace_v = np_rgb_to_hsv(replace_rgb.reshape(1, 1, 3))
    replace_h, replace_s, replace_v = replace_h[0, 0], replace_s[0, 0], replace_v[0, 0]

    hue_match = _hue_distance(h, target_h) < tolerance * 0.5
    sat_match = s > 0.05
    alpha = np.where(hue_match & sat_match, 1.0, 0.0).astype(np.float32)

    if preserve_light:
        result_rgb = np_hsv_to_rgb(np.where(alpha > 0, replace_h, h),
                                   np.where(alpha > 0, replace_s, s),
                                   np.where(alpha > 0, np.clip(v * (replace_v / max(v.mean(), 0.001)), 0, 1), v))
    else:
        result_rgb = np_hsv_to_rgb(np.where(alpha > 0, replace_h, h),
                                   np.where(alpha > 0, replace_s, s),
                                   np.where(alpha > 0, replace_v, v))

    result = np_array.copy()
    result[:, :, :3] = _blend(rgb, result_rgb, alpha)
    return result


def _chroma_key(np_array, target_rgb, replace_rgb, tolerance):
    rgb = np_array[:, :, :3] / 1.0
    dist = np.sqrt(np.sum((rgb - target_rgb) ** 2, axis=-1))
    alpha = np.where(dist < tolerance, 1.0, 0.0).astype(np.float32)

    result = np_array.copy()
    replaced = np.ones_like(rgb) * replace_rgb
    replaced[:, :, 3] = np_array[:, :, 3]
    result[:, :, :3] = _blend(rgb, replaced[:, :, :3], alpha)
    return result


def _color_range_replace(np_array, target_rgb, replace_rgb, tolerance, fuzziness):
    rgb = np_array[:, :, :3]

    lab_t = _srgb_to_lab_3(target_rgb)
    lab_s = _srgb_to_lab_vec(rgb)

    de = np.sqrt(np.sum((lab_s - lab_t) ** 2, axis=-1))
    de_max = np.sqrt(3.0) * 100.0
    dist_norm = de / de_max

    inner = tolerance * 0.6
    outer = tolerance

    alpha = np.where(dist_norm < inner, 1.0, 0.0).astype(np.float32)
    soft = (dist_norm >= inner) & (dist_norm < outer)
    if fuzziness > 0 and np.any(soft):
        soft_val = 1.0 - (dist_norm[soft] - inner) / max(outer - inner, fuzziness)
        alpha[soft] = np.clip(soft_val, 0.0, 1.0)

    result = np_array.copy()
    lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    lum_ratio = np.where(lum > 0.001, lum / np.maximum(lum.mean(), 0.001), 1.0)
    replaced = np.clip(replace_rgb * lum_ratio[..., np.newaxis], 0, 1)
    result[:, :, :3] = _blend(rgb, replaced, alpha)
    return result


def _srgb_to_lab_3(rgb):
    c = np.clip(rgb, 0, 1)
    lo = c <= 0.04045
    c[lo] /= 12.92
    c[~lo] = ((c[~lo] + 0.055) / 1.055) ** 2.4
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = np.dot(c, M.T)
    ref = np.array([0.9505, 1.0, 1.0890], dtype=np.float32)
    xyz = xyz / ref
    t = xyz ** (1.0 / 3.0)
    mask = xyz <= 0.008856
    t[mask] = 7.787 * xyz[mask] + 16.0 / 116.0
    return np.array([116.0 * t[1] - 16.0, 500.0 * (t[0] - t[1]), 200.0 * (t[1] - t[2])], dtype=np.float32)


def _srgb_to_lab_vec(rgb):
    c = np.clip(rgb, 0, 1).astype(np.float32)
    lo = c <= 0.04045
    c[lo] /= 12.92
    c[~lo] = ((c[~lo] + 0.055) / 1.055) ** 2.4
    M = np.array([[0.4124, 0.3576, 0.1805],
                  [0.2126, 0.7152, 0.0722],
                  [0.0193, 0.1192, 0.9505]], dtype=np.float32)
    xyz = np.tensordot(c, M, axes=([2], [1]))
    ref = np.array([0.9505, 1.0, 1.0890], dtype=np.float32)
    xyz = xyz / ref
    t = xyz ** (1.0 / 3.0)
    mask = xyz <= 0.008856
    t[mask] = 7.787 * xyz[mask] + 16.0 / 116.0
    L = 116.0 * t[..., 1] - 16.0
    A = 500.0 * (t[..., 0] - t[..., 1])
    B = 200.0 * (t[..., 1] - t[..., 2])
    return np.stack([L, A, B], axis=-1).astype(np.float32)

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
                    ('LAB_LOCK', "LAB 明度锁定", "保留纹理光影, 仅替换色度 (最自然)"),
                    ('HUE_SHIFT', "HSL 色相偏移", "色相偏移 + 饱和度迁移"),
                ],
                default='LAB_LOCK',
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
                name="容差", default=0.5, min=0.0, max=1.0,
                soft_min=0.05, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
                description="颜色匹配范围: 越大越宽泛",
            ),
            'crep_fuzziness': bpy.props.FloatProperty(
                name="羽化", default=0.15, min=0.0, max=0.5,
                soft_min=0.0, soft_max=0.3, subtype='FACTOR',
                update=_on_param_update,
                description="边缘过渡柔化程度",
            ),
            'crep_strength': bpy.props.FloatProperty(
                name="强度", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update,
                description="整体替换强度",
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
        layout.prop(props, "crep_strength", text="强度", slider=True)

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
        target = np.asarray(props.crep_target, dtype=np.float32)
        replace = np.asarray(props.crep_replace, dtype=np.float32)
        tol = max(0.01, props.crep_tolerance)
        fuzzy = max(0.0, props.crep_fuzziness)
        strength = props.crep_strength

        if props.crep_mode == 'LAB_LOCK':
            result = _lab_lock_replace(np_array, target, replace, tol, fuzzy, strength)
        else:
            result = _hsl_shift_replace(np_array, target, replace, tol, fuzzy, strength)

        return _blend(np_array, result, mask)


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


def _build_soft_mask(lab, target_rgb, tolerance, fuzziness):
    from ..utils.np_img_utils import np_rgb_to_lab
    target_lab = np_rgb_to_lab(np.full((1, 1, 4), list(target_rgb) + [1.0], dtype=np.float32))
    ta, tb = target_lab[0, 0, 1], target_lab[0, 0, 2]

    da = lab[:, :, 1] - ta
    db = lab[:, :, 2] - tb
    ab_dist = np.sqrt(da * da + db * db) / 128.0
    ab_dist = np.clip(ab_dist, 0.0, 1.0)

    inner = tolerance * 0.7
    outer = tolerance
    if outer <= inner:
        outer = inner + 0.01

    mask = np.where(ab_dist < inner, 1.0, 0.0).astype(np.float32)
    soft = (ab_dist >= inner) & (ab_dist < outer)
    t = (ab_dist[soft] - inner) / max(outer - inner, 1e-6)
    mask[soft] = 1.0 - t * (1.0 - fuzziness * 2.0)
    mask = np.clip(mask, 0.0, 1.0)
    return mask


def _lab_lock_replace(np_array, target, replace, tolerance, fuzziness, strength):
    from ..utils.np_img_utils import np_rgb_to_lab
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    lab = np_rgb_to_lab(np_array)
    L = lab[:, :, 0]
    a = lab[:, :, 1]
    b = lab[:, :, 2]

    mask = _build_soft_mask(lab, target, tolerance, fuzziness)

    replace_lab = np_rgb_to_lab(np.full((1, 1, 4), list(replace) + [1.0], dtype=np.float32))
    rt_a, rt_b = replace_lab[0, 0, 1], replace_lab[0, 0, 2]

    sa = np.std(a[mask > 0.1]) if np.any(mask > 0.1) else 1.0
    sb = np.std(b[mask > 0.1]) if np.any(mask > 0.1) else 1.0
    sa = max(sa, 1.0)
    sb = max(sb, 1.0)

    a_new = a + (rt_a - a) * mask
    b_new = b + (rt_b - b) * mask

    lab_new = np.stack([L, a_new, b_new], axis=-1)
    from ..utils.np_img_utils import np_lab_to_rgb
    result_rgb = np_lab_to_rgb(lab_new)

    result = np.zeros_like(np_array)
    s = strength
    result[:, :, :3] = np.clip(rgb * (1.0 - mask[..., np.newaxis] * s) + result_rgb * mask[..., np.newaxis] * s, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _hsl_shift_replace(np_array, target, replace, tolerance, fuzziness, strength):
    from ..utils.np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    h, s, v = np_rgb_to_hsv(rgb)

    target_hsv = np_rgb_to_hsv(target.reshape(1, 1, 3))
    replace_hsv = np_rgb_to_hsv(replace.reshape(1, 1, 3))
    t_h, t_s, _ = target_hsv[0][0, 0], target_hsv[1][0, 0], target_hsv[2][0, 0]
    r_h, r_s, r_v = replace_hsv[0][0, 0], replace_hsv[1][0, 0], replace_hsv[2][0, 0]

    dh = abs(h - t_h)
    dh = np.minimum(dh, 1.0 - dh)
    inner = tolerance * 0.5
    outer = tolerance
    if outer <= inner:
        outer = inner + 0.01

    mask = np.where(dh < inner, 1.0, 0.0).astype(np.float32)
    soft = (dh >= inner) & (dh < outer)
    if np.any(soft):
        t_val = (dh[soft] - inner) / max(outer - inner, 1e-6)
        mask[soft] = 1.0 - t_val * (1.0 - fuzziness * 2.0)
    mask = np.clip(mask, 0.0, 1.0)

    s_scale = np.where(mask > 0, r_s / np.maximum(t_s, 0.01), 1.0)
    s_new = np.clip(s * (1.0 + (s_scale - 1.0) * mask), 0.0, 1.0)

    v_blend = np.where(mask > 0, r_v / np.maximum(np.mean(v), 0.01), 1.0)
    v_new = np.clip(v * (1.0 + (v_blend - 1.0) * mask * 0.3), 0.0, 1.0)

    h_offset = (r_h - t_h + 0.5) % 1.0 - 0.5
    h_new = (h + h_offset * mask) % 1.0

    result_rgb = np_hsv_to_rgb(h_new, s_new, v_new)

    result = np.zeros_like(np_array)
    s_mask = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - s_mask) + result_rgb * s_mask, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _blend(original, replaced, mask):
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    return np.clip(original * (1.0 - mask) + replaced * mask, 0.0, 1.0)

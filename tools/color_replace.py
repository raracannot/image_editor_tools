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
                    ('YCBCR_LOCK', "YCbCr 亮度锁定", "视频级亮度分离, 感知均匀"),
                    ('LCH_LOCK', "LCh 色度锁定", "极坐标LAB, 保留饱和度结构"),
                    ('HSV_LOCK', "HSV 亮度锁定", "HSV亮度+饱和度, 仅偏移色相"),
                    ('HUE_SHIFT', "HSL 色相偏移", "色相偏移 + 饱和度迁移"),
                    ('HUE_FILL', "色相统一填充", "整块色相区域统一替换"),
                     ('RGB_GAIN', "RGB 增益偏移", "通道级增益+偏移 (非破坏式)"),
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
            'crep_lum_min': bpy.props.FloatProperty(
                name="亮度下限", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update, description="低于此亮度的像素不替换",
            ),
            'crep_lum_max': bpy.props.FloatProperty(
                name="亮度上限", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update, description="高于此亮度的像素不替换",
            ),
            'crep_sat_min': bpy.props.FloatProperty(
                name="饱和度下限", default=0.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_param_update, description="低于此饱和度的像素不替换",
            ),
            'crep_refine_mask': bpy.props.BoolProperty(
                name="边缘精炼", default=False,
                update=_on_param_update, description="对蒙版做 1-2px 模糊消除锯齿",
            ),
            'crep_spill_suppress': bpy.props.BoolProperty(
                name="溢出抑制", default=False,
                update=_on_param_update, description="过渡区去饱和减少旧色残留",
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

        box = layout.box()
        box.label(text="亮度/饱和度约束")
        row = box.row(align=True)
        row.prop(props, "crep_lum_min", text="亮度", slider=True)
        row.prop(props, "crep_lum_max", text="", slider=True)
        box.prop(props, "crep_sat_min", text="饱和度下限", slider=True)

        box = layout.box()
        box.label(text="精炼")
        box.prop(props, "crep_refine_mask", text="边缘精炼")
        box.prop(props, "crep_spill_suppress", text="溢出抑制")

        layout.separator()
        row = layout.row(align=True)
        row.prop(props, "crep_mask_img", text="蒙版")
        op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
        op.target_prop = "crep_mask_img"
        if props.crep_mask_img is not None:
            layout.prop(props, "crep_mask_invert")

    @staticmethod
    def process(np_array, props):
        external_mask = _get_mask(np_array, props)
        target = np.asarray(props.crep_target, dtype=np.float32)
        replace = np.asarray(props.crep_replace, dtype=np.float32)
        tol = max(0.001, props.crep_tolerance ** 0.5)
        fuzzy = max(0.0, props.crep_fuzziness)
        strength = props.crep_strength

        mode = props.crep_mode
        if mode == 'LAB_LOCK':
            result = _lab_lock_replace(np_array, target, replace, tol, fuzzy, strength)
        elif mode == 'YCBCR_LOCK':
            result = _ycbcr_lock_replace(np_array, target, replace, tol, fuzzy, strength)
        elif mode == 'LCH_LOCK':
            result = _lch_lock_replace(np_array, target, replace, tol, fuzzy, strength)
        elif mode == 'HSV_LOCK':
            result = _hsv_lock_replace(np_array, target, replace, tol, fuzzy, strength)
        elif mode == 'HUE_FILL':
            result = _hue_fill_replace(np_array, target, replace, tol, fuzzy, strength)
        elif mode == 'RGB_GAIN':
            result = _rgb_gain_replace(np_array, target, replace, tol, fuzzy, strength)
        else:
            result = _hsl_shift_replace(np_array, target, replace, tol, fuzzy, strength)

        lumsat_mask = _build_lumsat_mask(np_array, props.crep_lum_min, props.crep_lum_max, props.crep_sat_min)
        combined_mask = external_mask * lumsat_mask

        if props.crep_refine_mask:
            from ..utils.np_img_utils import np_blur_img
            combined_mask = np_blur_img(combined_mask, 2.0, 'edge')

        result = _blend(np_array, result, combined_mask)

        if props.crep_spill_suppress:
            result = _spill_suppress(result, combined_mask, replace)

        return result


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


def _build_soft_mask_generic(dist, tolerance, fuzziness, inner_factor=0.7):
    inner = tolerance * inner_factor
    outer = tolerance
    if outer <= inner:
        outer = inner + 0.01

    mask = np.where(dist < inner, 1.0, 0.0).astype(np.float32)
    soft = (dist >= inner) & (dist < outer)
    if np.any(soft):
        t_val = (dist[soft] - inner) / max(outer - inner, 1e-6)
        mask[soft] = 1.0 - t_val * (1.0 - fuzziness * 2.0)
    return np.clip(mask, 0.0, 1.0)


def _build_soft_mask(lab, target_rgb, tolerance, fuzziness):
    from ..utils.np_img_utils import np_rgb_to_lab
    target_lab = np_rgb_to_lab(np.full((1, 1, 4), list(target_rgb) + [1.0], dtype=np.float32))
    ta, tb = target_lab[0, 0, 1], target_lab[0, 0, 2]

    da = lab[:, :, 1] - ta
    db = lab[:, :, 2] - tb
    ab_dist = np.sqrt(da * da + db * db) / 128.0
    ab_dist = np.clip(ab_dist, 0.0, 1.0)

    return _build_soft_mask_generic(ab_dist, tolerance, fuzziness, 0.7)


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
    mask = _build_soft_mask_generic(dh, tolerance, fuzziness, 0.5)

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


def _ycbcr_lock_replace(np_array, target, replace, tolerance, fuzziness, strength):
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    Y, Cb, Cr = _rgb_to_ycbcr(rgb)
    t_Y, t_Cb, t_Cr = _rgb_to_ycbcr(target.reshape(1, 1, 3))
    r_Y, r_Cb, r_Cr = _rgb_to_ycbcr(replace.reshape(1, 1, 3))
    t_Cb, t_Cr = t_Cb[0, 0], t_Cr[0, 0]
    r_Cb, r_Cr = r_Cb[0, 0], r_Cr[0, 0]

    dCb = Cb - t_Cb
    dCr = Cr - t_Cr
    dist = np.sqrt(dCb * dCb + dCr * dCr) / 0.5
    dist = np.clip(dist, 0.0, 1.0)
    mask = _build_soft_mask_generic(dist, tolerance, fuzziness, 0.6)

    Cb_new = Cb + (r_Cb - Cb) * mask
    Cr_new = Cr + (r_Cr - Cr) * mask
    result_rgb = _ycbcr_to_rgb(Y, Cb_new, Cr_new)

    result = np.zeros_like(np_array)
    sm = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - sm) + result_rgb * sm, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _hue_fill_replace(np_array, target, replace, tolerance, fuzziness, strength):
    from ..utils.np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    h, s, v = np_rgb_to_hsv(rgb)
    th = np_rgb_to_hsv(target.reshape(1, 1, 3))[0][0, 0]
    rh = np_rgb_to_hsv(replace.reshape(1, 1, 3))[0][0, 0]

    dh = abs(h - th)
    dh = np.minimum(dh, 1.0 - dh)
    mask = _build_soft_mask_generic(dh, tolerance, fuzziness, 0.5)

    h_offset = (rh - th + 0.5) % 1.0 - 0.5
    h_new = (h + h_offset * mask) % 1.0
    result_rgb = np_hsv_to_rgb(h_new, s, v)

    result = np.zeros_like(np_array)
    sm = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - sm) + result_rgb * sm, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _rgb_gain_replace(np_array, target, replace, tolerance, fuzziness, strength):
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    t = np.clip(target, 0.01, 1.0)
    r = np.clip(replace, 0.0, 1.0)
    gain = np.clip(r / t, 0.01, 100.0)

    dist = np.sqrt(np.sum((rgb - target) ** 2, axis=-1)) / np.sqrt(3.0)
    mask = _build_soft_mask_generic(dist, tolerance, fuzziness, 0.5)

    result_rgb = np.clip(rgb * (1.0 + (gain - 1.0) * mask[..., np.newaxis]), 0.0, 1.0)

    result = np.zeros_like(np_array)
    sm = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - sm) + result_rgb * sm, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _rgb_to_ycbcr(rgb):
    M = np.array([[0.299, 0.587, 0.114],
                  [-0.168736, -0.331264, 0.5],
                  [0.5, -0.418688, -0.081312]], dtype=np.float32)
    ycbcr = np.tensordot(rgb, M.T, axes=([2], [1]))
    return ycbcr[..., 0], ycbcr[..., 1], ycbcr[..., 2]


def _ycbcr_to_rgb(Y, Cb, Cr):
    M = np.array([[1.0, 0.0, 1.402],
                  [1.0, -0.344136, -0.714136],
                  [1.0, 1.772, 0.0]], dtype=np.float32)
    ycbcr = np.stack([Y, Cb, Cr], axis=-1)
    rgb = np.tensordot(ycbcr, M.T, axes=([2], [1]))
    return np.clip(rgb, 0.0, 1.0)


def _lch_lock_replace(np_array, target, replace, tolerance, fuzziness, strength):
    from ..utils.np_img_utils import np_rgb_to_lab
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    lab = np_rgb_to_lab(np_array)
    L, a, b = lab[:, :, 0], lab[:, :, 1], lab[:, :, 2]
    C = np.sqrt(a * a + b * b)
    h = np.arctan2(b, a)

    target_lab = np_rgb_to_lab(np.full((1, 1, 4), list(target) + [1.0], dtype=np.float32))
    t_a, t_b = target_lab[0, 0, 1], target_lab[0, 0, 2]
    replace_lab = np_rgb_to_lab(np.full((1, 1, 4), list(replace) + [1.0], dtype=np.float32))
    r_a, r_b = replace_lab[0, 0, 1], replace_lab[0, 0, 2]

    da = a - t_a
    db = b - t_b
    dist = np.sqrt(da * da + db * db) / 128.0
    dist = np.clip(dist, 0.0, 1.0)
    mask = _build_soft_mask_generic(dist, tolerance, fuzziness, 0.7)

    a_new = a + (r_a - a) * mask
    b_new = b + (r_b - b) * mask

    lab_new = np.stack([L, a_new, b_new], axis=-1)
    from ..utils.np_img_utils import np_lab_to_rgb
    result_rgb = np_lab_to_rgb(lab_new)

    result = np.zeros_like(np_array)
    sm = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - sm) + result_rgb * sm, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _hsv_lock_replace(np_array, target, replace, tolerance, fuzziness, strength):
    from ..utils.np_img_utils import np_rgb_to_hsv, np_hsv_to_rgb
    rgb = np_array[:, :, :3].copy()
    alpha_ch = np_array[:, :, 3].copy()

    h, s, v = np_rgb_to_hsv(rgb)
    target_hsv = np_rgb_to_hsv(target.reshape(1, 1, 3))
    replace_hsv = np_rgb_to_hsv(replace.reshape(1, 1, 3))
    th, ts = target_hsv[0][0, 0], target_hsv[1][0, 0]
    rh, rs = replace_hsv[0][0, 0], replace_hsv[1][0, 0]

    dh = abs(h - th)
    dh = np.minimum(dh, 1.0 - dh)
    mask = _build_soft_mask_generic(dh, tolerance, fuzziness, 0.5)

    origin_s = np.mean(s[mask > 0.1]) if np.any(mask > 0.1) else ts
    if origin_s < 0.01:
        origin_s = 0.01
    s_ratio = rs / origin_s
    h_offset = (rh - th + 0.5) % 1.0 - 0.5
    h_new = (h + h_offset * mask) % 1.0
    s_new = np.clip(s * (1.0 + (s_ratio - 1.0) * mask), 0.0, 1.0)

    result_rgb = np_hsv_to_rgb(h_new, s_new, v)

    result = np.zeros_like(np_array)
    sm = mask[..., np.newaxis] * strength
    result[:, :, :3] = np.clip(rgb * (1.0 - sm) + result_rgb * sm, 0, 1)
    result[:, :, 3] = alpha_ch
    return result


def _build_lumsat_mask(np_array, lum_min, lum_max, sat_min):
    if lum_min <= 0.0 and lum_max >= 1.0 and sat_min <= 0.0:
        return np.ones(np_array.shape[:2], dtype=np.float32)

    rgb = np_array[:, :, :3]
    lum = rgb[:, :, 0] * 0.299 + rgb[:, :, 1] * 0.587 + rgb[:, :, 2] * 0.114
    max_c = np.max(rgb, axis=-1)
    min_c = np.min(rgb, axis=-1)
    sat = np.where(max_c > 1e-6, (max_c - min_c) / np.maximum(max_c, 1e-10), 0.0)

    mask = np.ones_like(lum, dtype=np.float32)
    if lum_min > 0.0:
        mask = np.where(lum < lum_min, 0.0, mask)
    if lum_max < 1.0:
        mask = np.where(lum > lum_max, 0.0, mask)
    if sat_min > 0.0:
        mask = np.where(sat < sat_min, 0.0, mask)
    return mask


def _spill_suppress(result, mask, replace_rgb):
    spill = (mask > 0.05) & (mask < 0.5)
    if not np.any(spill):
        return result

    fix = result[:, :, :3] * 0.85 + replace_rgb * 0.15
    result[:, :, :3] = np.where(spill[..., np.newaxis], np.clip(fix, 0.0, 1.0), result[:, :, :3])
    return result


def _blend(original, replaced, mask):
    if mask.ndim == 2:
        mask = mask[..., np.newaxis]
    return np.clip(original * (1.0 - mask) + replaced * mask, 0.0, 1.0)

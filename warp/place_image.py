import bpy
import gpu
import blf
import math
import numpy as np
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl
from ..utils.np_img_utils import blimg_2_npimg, npimg_2_blimg
from ..utils.blend_modes import apply_blend_mode
from .base_op import WarpModalBase


BLEND_MODE_ITEMS = [
    ('MIX', "正常", ""), ('DARKEN', "变暗", ""), ('MULTIPLY', "正片叠底", ""),
    ('BURN', "颜色加深", ""), ('LIGHTEN', "变亮", ""), ('SCREEN', "滤色", ""),
    ('DODGE', "颜色减淡", ""), ('ADD', "线性减淡", ""), ('OVERLAY', "叠加", ""),
    ('SOFT_LIGHT', "柔光", ""), ('LINEAR_LIGHT', "线性光", ""),
    ('DIFFERENCE', "差值", ""), ('EXCLUSION', "排除", ""),
    ('SUBTRACT', "减去", ""), ('DIVIDE', "划分", ""),
]


class PlaceImageEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):
        self.fg_image = None
        self._fg_tex = None
        self._fg_image_ref = None

        img_w, img_h = image.size
        self.offset_x = img_w / 2.0
        self.offset_y = img_h / 2.0
        self.scale = 0.5
        self.rotation = 0.0

        self._drag_mode = 'NONE'
        self._drag_start_pos = (0, 0)
        self._drag_start_offset = (img_w / 2.0, img_h / 2.0)
        self._drag_start_scale = 0.5
        self._drag_start_rot = 0.0
        self._drag_start_angle = 0.0

        self._load_fg_image()
        super().__init__(context, image)
        self._ensure_top_tex()

    def _load_fg_image(self):
        props = bpy.context.scene.image_editor_tools
        fg = getattr(props, 'place_img_fg', None)
        if fg is not None:
            self.fg_image = fg

    def _ensure_top_tex(self):
        if self.fg_image is None:
            return
        if self._fg_image_ref is not self.fg_image:
            try:
                if self._fg_tex is not None:
                    del self._fg_tex
            except Exception:
                pass
            self._fg_tex = gpu.texture.from_image(self.fg_image)
            self._fg_image_ref = self.fg_image

    def _get_quad_verts(self):
        offset_x, offset_y, disp_w, disp_h = self._img_rect
        if disp_w <= 0 or disp_h <= 0:
            return [(0, 0)] * 4

        img_w, img_h = self.original_image.size
        fw, fh = (img_w, img_h)
        if self.fg_image is not None:
            fw, fh = self.fg_image.size
        if fw <= 0 or fh <= 0:
            fw, fh = img_w, img_h

        ratio = fw / max(fh, 1)
        disp_ratio = disp_w / max(disp_h, 1)

        if ratio >= disp_ratio:
            sx = self.scale * disp_w
            sy = sx / ratio
        else:
            sy = self.scale * disp_h
            sx = sy * ratio

        cx = offset_x + self.offset_x / img_w * disp_w
        cy = offset_y + self.offset_y / img_h * disp_h

        hw = sx / 2.0
        hh = sy / 2.0
        ca, sa = math.cos(self.rotation), math.sin(self.rotation)

        corners = [
            (-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh),
        ]
        result = []
        for dx, dy in corners:
            rx = cx + dx * ca - dy * sa
            ry = cy + dx * sa + dy * ca
            result.append((rx, ry))
        return result

    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect
        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        self._load_fg_image()
        self._ensure_top_tex()

        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

        quad = self._get_quad_verts()

        if self._fg_tex is not None:
            gpu.state.blend_set('ALPHA')
            shader.bind()
            shader.uniform_sampler("image", self._fg_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": quad,
                "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
            }).draw(shader)
            gpu.state.blend_set('NONE')

        line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader.bind()
        line_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))

        p0, p1, p2, p3 = quad
        lines = [p0, p1, p1, p2, p2, p3, p3, p0]
        batch_for_shader(line_shader, 'LINES', {"pos": lines}).draw(line_shader)

        handles = [(px, py) for px, py in quad]
        h_size = 5
        for px, py in handles:
            batch_for_shader(line_shader, 'TRI_FAN', {
                "pos": [(px - h_size, py - h_size), (px + h_size, py - h_size),
                        (px + h_size, py + h_size), (px - h_size, py + h_size)]
            }).draw(line_shader)

        cx, cy = sum(p[0] for p in quad) / 4.0, sum(p[1] for p in quad) / 4.0
        top_y = min(p[1] for p in quad)
        rot_pt = (cx, top_y - 25)
        rot_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        rot_shader.bind()
        rot_shader.uniform_float("color", (0.2, 0.6, 1.0, 1.0))
        segments = 16
        circle_verts = [(math.cos(2.0 * math.pi * i / segments), math.sin(2.0 * math.pi * i / segments)) for i in range(segments)]
        rv = [(rot_pt[0] + v[0] * 6, rot_pt[1] + v[1] * 6) for v in circle_verts]
        batch_for_shader(rot_shader, 'TRI_FAN', {"pos": rv}).draw(rot_shader)

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        s = pget_tmpl("置入图像 | 拖拽移动 | 角点缩放 | 旋转 ({angle}°) [Shift吸附] | Enter应用 | Esc取消",
                      angle=f"{math.degrees(self.rotation):.1f}")
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, s)

    def handle_mouse_press(self, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        self._drag_mode = self._get_hit_zone(mx, my)
        if self._drag_mode == 'NONE':
            return False
        self._drag_start_pos = (mx, my)
        self._drag_start_offset = (self.offset_x, self.offset_y)
        self._drag_start_scale = self.scale
        self._drag_start_rot = self.rotation
        if self._drag_mode == 'ROTATE':
            ox, oy, dw, dh = self._img_rect
            img_w, img_h = self.original_image.size
            cx = ox + self.offset_x / img_w * dw
            cy = oy + self.offset_y / img_h * dh
            self._drag_start_angle = math.atan2(my - cy, mx - cx)
        return True

    def handle_mouse_move(self, event):
        if self._drag_mode == 'NONE':
            return False
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        img_w, img_h = self.original_image.size

        if self._drag_mode == 'MOVE':
            ndx = (mx - self._drag_start_pos[0]) / dw * img_w if dw > 0 else 0
            ndy = (my - self._drag_start_pos[1]) / dh * img_h if dh > 0 else 0
            self.offset_x = self._drag_start_offset[0] + ndx
            self.offset_y = self._drag_start_offset[1] + ndy

        elif self._drag_mode == 'SCALE':
            cx = ox + self._drag_start_offset[0] / img_w * dw
            cy = oy + self._drag_start_offset[1] / img_h * dh
            start_dist = max(1.0, math.hypot(self._drag_start_pos[0] - cx, self._drag_start_pos[1] - cy))
            cur_dist = max(1.0, math.hypot(mx - cx, my - cy))
            raw_scale = self._drag_start_scale * cur_dist / start_dist
            self.scale = max(0.01, min(10.0, raw_scale))

        elif self._drag_mode == 'ROTATE':
            cx = ox + self._drag_start_offset[0] / img_w * dw
            cy = oy + self._drag_start_offset[1] / img_h * dh
            cur_angle = math.atan2(my - cy, mx - cx)
            angle_diff = cur_angle - self._drag_start_angle
            raw_rot = self._drag_start_rot + angle_diff
            if event.shift:
                deg = math.degrees(raw_rot)
                deg = round(deg / 5.0) * 5.0
                self.rotation = math.radians(deg)
            else:
                self.rotation = raw_rot

        return True

    def handle_mouse_release(self, event):
        self._drag_mode = 'NONE'

    def _corner_hit_radius(self):
        return 15

    def _get_hit_zone(self, mx, my):
        quad = self._get_quad_verts()
        if len(quad) < 4:
            return 'NONE'

        p0, p1, p2, p3 = quad

        cx = (p0[0] + p1[0] + p2[0] + p3[0]) / 4.0
        cy = (p0[1] + p1[1] + p2[1] + p3[1]) / 4.0
        top_y = min(p0[1], p1[1], p2[1], p3[1])

        rot_pt = (cx, top_y - 25)
        if math.hypot(mx - rot_pt[0], my - rot_pt[1]) < 12:
            return 'ROTATE'

        cr = self._corner_hit_radius()
        for corner in quad:
            if math.hypot(mx - corner[0], my - corner[1]) < cr:
                return 'SCALE'

        if self._point_in_quad(mx, my, quad):
            return 'MOVE'

        return 'NONE'

    def _point_in_quad(self, mx, my, quad):
        p0, p1, p2, p3 = quad
        return self._cross(mx, my, p0, p1) >= 0 and \
               self._cross(mx, my, p1, p2) >= 0 and \
               self._cross(mx, my, p2, p3) >= 0 and \
               self._cross(mx, my, p3, p0) >= 0

    def _cross(self, px, py, a, b):
        return (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])

    def apply_to_original(self):
        try:
            self._load_fg_image()
            if self.fg_image is None:
                self.cleanup()
                return
            bg_np = blimg_2_npimg(self.original_image)
            fg_np = blimg_2_npimg(self.fg_image)
            result = self._composite(bg_np, fg_np)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            print(f"[置入图像] 应用失败: {e}")
        finally:
            self.cleanup()

    def save_as_copy(self):
        try:
            self._load_fg_image()
            if self.fg_image is None:
                self.cleanup()
                return
            bg_np = blimg_2_npimg(self.original_image)
            fg_np = blimg_2_npimg(self.fg_image)
            result = self._composite(bg_np, fg_np)
            new_name = self.original_image.name + "_placed"
            npimg_2_blimg(result, new_name, True)
            bpy.ops.ed.undo_push(message="置入图像另存")
        except Exception as e:
            print(f"[置入图像] 另存失败: {e}")
        finally:
            self.cleanup()

    def _composite(self, bg_np, fg_np):
        h, w = bg_np.shape[:2]
        fh, fw = fg_np.shape[:2]
        img_w, img_h = self.original_image.size

        props = bpy.context.scene.image_editor_tools
        mode = getattr(props, 'place_img_mode', 'MIX')
        opacity = getattr(props, 'place_img_opacity', 1.0)

        Y, X = np.mgrid[0:h, 0:w].astype(np.float32)
        px_cx = self.offset_x
        px_cy = self.offset_y
        rel_sx = self.scale * (img_w / max(fw, 1))
        rel_sy = self.scale * (img_h / max(fh, 1))
        scl = min(rel_sx, rel_sy)
        src_cx = fw / 2.0
        src_cy = fh / 2.0
        cos_a = math.cos(-self.rotation)
        sin_a = math.sin(-self.rotation)

        dx = X - px_cx
        dy = Y - px_cy
        src_x = (dx * cos_a - dy * sin_a) / scl + src_cx
        src_y = (dx * sin_a + dy * cos_a) / scl + src_cy
        src_x = src_x.astype(np.float32)
        src_y = src_y.astype(np.float32)

        valid = (src_x >= 0) & (src_x < fw - 1) & (src_y >= 0) & (src_y < fh - 1)

        x0 = np.floor(src_x).astype(np.int32)
        y0 = np.floor(src_y).astype(np.int32)
        x1 = np.clip(x0 + 1, 0, fw - 1)
        y1 = np.clip(y0 + 1, 0, fh - 1)
        x0 = np.clip(x0, 0, fw - 1)
        y0 = np.clip(y0, 0, fh - 1)

        fx = src_x - x0.astype(np.float32)
        fy = src_y - y0.astype(np.float32)
        fx = fx[..., np.newaxis]
        fy = fy[..., np.newaxis]

        v00 = fg_np[y0, x0, :]
        v10 = fg_np[y0, x1, :]
        v01 = fg_np[y1, x0, :]
        v11 = fg_np[y1, x1, :]
        sampled = (v00 * (1 - fx) * (1 - fy) + v10 * fx * (1 - fy) +
                   v01 * (1 - fx) * fy + v11 * fx * fy)
        sampled[~valid] = [0, 0, 0, 0]

        fg_rgb = sampled[:, :, :3]
        fg_a = sampled[:, :, 3] * opacity
        bg_rgb = bg_np[:, :, :3]
        bg_a = bg_np[:, :, 3]

        blended = apply_blend_mode(bg_rgb, fg_rgb, mode)

        alpha = fg_a
        out_alpha = alpha + bg_a * (1.0 - alpha)
        out_alpha_safe = np.where(out_alpha < 1e-5, 1.0, out_alpha)
        out_rgb = (blended * alpha[..., np.newaxis] + bg_rgb * bg_a[..., np.newaxis] * (1.0 - alpha)[..., np.newaxis]) / out_alpha_safe[..., np.newaxis]

        result = np.zeros_like(bg_np)
        result[:, :, :3] = out_rgb
        result[:, :, 3] = np.where(valid, out_alpha, bg_a)
        return result

    def cleanup(self):
        if self._fg_tex is not None:
            try:
                del self._fg_tex
            except Exception:
                pass
            self._fg_tex = None
            self._fg_image_ref = None
        super().cleanup()


class IMAGE_OT_place_image_modal(WarpModalBase):
    bl_idname = "image_editor_tools.place_image_modal"
    bl_label = "置入图像"
    engine_class = PlaceImageEngine
    tool_key = 'warp:置入图像'
    tool_label = '置入图像'
    status_text = "置入图像: 拖拽移动 | 角点缩放 | 旋转 [Shift吸附] | Enter应用 | Esc取消"
    _drag_attr = '_drag_mode'
    _drag_none = 'NONE'

    def _custom_keys(self, context, event, engine):
        if event.type == 'F' and event.value == 'PRESS':
            engine.offset_x = engine.original_image.size[0] / 2.0
            engine.offset_y = engine.original_image.size[1] / 2.0
            engine.scale = 0.5
            engine.rotation = 0.0
            return True
        return False

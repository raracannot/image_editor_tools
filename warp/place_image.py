import bpy
import gpu
import blf
import math
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..utils import gpu_img_utils as giu
from ..translation import pget_tmpl
from .base_op import WarpModalBase


class PlaceImageEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):
        self.fg_image = None
        self._fg_raw_tex = None
        self._fg_raw_ref = None
        self._bg_raw_tex = None

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
        if self._fg_raw_ref is not self.fg_image:
            self._fg_raw_tex = giu.tex_from_image_raw(self.fg_image)
            self._fg_raw_ref = self.fg_image

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

    def _draw_composite_live(self, offset_x, offset_y, disp_w, disp_h):
        # 用合成着色器在屏幕做单 pass 合成(实时混合模式/不透明度)。
        # 用裸值纹理 + srgb_out=1，混合空间与落地一致(线性光等阈值类混合模式预览=应用)。
        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        if self.fg_image is None or self._fg_raw_tex is None or self._cached_orig_tex is None:
            shader = state.get_display_shader()
            shader.bind()
            shader.uniform_sampler("image", self._cached_orig_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
            }).draw(shader)
            return

        if self._bg_raw_tex is None:
            self._bg_raw_tex = giu.tex_from_image_raw(self.original_image)

        img_w, img_h = self.original_image.size
        fw, fh = self.fg_image.size

        props = bpy.context.scene.image_editor_tools
        mode = getattr(props, 'place_img_mode', 'MIX')
        opacity = getattr(props, 'place_img_opacity', 1.0)

        shader = giu.get_composite_shader()
        giu.bind_composite(
            shader, self._bg_raw_tex, self._fg_raw_tex,
            img_w, img_h, fw, fh,
            self.offset_x, self.offset_y, self.scale, self.rotation,
            blend_mode=mode, opacity=opacity, srgb_out=1)
        gpu.state.blend_set('NONE')
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect
        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        self._load_fg_image()
        self._ensure_top_tex()

        self._draw_composite_live(offset_x, offset_y, disp_w, disp_h)

        quad = self._get_quad_verts()

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
        hw = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 2.0
        hh = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 2.0
        ca, sa = math.cos(self.rotation), math.sin(self.rotation)
        def _rot_pt(px, py):
            dx, dy = px - cx, py - cy
            return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

        rot_offset = 20
        rot_pts = [
            _rot_pt(cx - hw - rot_offset, cy - hh - rot_offset),
            _rot_pt(cx + hw + rot_offset, cy - hh - rot_offset),
            _rot_pt(cx + hw + rot_offset, cy + hh + rot_offset),
            _rot_pt(cx - hw - rot_offset, cy + hh + rot_offset),
        ]

        rot_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        rot_shader.bind()
        rot_shader.uniform_float("color", (0.2, 0.6, 1.0, 1.0))
        segments = 16
        circle_verts = [(math.cos(2.0 * math.pi * i / segments), math.sin(2.0 * math.pi * i / segments)) for i in range(segments)]
        for px, py in rot_pts:
            rv = [(px + v[0] * 6, py + v[1] * 6) for v in circle_verts]
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
        hw = math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / 2.0
        hh = math.hypot(p2[0] - p1[0], p2[1] - p1[1]) / 2.0

        ca, sa = math.cos(-self.rotation), math.sin(-self.rotation)
        dx, dy = mx - cx, my - cy
        lmx = cx + dx * ca - dy * sa
        lmy = cy + dx * sa + dy * ca

        rot_offset = 20
        rot_hit_size = 15
        rot_local_pts = [
            (cx - hw - rot_offset, cy - hh - rot_offset),
            (cx + hw + rot_offset, cy - hh - rot_offset),
            (cx + hw + rot_offset, cy + hh + rot_offset),
            (cx - hw - rot_offset, cy + hh + rot_offset),
        ]
        for rx, ry in rot_local_pts:
            if abs(lmx - rx) < rot_hit_size and abs(lmy - ry) < rot_hit_size:
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
            props = bpy.context.scene.image_editor_tools
            mode = getattr(props, 'place_img_mode', 'MIX')
            opacity = getattr(props, 'place_img_opacity', 1.0)
            result = giu.composite_transform_to_npimg(
                self.original_image, self.fg_image,
                self.offset_x, self.offset_y, self.scale, self.rotation,
                blend_mode=mode, opacity=opacity)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            print(f"[置入图像] 应用失败: {e}")
        finally:
            self.cleanup()

    def save_as_copy(self):
        try:
            from ..utils.np_img_utils import npimg_2_blimg
            self._load_fg_image()
            if self.fg_image is None:
                self.cleanup()
                return
            props = bpy.context.scene.image_editor_tools
            mode = getattr(props, 'place_img_mode', 'MIX')
            opacity = getattr(props, 'place_img_opacity', 1.0)
            result = giu.composite_transform_to_npimg(
                self.original_image, self.fg_image,
                self.offset_x, self.offset_y, self.scale, self.rotation,
                blend_mode=mode, opacity=opacity)
            new_name = self.original_image.name + "_placed"
            new_img = npimg_2_blimg(result, new_name, False)
            bpy.context.space_data.image = new_img
            bpy.ops.ed.undo_push(message="置入图像另存")
        except Exception as e:
            print(f"[置入图像] 另存失败: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        for attr in ('_fg_raw_tex', '_bg_raw_tex'):
            if getattr(self, attr, None) is not None:
                try:
                    delattr(self, attr)
                except Exception:
                    pass
                setattr(self, attr, None)
        self._fg_raw_ref = None
        super().cleanup()

    def _on_prop_update(self):
        self._load_fg_image()
        self._ensure_top_tex()

    @staticmethod
    def get_properties():
        from . import _on_warp_param_update
        from ..utils.blend_modes import BLEND_MODE_ITEMS
        return {
            'place_img_fg': bpy.props.PointerProperty(
                name="前景图", type=bpy.types.Image,
                description="用于置入叠加的前景图像",
                update=_on_warp_param_update,
            ),
            'place_img_mode': bpy.props.EnumProperty(
                name="混合模式",
                items=BLEND_MODE_ITEMS,
                default='MIX',
                update=_on_warp_param_update,
            ),
            'place_img_opacity': bpy.props.FloatProperty(
                name="不透明度", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_warp_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        row = layout.row(align=True)
        row.prop(props, "place_img_fg", text="前景图")
        op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
        op.target_prop = "place_img_fg"
        if props.place_img_fg is not None:
            layout.prop(props, "place_img_mode", text="模式")
            layout.prop(props, "place_img_opacity", text="不透明度", slider=True)
        else:
            layout.label(text="请选择一幅前景图", icon='INFO')
        layout.separator()


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

import bpy
import gpu
import blf
import math
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl
from ..utils import gpu_img_utils as giu
from .base_op import WarpModalBase


TEMP_IMAGE_NAME = "._place_text_temp"
COMPOSITE_IMAGE_NAME = "._place_text_composite"


class PlaceTextEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):
        self._text_image = None
        self._text_tex = None
        self._text_params_hash = None
        self._bg_raw_tex = None
        self._text_raw_tex = None

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

        super().__init__(context, image)
        self._ensure_text_tex()

    def _get_props(self):
        return bpy.context.scene.image_editor_tools

    def _make_text_params_hash(self):
        props = self._get_props()
        return (
            getattr(props, 'place_text_content', ''),
            getattr(props, 'place_text_font_path', ''),
            getattr(props, 'place_text_font_size', 120),
            tuple(getattr(props, 'place_text_color', (1.0, 1.0, 1.0, 1.0))),
            getattr(props, 'place_text_letter_spacing', 0.0),
            getattr(props, 'place_text_italic_angle', 0.0),
            getattr(props, 'place_text_padding', 0.2),
            getattr(props, 'place_text_mode', 'MIX'),
            getattr(props, 'place_text_opacity', 1.0),
        )

    def _generate_text_image(self):
        props = self._get_props()
        text = getattr(props, 'place_text_content', '')
        font_path = getattr(props, 'place_text_font_path', '')
        font_size = getattr(props, 'place_text_font_size', 120)
        color = getattr(props, 'place_text_color', (1.0, 1.0, 1.0, 1.0))
        letter_spacing = getattr(props, 'place_text_letter_spacing', 0.0)
        italic_angle = getattr(props, 'place_text_italic_angle', 0.0)
        padding = getattr(props, 'place_text_padding', 0.2)

        return giu.gpu_text_to_image(
            text, TEMP_IMAGE_NAME,
            font_path=font_path,
            font_size=font_size,
            color=tuple(color),
            letter_spacing=letter_spacing,
            italic_angle=italic_angle,
            padding=padding,
        )

    def _ensure_text_tex(self):
        current_hash = self._make_text_params_hash()
        if current_hash == self._text_params_hash and self._text_tex is not None:
            return
        self._text_params_hash = current_hash
        self._text_image = self._generate_text_image()
        if self._text_tex is not None:
            try:
                del self._text_tex
            except Exception:
                pass
        self._text_tex = gpu.texture.from_image(self._text_image)
        try:
            self._text_raw_tex = giu.tex_from_image_raw(self._text_image)
        except Exception:
            self._text_raw_tex = None

    def _get_quad_verts(self):
        offset_x, offset_y, disp_w, disp_h = self._img_rect
        if disp_w <= 0 or disp_h <= 0:
            return [(0, 0)] * 4

        img_w, img_h = self.original_image.size
        fw, fh = (img_w, img_h)
        if self._text_image is not None:
            fw, fh = self._text_image.size
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

        self._ensure_text_tex()

        if self._drag_mode != 'NONE':
            self._draw_live_overlay(offset_x, offset_y, disp_w, disp_h)
        else:
            self._draw_composite(offset_x, offset_y, disp_w, disp_h)

        quad = self._get_quad_verts()
        self._draw_handles(quad)

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        s = pget_tmpl("置入文字 | 拖拽移动 | 角点缩放 | 旋转 ({angle}°) [Shift吸附] | Enter应用 | Esc取消",
                      angle=f"{math.degrees(self.rotation):.1f}")
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, s)

    def _draw_live_overlay(self, offset_x, offset_y, disp_w, disp_h):
        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

        quad = self._get_quad_verts()
        if self._text_tex is not None:
            gpu.state.blend_set('ALPHA')
            shader.bind()
            shader.uniform_sampler("image", self._text_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": quad,
                "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
            }).draw(shader)
            gpu.state.blend_set('NONE')

    def _draw_composite(self, offset_x, offset_y, disp_w, disp_h):
        # 用裸值纹理 + srgb_out=1 在屏幕做单 pass 合成，混合空间与落地一致
        # (线性光等阈值类混合模式预览=应用)。落地/另存走 composite_transform_to_npimg。
        if self._text_raw_tex is None or self._cached_orig_tex is None:
            self._draw_live_overlay(offset_x, offset_y, disp_w, disp_h)
            return

        if self._bg_raw_tex is None:
            self._bg_raw_tex = giu.tex_from_image_raw(self.original_image)

        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        img_w, img_h = self.original_image.size
        if self._text_image is not None:
            fw, fh = self._text_image.size
        else:
            fw, fh = img_w, img_h

        props = self._get_props()
        mode = getattr(props, 'place_text_mode', 'MIX')
        opacity = getattr(props, 'place_text_opacity', 1.0)

        shader = giu.get_composite_shader()
        giu.bind_composite(
            shader, self._bg_raw_tex, self._text_raw_tex,
            img_w, img_h, fw, fh,
            self.offset_x, self.offset_y, self.scale, self.rotation,
            blend_mode=mode, opacity=opacity, srgb_out=1)
        gpu.state.blend_set('NONE')
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

    def _draw_handles(self, quad):
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

        cx = sum(p[0] for p in quad) / 4.0
        cy = sum(p[1] for p in quad) / 4.0
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
            self._ensure_text_tex()
            if self._text_image is None:
                self.cleanup()
                return
            props = self._get_props()
            mode = getattr(props, 'place_text_mode', 'MIX')
            opacity = getattr(props, 'place_text_opacity', 1.0)
            result = giu.composite_transform_to_npimg(
                self.original_image, self._text_image,
                self.offset_x, self.offset_y, self.scale, self.rotation,
                blend_mode=mode, opacity=opacity)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            print(f"[置入文字] 应用失败: {e}")
        finally:
            self.cleanup()

    def save_as_copy(self):
        try:
            from ..utils.np_img_utils import npimg_2_blimg
            self._ensure_text_tex()
            if self._text_image is None:
                self.cleanup()
                return
            props = self._get_props()
            mode = getattr(props, 'place_text_mode', 'MIX')
            opacity = getattr(props, 'place_text_opacity', 1.0)
            result = giu.composite_transform_to_npimg(
                self.original_image, self._text_image,
                self.offset_x, self.offset_y, self.scale, self.rotation,
                blend_mode=mode, opacity=opacity)
            new_name = self.original_image.name + "_text"
            new_img = npimg_2_blimg(result, new_name, False)
            bpy.context.space_data.image = new_img
            bpy.ops.ed.undo_push(message="置入文字另存")
        except Exception as e:
            print(f"[置入文字] 另存失败: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self._text_tex is not None:
            try:
                del self._text_tex
            except Exception:
                pass
            self._text_tex = None
        if self._text_image is not None:
            try:
                bpy.data.images.remove(self._text_image)
            except Exception:
                pass
            self._text_image = None
        self._text_params_hash = None
        for attr in ('_bg_raw_tex', '_text_raw_tex'):
            if getattr(self, attr, None) is not None:
                try:
                    delattr(self, attr)
                except Exception:
                    pass
                setattr(self, attr, None)
        super().cleanup()

    def _on_prop_update(self):
        self._text_params_hash = None
        self._ensure_text_tex()

    @staticmethod
    def get_properties():
        from . import _on_warp_param_update
        from ..utils.blend_modes import BLEND_MODE_ITEMS
        return {
            'place_text_content': bpy.props.StringProperty(
                name="文字内容", default="文字",
                description="要置入的文字内容",
                update=_on_warp_param_update,
            ),
            'place_text_font_path': bpy.props.StringProperty(
                name="字体文件",
                subtype='FILE_PATH',
                default="",
                description="自定义字体文件路径，留空使用默认字体",
                update=_on_warp_param_update,
            ),
            'place_text_font_size': bpy.props.IntProperty(
                name="字号", default=120, min=10, max=2000,
                description="文字大小（像素）",
                update=_on_warp_param_update,
            ),
            'place_text_color': bpy.props.FloatVectorProperty(
                name="文字颜色",
                subtype='COLOR',
                size=4,
                default=(1.0, 1.0, 1.0, 1.0),
                min=0.0, max=1.0,
                description="文字颜色 (RGBA)",
                update=_on_warp_param_update,
            ),
            'place_text_letter_spacing': bpy.props.FloatProperty(
                name="字间距", default=0.0, min=-50.0, max=500.0,
                description="字符之间的额外间距（像素）",
                update=_on_warp_param_update,
            ),
            'place_text_italic_angle': bpy.props.FloatProperty(
                name="倾斜角度", default=0.0, min=-45.0, max=45.0,
                description="文字倾斜角度（度）",
                update=_on_warp_param_update,
            ),
            'place_text_padding': bpy.props.FloatProperty(
                name="出血", default=0.2, min=0.0, max=5.0,
                description="基于字高的出血比例，实际出血 = 系数 × 字号",
                update=_on_warp_param_update,
            ),
            'place_text_mode': bpy.props.EnumProperty(
                name="混合模式",
                items=BLEND_MODE_ITEMS,
                default='MIX',
                update=_on_warp_param_update,
            ),
            'place_text_opacity': bpy.props.FloatProperty(
                name="不透明度", default=1.0, min=0.0, max=1.0,
                soft_min=0.0, soft_max=1.0, subtype='FACTOR',
                update=_on_warp_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "place_text_content", text="文字")
        layout.prop(props, "place_text_font_path", text="字体")
        row = layout.row(align=True)
        row.prop(props, "place_text_font_size", text="字号")
        row.prop(props, "place_text_color", text="")
        row = layout.row(align=True)
        row.prop(props, "place_text_letter_spacing", text="字间距")
        row.prop(props, "place_text_italic_angle", text="倾斜")
        layout.prop(props, "place_text_padding", text="出血", slider=True)
        if props.place_text_content:
            layout.prop(props, "place_text_mode", text="模式")
            layout.prop(props, "place_text_opacity", text="不透明度", slider=True)
        else:
            layout.label(text="请输入文字内容", icon='INFO')
        layout.separator()


class IMAGE_OT_place_text_modal(WarpModalBase):
    bl_idname = "image_editor_tools.place_text_modal"
    bl_label = "置入文字"
    engine_class = PlaceTextEngine
    tool_key = 'warp:置入文字'
    tool_label = '置入文字'
    status_text = "置入文字: 拖拽移动 | 角点缩放 | 旋转 [Shift吸附] | Enter应用 | Esc取消"
    _drag_attr = '_drag_mode'
    _drag_none = 'NONE'

    def modal(self, context, event):
        if event.type in {'RET', 'NUMPAD_ENTER'}:
            engine = self.engine_class._active_instance
            if engine is not None:
                mx, my = event.mouse_region_x, event.mouse_region_y
                quad = engine._get_quad_verts()
                if not engine._point_in_quad(mx, my, quad):
                    return {'PASS_THROUGH'}
        return super().modal(context, event)

    def _custom_keys(self, context, event, engine):
        if event.type == 'F' and event.value == 'PRESS':
            engine.offset_x = engine.original_image.size[0] / 2.0
            engine.offset_y = engine.original_image.size[1] / 2.0
            engine.scale = 0.5
            engine.rotation = 0.0
            return True
        return False

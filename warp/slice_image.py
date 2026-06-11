import bpy
import gpu
import blf
import math
import numpy as np
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl
from ..np_img_utils import blimg_2_npimg, npimg_2_blimg


SNAP_STEP = 0.1
HIT_RADIUS = 14


def _redistribute_lines(count):
    if count <= 0:
        return []
    step = 1.0 / (count + 1)
    return [step * (i + 1) for i in range(count)]


class SliceImageEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):
        self._cols = 2
        self._rows = 2
        self.v_lines = _redistribute_lines(self._cols)
        self.h_lines = _redistribute_lines(self._rows)
        self._drag_line = None
        self._sync_from_props()
        super().__init__(context, image)

    def _sync_from_props(self):
        props = bpy.context.scene.image_editor_tools
        cols = getattr(props, 'slice_cols', 2)
        rows = getattr(props, 'slice_rows', 2)
        if cols != self._cols:
            self._cols = cols
            self.v_lines = _redistribute_lines(cols)
        if rows != self._rows:
            self._rows = rows
            self.h_lines = _redistribute_lines(rows)

    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect
        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        self._sync_from_props()

        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

        line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader.bind()
        line_shader.uniform_float("color", (1.0, 0.2, 0.2, 0.8))

        for pos in self.v_lines:
            px = x0 + pos * disp_w
            batch_for_shader(line_shader, 'LINES', {
                "pos": [(px, y0), (px, y1)],
            }).draw(line_shader)

        line_shader.uniform_float("color", (0.2, 0.6, 1.0, 0.8))

        for pos in self.h_lines:
            py = y0 + pos * disp_h
            batch_for_shader(line_shader, 'LINES', {
                "pos": [(x0, py), (x1, py)],
            }).draw(line_shader)

        handle_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        handle_shader.bind()
        h_size = 4

        for pos in self.v_lines:
            px = x0 + pos * disp_w
            handle_shader.uniform_float("color", (1.0, 0.4, 0.4, 1.0))
            for py in (y0, y1):
                batch_for_shader(handle_shader, 'TRI_FAN', {
                    "pos": [(px - h_size, py - h_size), (px + h_size, py - h_size),
                            (px + h_size, py + h_size), (px - h_size, py + h_size)]
                }).draw(handle_shader)

        for pos in self.h_lines:
            py = y0 + pos * disp_h
            handle_shader.uniform_float("color", (0.4, 0.6, 1.0, 1.0))
            for px in (x0, x1):
                batch_for_shader(handle_shader, 'TRI_FAN', {
                    "pos": [(px - h_size, py - h_size), (px + h_size, py - h_size),
                            (px + h_size, py + h_size), (px - h_size, py + h_size)]
                }).draw(handle_shader)

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        s = pget_tmpl("切分图像 | 拖拽分割线 | Shift 吸附 | 纵:{cols} 横:{rows} | Enter 应用 | Esc 取消",
                      cols=str(self._cols), rows=str(self._rows))
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, s)

    def handle_mouse_press(self, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        if dw <= 0 or dh <= 0:
            return False

        self._drag_line = None
        best_dist = HIT_RADIUS

        for i, pos in enumerate(self.v_lines):
            px = ox + pos * dw
            dist = abs(mx - px)
            if dist < best_dist:
                best_dist = dist
                self._drag_line = ('v', i)

        for i, pos in enumerate(self.h_lines):
            py = oy + pos * dh
            dist = abs(my - py)
            if dist < best_dist:
                best_dist = dist
                self._drag_line = ('h', i)

        return self._drag_line is not None

    def handle_mouse_move(self, event):
        if self._drag_line is None:
            return False
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        if dw <= 0 or dh <= 0:
            return False

        axis, index = self._drag_line
        if axis == 'v':
            raw = (mx - ox) / dw
        else:
            raw = (my - oy) / dh

        if event.shift:
            raw = round(raw / SNAP_STEP) * SNAP_STEP

        raw = max(0.02, min(0.98, raw))
        (self.v_lines if axis == 'v' else self.h_lines)[index] = raw
        return True

    def handle_mouse_release(self, event):
        self._drag_line = None

    def save_as_copy(self):
        self._do_slice(save_mode='copy')

    def apply_to_original(self):
        self._do_slice(save_mode='copy')

    def _do_slice(self, save_mode='copy'):
        try:
            np_array = blimg_2_npimg(self.original_image)
            h, w = np_array.shape[:2]

            v_positions = [0.0] + sorted(self.v_lines) + [1.0]
            h_positions = [0.0] + sorted(self.h_lines) + [1.0]

            v_px = [max(0, min(w, int(round(p * w)))) for p in v_positions]
            h_px = [max(0, min(h, int(round(p * h)))) for p in h_positions]

            base_name = self.original_image.name
            idx = 1
            for row in range(len(h_px) - 1):
                y0, y1 = h_px[row], h_px[row + 1]
                if y1 <= y0:
                    continue
                for col in range(len(v_px) - 1):
                    x0, x1 = v_px[col], v_px[col + 1]
                    if x1 <= x0:
                        continue
                    cell = np_array[y0:y1, x0:x1, :].copy()
                    cell_name = f"{base_name}_{idx:02d}"
                    new_img = npimg_2_blimg(cell, cell_name, True)
                    bpy.context.space_data.image = new_img
                    idx += 1

            bpy.ops.ed.undo_push(message="切分图像")
        except Exception as e:
            print(f"[切分图像] 失败: {e}")
        finally:
            self.cleanup()


class IMAGE_OT_slice_image_modal(bpy.types.Operator):
    bl_idname = "image_editor_tools.slice_image_modal"
    bl_label = "切分图像"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and context.space_data.type == 'IMAGE_EDITOR'
            and context.space_data.image is not None
        )

    def modal(self, context, event):
        engine = SliceImageEngine._active_instance
        if not engine or engine.should_exit:
            return self._finish(context)
        context.area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            engine.cleanup()
            return self._finish(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'}:
            engine.apply_to_original()
            return self._finish(context)

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if engine.handle_mouse_press(event):
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            elif event.value == 'RELEASE':
                engine.handle_mouse_release(event)
                return {'PASS_THROUGH'}

        elif event.type == 'MOUSEMOVE':
            if engine._drag_line is not None:
                engine.handle_mouse_move(event)
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.space_data is None or context.space_data.type != 'IMAGE_EDITOR' or not context.space_data.image:
            self.report({'WARNING'}, "请在图像编辑器中打开一张图片")
            return {'CANCELLED'}

        if SliceImageEngine._active_instance:
            SliceImageEngine._active_instance.cleanup()

        self._prev_ui_mode = str(context.space_data.ui_mode)
        if context.area.ui_type != 'IMAGE_EDITOR':
            context.area.ui_type = 'IMAGE_EDITOR'
        context.space_data.ui_mode = 'VIEW'

        bpy.ops.ed.undo_push(message="切分图像")
        state.current_tool = 'warp:切分图像'
        SliceImageEngine(context, context.space_data.image)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("切分图像: 拖拽分割线 | Shift 吸附 | Enter 应用 | Esc 取消")
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        state.current_tool = 'NONE'
        prev = getattr(self, '_prev_ui_mode', None)
        if prev is not None:
            try:
                area = context.area
                if area is not None and area.type == 'IMAGE_EDITOR':
                    sp = area.spaces.active
                    if sp is not None:
                        sp.ui_mode = prev
            except Exception:
                pass
        context.workspace.status_text_set(None)
        return {'FINISHED'}

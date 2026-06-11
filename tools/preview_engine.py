import bpy
import gpu
import blf
import time
import numpy as np
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl
from ..utils.np_img_utils import blimg_2_npimg, npimg_2_blimg, np_clamp_image_size

PREVIEW_MAX_SIDE = 1024
TEMP_IMAGE_NAME = "._tool_preview_temp"


class PreviewEngine(BaseEngine):
    _engine_type = 'tool'

    def __init__(self, context, image, tool):
        self.tool = tool
        self.preview_np = None
        self.preview_bl_image = None
        self.compare_x = 0.5
        self.display_mode = 0
        self._dragging = False
        self._last_click_time = 0.0
        self._cached_preview_tex = None
        self._cached_preview_image = None
        super().__init__(context, image)

    def _on_init_before_draw(self):
        self._init_preview()

    def _init_preview(self):
        try:
            full_np = blimg_2_npimg(self.original_image)
            self.preview_np = np_clamp_image_size(full_np, PREVIEW_MAX_SIDE)
            self._process_preview()
        except Exception as e:
            self._report_error(f"预览初始化失败: {e}")

    def _process_preview(self):
        if self.preview_np is None:
            return
        try:
            props = bpy.context.scene.image_editor_tools
            result = self.tool.process(self.preview_np, props)

            if self.preview_bl_image is not None:
                pw, ph = self.preview_bl_image.size
                rh, rw = result.shape[:2]
                if rw != pw or rh != ph:
                    name = self.preview_bl_image.name
                    if name in bpy.data.images:
                        bpy.data.images.remove(bpy.data.images[name])
                    self.preview_bl_image = None

            if self.preview_bl_image is None:
                self.preview_bl_image = npimg_2_blimg(result, TEMP_IMAGE_NAME, True)
            else:
                self.preview_bl_image.pixels.foreach_set(result.ravel())
                self.preview_bl_image.update()
            self._cached_preview_image = None
        except Exception as e:
            self._report_error(f"预览处理失败: {e}")

    def _draw(self):
        if self.preview_bl_image is None:
            return
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect
        self._draw_in_region(offset_x, offset_y, disp_w, disp_h)

    def _draw_in_region(self, offset_x, offset_y, disp_w, disp_h):
        preview_img = self.preview_bl_image

        if self._cached_preview_image is not preview_img:
            self._cached_preview_tex = gpu.texture.from_image(preview_img)
            self._cached_preview_image = preview_img

        x0 = offset_x
        x1 = offset_x + disp_w
        y0 = offset_y
        y1 = offset_y + disp_h

        bg_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        bg_shader.bind()
        bg_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        batch_for_shader(bg_shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
        }).draw(bg_shader)

        shader = state.get_display_shader()
        shader.bind()

        if self.display_mode < 2:
            if self.display_mode == 0:
                left_tex = self._cached_orig_tex
                right_tex = self._cached_preview_tex
            else:
                left_tex = self._cached_preview_tex
                right_tex = self._cached_orig_tex

            split = self.compare_x
            xm = offset_x + split * disp_w

            shader.uniform_sampler("image", left_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": [(x0, y0), (xm, y0), (xm, y1), (x0, y1)],
                "texCoord": [(0.0, 0.0), (split, 0.0), (split, 1.0), (0.0, 1.0)],
            }).draw(shader)

            shader.uniform_sampler("image", right_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": [(xm, y0), (x1, y0), (x1, y1), (xm, y1)],
                "texCoord": [(split, 0.0), (1.0, 0.0), (1.0, 1.0), (split, 1.0)],
            }).draw(shader)

            line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
            line_shader.bind()
            line_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
            lw = 2
            batch_for_shader(line_shader, 'TRI_FAN', {
                "pos": [
                    (xm - lw, y0),
                    (xm + lw, y0),
                    (xm + lw, y1),
                    (xm - lw, y1),
                ],
            }).draw(line_shader)
        else:
            if self.display_mode == 2:
                full_tex = self._cached_orig_tex
            else:
                full_tex = self._cached_preview_tex

            shader.uniform_sampler("image", full_tex)
            batch_for_shader(shader, 'TRI_FAN', {
                "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
                "texCoord": [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)],
            }).draw(shader)

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        mode_texts = ["原图 | 预览图", "预览图 | 原图", "仅原图", "仅预览图"]
        text = pget_tmpl("{mode}(双击切换)", mode=mode_texts[self.display_mode])
        text_x = x0
        text_y = y0 - 25
        blf.position(font_id, text_x, text_y, 0)
        blf.draw(font_id, text)

    def handle_mouse_press(self, event):
        ox, oy, dw, dh = self._img_rect
        if dw <= 0 or dh <= 0:
            return False

        mx = event.mouse_region_x
        my = event.mouse_region_y

        in_image = (ox <= mx <= ox + dw) and (oy <= my <= oy + dh)
        if not in_image:
            return False

        if self.display_mode < 2:
            line_x = ox + self.compare_x * dw
            if abs(mx - line_x) < 24:
                self._dragging = True
                return True

        now = time.time()
        if now - self._last_click_time < 0.3:
            self.display_mode = (self.display_mode + 1) % 4
            self._last_click_time = 0.0
            return True
        self._last_click_time = now
        return False

    def handle_mouse_move(self, event):
        if not self._dragging or self.display_mode >= 2:
            return False
        ox, oy, dw, dh = self._img_rect
        if dw <= 0 or dh <= 0:
            return False

        mx = event.mouse_region_x
        self.compare_x = max(0.05, min(0.95, (mx - ox) / dw))
        return True

    def handle_mouse_release(self, event):
        self._dragging = False

    def apply_to_original(self):
        try:
            props = bpy.context.scene.image_editor_tools
            full_np = blimg_2_npimg(self.original_image)
            result = self.tool.process(full_np, props)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            self._report_error(f"保存失败: {e}")
        finally:
            self.cleanup()

    def apply_to_new(self):
        try:
            props = bpy.context.scene.image_editor_tools
            full_np = blimg_2_npimg(self.original_image)
            result = self.tool.process(full_np, props)
            new_name = self.original_image.name + "_" + self.tool.tool_id
            new_img = npimg_2_blimg(result, new_name, True)
            bpy.context.space_data.image = new_img
        except Exception as e:
            self._report_error(f"另存失败: {e}")
        finally:
            self.cleanup()

    def cleanup(self):
        if self.preview_bl_image is not None:
            name = self.preview_bl_image.name
            if name in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[name])
            self.preview_bl_image = None
        self.preview_np = None
        self._cached_preview_tex = None
        self._cached_preview_image = None
        super().cleanup()

import bpy
import gpu
import blf
import math
import numpy as np
import mathutils
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl


class MeshWarpEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):
        self.grid_size = 4

        self.src_pts = self._generate_grid()
        self.tgt_pts = self._generate_grid()

        self.preview_res = 32
        self.dense_uvs = []
        self.dense_indices = []

        for y in range(self.preview_res):
            for x in range(self.preview_res):
                u = x / (self.preview_res - 1)
                v = y / (self.preview_res - 1)
                self.dense_uvs.append((u, v))

        for y in range(self.preview_res - 1):
            for x in range(self.preview_res - 1):
                i0 = y * self.preview_res + x
                i1 = i0 + 1
                i2 = i0 + self.preview_res
                i3 = i2 + 1
                self.dense_indices.extend([(i0, i1, i2), (i1, i3, i2)])

        self.dense_uvs_np = np.array(self.dense_uvs, dtype=np.float64)
        self.dense_deformed_np = self.dense_uvs_np.copy()

        self.padding_modes = ['TRANSPARENT', 'WHITE', 'BLACK']
        self.padding_mode_idx = 0
        self.padding_mode = self.padding_modes[self.padding_mode_idx]

        self._drag_idx = -1

        super().__init__(context, image)

    def _generate_grid(self):
        pts = []
        for y in range(self.grid_size):
            for x in range(self.grid_size):
                nx = x / (self.grid_size - 1)
                ny = y / (self.grid_size - 1)
                pts.append([0.1 + nx * 0.8, 0.1 + ny * 0.8])
        return pts

    def update_preview_mesh(self):
        Q = np.array(self.src_pts, dtype=np.float64)
        P = np.array(self.tgt_pts, dtype=np.float64)
        N = len(Q)

        diff = Q[:, np.newaxis, :] - Q[np.newaxis, :, :]
        r2 = np.sum(diff ** 2, axis=-1)
        K = np.zeros_like(r2)
        mask = r2 > 0
        K[mask] = r2[mask] * np.log(r2[mask])

        L = np.zeros((N + 3, N + 3), dtype=np.float64)
        L[:N, :N] = K
        L[:N, N:N + 2] = Q
        L[:N, N + 2] = 1.0
        L[N:N + 2, :N] = Q.T
        L[N + 2, :N] = 1.0

        V = np.zeros((N + 3, 2), dtype=np.float64)
        V[:N, :] = P

        try:
            W = np.linalg.solve(L, V)
        except np.linalg.LinAlgError:
            return

        w_weights = W[:N, :]
        a_affine = W[N:N + 2, :]
        a_trans = W[N + 2, :]

        diff_eval = self.dense_uvs_np[:, np.newaxis, :] - Q[np.newaxis, :, :]
        r2_eval = np.sum(diff_eval ** 2, axis=-1)
        K_eval = np.zeros_like(r2_eval)
        mask_eval = r2_eval > 0
        K_eval[mask_eval] = r2_eval[mask_eval] * np.log(r2_eval[mask_eval])

        self.dense_deformed_np = np.dot(K_eval, w_weights) + np.dot(self.dense_uvs_np, a_affine) + a_trans

    def cycle_padding_mode(self):
        self.padding_mode_idx = (self.padding_mode_idx + 1) % len(self.padding_modes)
        self.padding_mode = self.padding_modes[self.padding_mode_idx]

    def reset_transform(self):
        self.tgt_pts = self._generate_grid()
        self.update_preview_mesh()

    def handle_mouse_press(self, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        hit_radius = 15
        self._drag_idx = -1

        for i, pt in enumerate(self.tgt_pts):
            px = ox + pt[0] * dw
            py = oy + pt[1] * dh
            if math.hypot(mx - px, my - py) < hit_radius:
                self._drag_idx = i
                break
        return self._drag_idx != -1

    def handle_mouse_move(self, event):
        if self._drag_idx == -1:
            return False
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect

        if dw > 0 and dh > 0:
            self.tgt_pts[self._drag_idx][0] = (mx - ox) / dw
            self.tgt_pts[self._drag_idx][1] = (my - oy) / dh
            self.update_preview_mesh()
        return True

    def handle_mouse_release(self, event):
        self._drag_idx = -1

    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect

        x0, y0 = offset_x, offset_y

        screen_verts = []
        for pt in self.dense_deformed_np:
            screen_verts.append((x0 + pt[0] * disp_w, y0 + pt[1] * disp_h))

        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRIS', {
            "pos": screen_verts,
            "texCoord": self.dense_uvs,
        }, indices=self.dense_indices).draw(shader)

        line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader.bind()
        line_shader.uniform_float("color", (1.0, 1.0, 1.0, 0.5))

        lines = []
        gs = self.grid_size

        def get_pt(idx):
            pt = self.tgt_pts[idx]
            return (x0 + pt[0] * disp_w, y0 + pt[1] * disp_h)

        for y in range(gs):
            for x in range(gs):
                idx = y * gs + x
                p_curr = get_pt(idx)
                if x < gs - 1:
                    lines.extend([p_curr, get_pt(idx + 1)])
                if y < gs - 1:
                    lines.extend([p_curr, get_pt(idx + gs)])

        if lines:
            batch_for_shader(line_shader, 'LINES', {"pos": lines}).draw(line_shader)

        point_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        point_shader.bind()

        h_size = 4
        for i in range(len(self.tgt_pts)):
            px, py = get_pt(i)
            s = h_size * 1.5 if i == self._drag_idx else h_size
            if i == self._drag_idx:
                point_shader.uniform_float("color", (1.0, 0.2, 0.2, 1.0))
            else:
                point_shader.uniform_float("color", (1.0, 0.8, 0.2, 1.0))

            batch_for_shader(point_shader, 'TRI_FAN', {
                "pos": [(px - s, py - s), (px + s, py - s), (px + s, py + s), (px - s, py + s)]
            }).draw(point_shader)

        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        mode_names = {'TRANSPARENT': '透明', 'WHITE': '白底', 'BLACK': '黑底'}
        pad_str = mode_names.get(self.padding_mode, self.padding_mode)
        text = pget_tmpl(
            "网格变形 (GPU烘焙) | 拖拽控制点 | 填充: {pad} [P切换] | F 重置 | Enter 应用 | Esc 取消",
            pad=pad_str,
        )
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, text)

    def save_as_copy(self):
        try:
            from ..utils.np_img_utils import npimg_2_blimg
            img_w, img_h = self.original_image.size
            offscreen = gpu.types.GPUOffScreen(img_w, img_h)
            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()
                pad_color = (0.0, 0.0, 0.0, 0.0)
                if self.padding_mode == 'WHITE': pad_color = (1.0, 1.0, 1.0, 1.0)
                elif self.padding_mode == 'BLACK': pad_color = (0.0, 0.0, 0.0, 1.0)
                if hasattr(fb, "clear"): fb.clear(color=pad_color)
                else: gpu.state.clear_color_and_depth(pad_color, 1.0)
                gpu.matrix.push(); gpu.matrix.push_projection()
                proj = mathutils.Matrix.Identity(4)
                proj[0][0] = 2.0 / img_w; proj[1][1] = 2.0 / img_h
                proj[2][2] = -1.0; proj[0][3] = -1.0; proj[1][3] = -1.0; proj[2][3] = 0.0
                gpu.matrix.load_projection_matrix(proj)
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))
                verts = [(pt[0] * img_w, pt[1] * img_h) for pt in self.dense_deformed_np]
                try: shader = state.get_display_shader()
                except Exception: shader = gpu.shader.from_builtin('2D_IMAGE')
                shader.bind(); shader.uniform_sampler("image", self._cached_orig_tex)
                batch = batch_for_shader(shader, 'TRIS', {"pos": verts, "texCoord": self.dense_uvs}, indices=self.dense_indices)
                gpu.state.blend_set('ALPHA'); batch.draw(shader); gpu.state.blend_set('NONE')
                buffer = fb.read_color(0, 0, img_w, img_h, 4, 0, 'FLOAT')
                gpu.matrix.pop_projection(); gpu.matrix.pop()
            offscreen.free()
            pixel_data = np.array(buffer.to_list(), dtype=np.float32).ravel()
            result = pixel_data.reshape(img_h, img_w, 4)
            if not state._HAS_SRGB_SHADER:
                from ..utils.np_img_utils import np_linear_to_srgb
                result = np_linear_to_srgb(result)
            new_name = self.original_image.name + "_warped"
            npimg_2_blimg(result, new_name, True)
            bpy.ops.ed.undo_push(message="贝塞尔扭曲另存")
        except Exception as e:
            print(f"[贝塞尔扭曲] 另存失败: {e}")

    def apply_to_original(self):
        try:
            img_w, img_h = self.original_image.size
            offscreen = gpu.types.GPUOffScreen(img_w, img_h)

            with offscreen.bind():
                fb = gpu.state.active_framebuffer_get()

                pad_color = (0.0, 0.0, 0.0, 0.0)
                if self.padding_mode == 'WHITE':
                    pad_color = (1.0, 1.0, 1.0, 1.0)
                elif self.padding_mode == 'BLACK':
                    pad_color = (0.0, 0.0, 0.0, 1.0)

                if hasattr(fb, "clear"):
                    fb.clear(color=pad_color)
                else:
                    gpu.state.clear_color_and_depth(pad_color, 1.0)

                gpu.matrix.push()
                gpu.matrix.push_projection()

                proj = mathutils.Matrix.Identity(4)
                proj[0][0] = 2.0 / img_w
                proj[1][1] = 2.0 / img_h
                proj[2][2] = -1.0
                proj[0][3] = -1.0
                proj[1][3] = -1.0
                proj[2][3] = 0.0
                gpu.matrix.load_projection_matrix(proj)
                gpu.matrix.load_matrix(mathutils.Matrix.Identity(4))

                verts = [(pt[0] * img_w, pt[1] * img_h) for pt in self.dense_deformed_np]

                try:
                    shader = state.get_offscreen_shader()
                except Exception:
                    shader = gpu.shader.from_builtin('2D_IMAGE')

                shader.bind()
                shader.uniform_sampler("image", self._cached_orig_tex)

                batch = batch_for_shader(shader, 'TRIS', {
                    "pos": verts,
                    "texCoord": self.dense_uvs
                }, indices=self.dense_indices)

                gpu.state.blend_set('ALPHA')
                batch.draw(shader)
                gpu.state.blend_set('NONE')

                buffer = fb.read_color(0, 0, img_w, img_h, 4, 0, 'FLOAT')

                gpu.matrix.pop_projection()
                gpu.matrix.pop()

            offscreen.free()

            pixel_data = np.array(buffer.to_list(), dtype=np.float32).ravel()
            result = pixel_data.reshape(img_h, img_w, 4)
            if not state._HAS_SRGB_SHADER:
                from ..utils.np_img_utils import np_linear_to_srgb
                result = np_linear_to_srgb(result)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()

        except Exception as e:
            print(f"[贝塞尔扭曲] 应用失败: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.cleanup()

    def cleanup(self):
        super().cleanup()


class IMAGE_OT_mesh_warp_modal(bpy.types.Operator):
    bl_idname = "image_editor_tools.mesh_warp_modal"
    bl_label = "贝塞尔扭曲"
    bl_options = {'REGISTER', 'UNDO'}

    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and context.space_data.type == 'IMAGE_EDITOR'
            and context.space_data.image is not None
        )

    def modal(self, context, event):
        engine = MeshWarpEngine._active_instance
        if not engine or engine.should_exit:
            return self._finish(context)
        context.area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            engine.cleanup()
            return self._finish(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'}:
            engine.apply_to_original()
            return self._finish(context)

        if event.type == 'P' and event.value == 'PRESS':
            engine.cycle_padding_mode()
            return {'RUNNING_MODAL'}
        if event.type == 'F' and event.value == 'PRESS':
            engine.reset_transform()
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                if engine.handle_mouse_press(event):
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}
            elif event.value == 'RELEASE':
                was_dragging = (engine._drag_idx != -1)
                engine.handle_mouse_release(event)
                if was_dragging:
                    return {'RUNNING_MODAL'}
                return {'PASS_THROUGH'}

        elif event.type == 'MOUSEMOVE':
            if engine._drag_idx != -1:
                engine.handle_mouse_move(event)
                return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.space_data.type != 'IMAGE_EDITOR' or not context.space_data.image:
            self.report({'WARNING'}, "请在图像编辑器中打开一张图片")
            return {'CANCELLED'}
        if MeshWarpEngine._active_instance:
            MeshWarpEngine._active_instance.cleanup()

        self._prev_ui_mode = str(context.space_data.ui_mode)
        if context.area.ui_type != 'IMAGE_EDITOR':
            context.area.ui_type = 'IMAGE_EDITOR'
        context.space_data.ui_mode = 'VIEW'

        bpy.ops.ed.undo_push(message="贝塞尔扭曲")
        state.current_tool = 'warp:贝塞尔扭曲'
        MeshWarpEngine(context, context.space_data.image)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set("贝塞尔扭曲: 拖拽控制点 | P填色 | F重置 | Enter应用 | Esc取消")
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

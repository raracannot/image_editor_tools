import bpy
import gpu
import blf
import math
import numpy as np
from gpu_extras.batch import batch_for_shader
from ..engine_base import BaseEngine
from .. import state
from ..translation import pget_tmpl
from ..utils.np_img_utils import blimg_2_npimg
from .base_op import WarpModalBase

# ==========================================
# 核心处理逻辑 (纯 Numpy 单应性矩阵求解)
# ==========================================

class PerspectiveTool:
    @staticmethod
    def process_full_homography(np_array, quad_src, quad_tgt, padding_mode='STRETCH'):
        """
        使用纯正单应性矩阵处理全图，并支持5种边缘填充模式
        """
        h, w = np_array.shape[:2]
        
        S = np.array(quad_src, dtype=np.float64)
        S[:, 0] *= w
        S[:, 1] *= h
        
        P = np.array(quad_tgt, dtype=np.float64)
        P[:, 0] *= w
        P[:, 1] *= h
        
        A = np.zeros((8, 8), dtype=np.float64)
        B = np.zeros(8, dtype=np.float64)
        
        for i in range(4):
            x, y = P[i]
            u, v = S[i]
            
            A[2*i, 0:3] = [x, y, 1]
            A[2*i, 6:8] = [-x*u, -y*u]
            B[2*i] = u
            
            A[2*i+1, 3:6] = [x, y, 1]
            A[2*i+1, 6:8] = [-x*v, -y*v]
            B[2*i+1] = v
            
        try:
            H = np.linalg.solve(A, B)
            H = np.append(H, 1.0).reshape(3, 3)
        except np.linalg.LinAlgError:
            print("[透视形变] 矩阵求解失败")
            return np_array
            
        Y, X = np.mgrid[0:h, 0:w].astype(np.float64)
        X_flat = X.ravel()
        Y_flat = Y.ravel()
        Ones = np.ones_like(X_flat)
        
        Coords = np.vstack([X_flat, Y_flat, Ones])
        Transformed = H @ Coords
        
        U_flat = Transformed[0] / Transformed[2]
        V_flat = Transformed[1] / Transformed[2]
        
        # 转换为整数坐标
        U_int = np.round(U_flat).astype(np.int32)
        V_int = np.round(V_flat).astype(np.int32)
        
        # 根据填充模式处理越界像素
        if padding_mode == 'STRETCH':
            U_px = np.clip(U_int, 0, w - 1)
            V_px = np.clip(V_int, 0, h - 1)
            result_flat = np_array[V_px, U_px]
        elif padding_mode == 'REPEAT':
            U_px = U_int % w
            V_px = V_int % h
            result_flat = np_array[V_px, U_px]
        else:
            U_px = np.clip(U_int, 0, w - 1)
            V_px = np.clip(V_int, 0, h - 1)
            result_flat = np_array[V_px, U_px].copy()
            
            # 计算越界掩码
            invalid_mask = (U_int < 0) | (U_int >= w) | (V_int < 0) | (V_int >= h)
            
            if padding_mode == 'TRANSPARENT':
                result_flat[invalid_mask] = [0.0, 0.0, 0.0, 0.0]
            elif padding_mode == 'WHITE':
                result_flat[invalid_mask] = [1.0, 1.0, 1.0, 1.0]
            elif padding_mode == 'BLACK':
                result_flat[invalid_mask] = [0.0, 0.0, 0.0, 1.0]
                
        return result_flat.reshape(h, w, 4)




# ==========================================
# 预览与交互引擎
# ==========================================

class PerspectiveEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):

        self.quad_src = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        self.quad_tgt = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]

        self.mode = 'LAYOUT'

        self.padding_modes = ['STRETCH', 'REPEAT', 'TRANSPARENT', 'WHITE', 'BLACK']
        self.padding_mode_idx = 0
        self.padding_mode = self.padding_modes[self.padding_mode_idx]

        self._drag_idx = -1

        super().__init__(context, image)

    def cycle_padding_mode(self):
        """循环切换填充模式"""
        self.padding_mode_idx = (self.padding_mode_idx + 1) % len(self.padding_modes)
        self.padding_mode = self.padding_modes[self.padding_mode_idx]
        
    def reset_transform(self):
        """F键：布局模式重置，变形模式吸附为最贴合的长方形"""
        if self.mode == 'LAYOUT':
            # 布局模式：全局重置为默认的居中方形
            self.quad_src = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
            self.quad_tgt = [[0.2, 0.2], [0.8, 0.2], [0.8, 0.8], [0.2, 0.8]]
        else:
            # 变形模式：基于中心点和平均边长，生成最平滑过渡的长方形
            pts = self.quad_tgt
            
            # 1. 计算当前几何中心
            cx = sum(p[0] for p in pts) / 4.0
            cy = sum(p[1] for p in pts) / 4.0
            
            # 2. 计算平均宽度 (上下两边的平均长度)
            w_top = math.hypot(pts[2][0] - pts[3][0], pts[2][1] - pts[3][1])
            w_bottom = math.hypot(pts[1][0] - pts[0][0], pts[1][1] - pts[0][1])
            avg_w = (w_top + w_bottom) / 2.0
            
            # 3. 计算平均高度 (左右两边的平均长度)
            h_left = math.hypot(pts[3][0] - pts[0][0], pts[3][1] - pts[0][1])
            h_right = math.hypot(pts[2][0] - pts[1][0], pts[2][1] - pts[1][1])
            avg_h = (h_left + h_right) / 2.0
            
            # 4. 生成新的正交长方形
            half_w = avg_w / 2.0
            half_h = avg_h / 2.0
            
            self.quad_tgt = [
                [cx - half_w, cy - half_h], # 左下
                [cx + half_w, cy - half_h], # 右下
                [cx + half_w, cy + half_h], # 右上
                [cx - half_w, cy + half_h]  # 左上
            ]



    def handle_mouse_press(self, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        
        hit_radius = 20.0
        self._drag_idx = -1
        
        # 根据当前模式决定检测哪个四边形
        active_quad = self.quad_src if self.mode == 'LAYOUT' else self.quad_tgt
        
        for i, (qx, qy) in enumerate(active_quad):
            px = ox + qx * dw
            py = oy + qy * dh
            dist = math.hypot(mx - px, my - py)
            if dist < hit_radius:
                self._drag_idx = i
                break
                
        return self._drag_idx != -1

    def handle_mouse_move(self, event):
        if self._drag_idx == -1:
            return False

        mx, my = event.mouse_region_x, event.mouse_region_y
        ox, oy, dw, dh = self._img_rect
        
        if dw > 0 and dh > 0:
            nx = (mx - ox) / dw
            ny = (my - oy) / dh
            
            nx = max(0.0, min(1.0, nx))
            ny = max(0.0, min(1.0, ny))
            
            if self.mode == 'LAYOUT':
                # 布局模式下，源和目标一起移动
                self.quad_src[self._drag_idx] = [nx, ny]
                self.quad_tgt[self._drag_idx] = [nx, ny]
            else:
                # 变形模式下，只移动目标
                self.quad_tgt[self._drag_idx] = [nx, ny]
            
        return True

    def handle_mouse_release(self, event):
        self._drag_idx = -1


    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect

        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        # 1. 绘制底图
        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

        # 2. 绘制全屏半透明遮罩
        mask_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        mask_shader.bind()
        mask_shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
        batch_for_shader(mask_shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        }).draw(mask_shader)

        # 3. 计算当前显示的四边形屏幕坐标 (布局模式显源，变形模式显目标)
        active_quad = self.quad_src if self.mode == 'LAYOUT' else self.quad_tgt
        screen_pts = []
        for qx, qy in active_quad:
            screen_pts.append((x0 + qx * disp_w, y0 + qy * disp_h))

        # 4. 绘制四边形内部的高亮原图 (使用 3x3 细分网格消除仿射拉伸)
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        
        GRID_N = 10 # 3x3网格，可随时调大(如10)以获得更完美的透视平滑度
        grid_pos = []
        grid_uv = []
        grid_indices = []
        
        # 双线性插值辅助函数
        def bilerp(p0, p1, p2, p3, u, v):
            x = (1-u)*(1-v)*p0[0] + u*(1-v)*p1[0] + u*v*p2[0] + (1-u)*v*p3[0]
            y = (1-u)*(1-v)*p0[1] + u*(1-v)*p1[1] + u*v*p2[1] + (1-u)*v*p3[1]
            return (x, y)
            
        # 生成顶点和UV
        for j in range(GRID_N + 1):
            v_frac = j / GRID_N
            for i in range(GRID_N + 1):
                u_frac = i / GRID_N
                grid_pos.append(bilerp(screen_pts[0], screen_pts[1], screen_pts[2], screen_pts[3], u_frac, v_frac))
                grid_uv.append(bilerp(self.quad_src[0], self.quad_src[1], self.quad_src[2], self.quad_src[3], u_frac, v_frac))
                
        # 生成三角形索引 (18个小三角形)
        for j in range(GRID_N):
            for i in range(GRID_N):
                idx0 = j * (GRID_N + 1) + i
                idx1 = idx0 + 1
                idx2 = idx0 + (GRID_N + 1) + 1
                idx3 = idx0 + (GRID_N + 1)
                grid_indices.extend([(idx0, idx1, idx2), (idx0, idx2, idx3)])
                
        # 使用 TRIS 模式绘制细分网格
        batch_for_shader(shader, 'TRIS', {
            "pos": grid_pos,
            "texCoord": grid_uv, 
        }, indices=grid_indices).draw(shader)

        # 5. 绘制连线
        line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader.bind()
        if self.mode == 'LAYOUT':
            line_shader.uniform_float("color", (0.2, 0.8, 1.0, 1.0)) # 蓝色：定义平面
        else:
            line_shader.uniform_float("color", (1.0, 0.2, 0.2, 1.0)) # 红色：扭曲变形
            
        gpu.state.line_width_set(2.0)
        lines_pts = [
            screen_pts[0], screen_pts[1],
            screen_pts[1], screen_pts[2],
            screen_pts[2], screen_pts[3],
            screen_pts[3], screen_pts[0]
        ]
        batch_for_shader(line_shader, 'LINES', {"pos": lines_pts}).draw(line_shader)
        gpu.state.line_width_set(1.0)

        # 6. 绘制四个角的手柄
        h_size = 6
        for px, py in screen_pts:
            batch_for_shader(line_shader, 'TRI_FAN', {
                "pos": [(px-h_size, py-h_size), (px+h_size, py-h_size), (px+h_size, py+h_size), (px-h_size, py+h_size)]
            }).draw(line_shader)

        # 7. 绘制文字
        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        
        mode_names = {'STRETCH': '拉伸', 'REPEAT': '重复', 'TRANSPARENT': '透明', 'WHITE': '白底', 'BLACK': '黑底'}
        pad_str = mode_names.get(self.padding_mode, self.padding_mode)
        
        mode_str = "【1. 布局】定义平面" if self.mode == 'LAYOUT' else "【2. 变形】拖拽图钉"
        text = pget_tmpl(
            "{mode} | 填充: {pad} [P切换] | L 切换模式 | F 重置 | Enter 应用 | Esc 取消",
            mode=mode_str,
            pad=pad_str,
        )
        
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, text)


    def save_as_copy(self):
        try:
            from ..utils.np_img_utils import npimg_2_blimg
            full_np = blimg_2_npimg(self.original_image)
            result = PerspectiveTool.process_full_homography(
                full_np, self.quad_src, self.quad_tgt, self.padding_mode
            )
            new_name = self.original_image.name + "_warped"
            npimg_2_blimg(result, new_name, True)
            bpy.ops.ed.undo_push(message="透视形变另存")
        except Exception as e:
            print(f"[透视形变] 另存失败: {e}")
        finally:
            self.cleanup()

    def apply_to_original(self):
        try:
            full_np = blimg_2_npimg(self.original_image)
            # 传入当前的填充模式
            result = PerspectiveTool.process_full_homography(
                full_np, self.quad_src, self.quad_tgt, self.padding_mode
            )
            
            h, w = result.shape[:2]
            self.original_image.scale(w, h)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            print(f"[透视形变] 应用失败: {e}")
        finally:
            self.cleanup()



    def cleanup(self):
        super().cleanup()

# ==========================================
# 操作器与面板
# ==========================================

class IMAGE_OT_perspective_warp_modal(WarpModalBase):
    bl_idname = "image_editor_tools.perspective_warp_modal"
    bl_label = "透视形变"
    engine_class = PerspectiveEngine
    tool_key = 'warp:透视形变'
    tool_label = '透视形变'
    status_text = "透视形变: L切换模式 | P填色 | F重置 | Enter应用 | Esc取消"

    def _custom_keys(self, context, event, engine):
        if event.type == 'F' and event.value == 'PRESS':
            engine.reset_transform()
            return True
        if event.type == 'P' and event.value == 'PRESS':
            engine.cycle_padding_mode()
            return True
        if event.type == 'L' and event.value == 'PRESS':
            engine.mode = 'WARP' if engine.mode == 'LAYOUT' else 'LAYOUT'
            return True
        return False

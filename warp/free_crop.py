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
from .base_op import WarpModalBase

# ==========================================
# 核心处理逻辑
# ==========================================

class CropTool:
    @staticmethod
    def process(np_array, crop_rect=(0.0, 0.0, 1.0, 1.0), rotation=0.0, padding_mode='STRETCH'):
        """
        使用 numpy 进行高效的旋转和裁切
        crop_rect: (xmin, ymin, xmax, ymax) 归一化坐标 [0, 1]
        rotation: 旋转角度（弧度）
        padding_mode: 填充模式 'STRETCH', 'REPEAT', 'TRANSPARENT', 'WHITE', 'BLACK'
        """
        h, w = np_array.shape[:2]
        xmin, ymin, xmax, ymax = crop_rect
        
        # 计算输出图像的像素尺寸
        out_w = max(1, int((xmax - xmin) * w))
        out_h = max(1, int((ymax - ymin) * h))
        
        # 创建输出图像的网格坐标
        Y, X = np.mgrid[0:out_h, 0:out_w].astype(np.float32)
        
        # 将输出坐标映射到裁切框中心相对坐标
        dx = X - out_w / 2.0
        dy = Y - out_h / 2.0
        
        # 计算裁切框在原图中的中心点
        cx = (xmin + xmax) * 0.5 * w
        cy = (ymin + ymax) * 0.5 * h
        
        # 修复：映射回原图时，应该使用与裁切框相同的正向旋转角度
        ca = np.cos(rotation)
        sa = np.sin(rotation)
        
        orig_X = cx + dx * ca - dy * sa
        orig_Y = cy + dx * sa + dy * ca
        
        orig_X_int = np.round(orig_X).astype(np.int32)
        orig_Y_int = np.round(orig_Y).astype(np.int32)
        
        if padding_mode == 'STRETCH':
            # 拉伸填充：直接限制在原图范围内 (最近邻插值)
            orig_X_safe = np.clip(orig_X_int, 0, w - 1)
            orig_Y_safe = np.clip(orig_Y_int, 0, h - 1)
            result = np_array[orig_Y_safe, orig_X_safe]
        elif padding_mode == 'REPEAT':
            # 重复填充：利用取模运算实现坐标循环
            orig_X_safe = orig_X_int % w
            orig_Y_safe = orig_Y_int % h
            result = np_array[orig_Y_safe, orig_X_safe]
        else:
            # 其他填充：先获取安全坐标用于提取像素
            orig_X_safe = np.clip(orig_X_int, 0, w - 1)
            orig_Y_safe = np.clip(orig_Y_int, 0, h - 1)
            result = np_array[orig_Y_safe, orig_X_safe].copy()
            
            # 计算越界掩码 (Mask)
            invalid_mask = (orig_X_int < 0) | (orig_X_int >= w) | (orig_Y_int < 0) | (orig_Y_int >= h)
            
            # 根据模式替换越界像素的颜色 [R, G, B, A]
            if padding_mode == 'TRANSPARENT':
                result[invalid_mask] = [0.0, 0.0, 0.0, 0.0]
            elif padding_mode == 'WHITE':
                result[invalid_mask] = [1.0, 1.0, 1.0, 1.0]
            elif padding_mode == 'BLACK':
                result[invalid_mask] = [0.0, 0.0, 0.0, 1.0]
                
        return result

# ==========================================
# 预览与交互引擎
# ==========================================

class CropEngine(BaseEngine):
    _engine_type = 'warp'

    def __init__(self, context, image):

        self.crop_rect = [0.1, 0.1, 0.9, 0.9]
        self.rotation = 0.0

        self.padding_modes = ['STRETCH', 'REPEAT', 'TRANSPARENT', 'WHITE', 'BLACK']
        self.padding_mode_idx = 0
        self.padding_mode = self.padding_modes[self.padding_mode_idx]

        self._drag_mode = 'NONE'
        self._drag_start_pos = (0, 0)
        self._drag_start_rect = []
        self._drag_start_rot = 0.0
        self._drag_start_angle = 0.0

        super().__init__(context, image)

    def cycle_padding_mode(self):
        """循环切换填充模式"""
        self.padding_mode_idx = (self.padding_mode_idx + 1) % len(self.padding_modes)
        self.padding_mode = self.padding_modes[self.padding_mode_idx]

    def reset_transform(self):
        """F键：填满边界初始化，重置旋转和裁切框"""
        self.crop_rect = [0.0, 0.0, 1.0, 1.0]
        self.rotation = 0.0

    def handle_mouse_press(self, event):
        mx, my = event.mouse_region_x, event.mouse_region_y
        self._drag_mode = self._get_hit_zone(mx, my)
        self._drag_start_pos = (mx, my)
        self._drag_start_rect = list(self.crop_rect)
        self._drag_start_rot = self.rotation
        
        # 如果是旋转模式，记录初始的鼠标角度
        if self._drag_mode == 'ROTATE':
            ox, oy, dw, dh = self._img_rect
            cx = ox + (self.crop_rect[0] + self.crop_rect[2]) * 0.5 * dw
            cy = oy + (self.crop_rect[1] + self.crop_rect[3]) * 0.5 * dh
            self._drag_start_angle = math.atan2(my - cy, mx - cx)
            
        return True



    def _draw(self):
        rect = self._get_image_rect()
        if rect is None:
            return
        offset_x, offset_y, disp_w, disp_h = rect

        x0, y0 = offset_x, offset_y
        x1, y1 = offset_x + disp_w, offset_y + disp_h

        # 1. 计算裁切框的旋转顶点
        cx = x0 + (self.crop_rect[0] + self.crop_rect[2]) * 0.5 * disp_w
        cy = y0 + (self.crop_rect[1] + self.crop_rect[3]) * 0.5 * disp_h
        hw = (self.crop_rect[2] - self.crop_rect[0]) * 0.5 * disp_w
        hh = (self.crop_rect[3] - self.crop_rect[1]) * 0.5 * disp_h

        ca, sa = math.cos(self.rotation), math.sin(self.rotation)
        def rot_pt(px, py):
            dx, dy = px - cx, py - cy
            return (cx + dx * ca - dy * sa, cy + dx * sa + dy * ca)

        p0 = rot_pt(cx - hw, cy - hh) # 左下
        p1 = rot_pt(cx + hw, cy - hh) # 右下
        p2 = rot_pt(cx + hw, cy + hh) # 右上
        p3 = rot_pt(cx - hw, cy + hh) # 左上
        box_pts = [p0, p1, p2, p3]

        # 2. 绘制完整底图 (底图不旋转)
        shader = state.get_display_shader()
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)],
            "texCoord": [(0, 0), (1, 0), (1, 1), (0, 1)],
        }).draw(shader)

        # 3. 绘制全屏半透明遮罩
        mask_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        mask_shader.bind()
        mask_shader.uniform_float("color", (0.0, 0.0, 0.0, 0.6))
        batch_for_shader(mask_shader, 'TRI_FAN', {
            "pos": [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
        }).draw(mask_shader)

        # 4. 重新绘制裁切框内部的图像 (高亮显示)
        def get_uv(pt):
            return ((pt[0] - x0) / disp_w, (pt[1] - y0) / disp_h)
        
        box_uvs = [get_uv(p) for p in box_pts]
        
        shader.bind()
        shader.uniform_sampler("image", self._cached_orig_tex)
        batch_for_shader(shader, 'TRI_FAN', {
            "pos": box_pts,
            "texCoord": box_uvs,
        }).draw(shader)

        # 5. 绘制白色裁切框和手柄
        line_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        line_shader.bind()
        line_shader.uniform_float("color", (1.0, 1.0, 1.0, 1.0))
        
        # 使用 LINES 模式，明确指定每一条边的起点和终点，这是最稳妥的绘制方式
        p0, p1, p2, p3 = box_pts
        lines_pts = [
            p0, p1,  # 底边
            p1, p2,  # 右边
            p2, p3,  # 顶边
            p3, p0   # 左边
        ]
        batch_for_shader(line_shader, 'LINES', {
            "pos": lines_pts,
        }).draw(line_shader)

        # 额外加粗绘制底边 (线宽为 5)
        gpu.state.line_width_set(5.0)
        batch_for_shader(line_shader, 'LINES', {
            "pos": [p0, p1],
        }).draw(line_shader)
        
        # 恢复默认线宽，以免影响手柄和后续其他 UI 的绘制
        gpu.state.line_width_set(1.0)

        # 绘制四个角的手柄
        h_size = 5
        for px, py in box_pts:
            batch_for_shader(line_shader, 'TRI_FAN', {
                "pos": [(px-h_size, py-h_size), (px+h_size, py-h_size), (px+h_size, py+h_size), (px-h_size, py+h_size)]
            }).draw(line_shader)

        # 新增：绘制四个角外侧的旋转手柄 (蓝色圆形)
        rot_offset = 20
        rot_pts = [
            rot_pt(cx - hw - rot_offset, cy - hh - rot_offset), # 左下外侧
            rot_pt(cx + hw + rot_offset, cy - hh - rot_offset), # 右下外侧
            rot_pt(cx + hw + rot_offset, cy + hh + rot_offset), # 右上外侧
            rot_pt(cx - hw - rot_offset, cy + hh + rot_offset)  # 左上外侧
        ]
        
        rot_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        rot_shader.bind()
        rot_shader.uniform_float("color", (0.2, 0.6, 1.0, 1.0)) # 蓝色
        
        # 生成圆形的顶点
        circle_verts = []
        segments = 16
        for i in range(segments):
            theta = 2.0 * math.pi * i / segments
            circle_verts.append((math.cos(theta), math.sin(theta)))
            
        for px, py in rot_pts:
            verts = [(px + v[0]*6, py + v[1]*6) for v in circle_verts]
            batch_for_shader(rot_shader, 'TRI_FAN', {"pos": verts}).draw(rot_shader)

        # 6. 绘制状态文字
        font_id = 0
        blf.size(font_id, 16)
        blf.color(font_id, 0.9, 0.9, 0.9, 1.0)
        
        # 转换模式名称为中文显示
        mode_names = {'STRETCH': '拉伸', 'REPEAT': '重复', 'TRANSPARENT': '透明', 'WHITE': '白底', 'BLACK': '黑底'}
        mode_str = mode_names.get(self.padding_mode, self.padding_mode)
        text = pget_tmpl(
            "自由裁切 | 拖拽边缘 | 旋转 ({angle}°) [Shift吸附] | 填充: {mode} [按P切换] | F填满 | Enter应用 | Esc取消",
            angle=f"{math.degrees(self.rotation):.1f}",
            mode=mode_str,
        )
        blf.position(font_id, x0, y0 - 25, 0)
        blf.draw(font_id, text)

    def _get_hit_zone(self, mx, my):
        ox, oy, dw, dh = self._img_rect
        
        # 计算中心点
        cx = ox + (self.crop_rect[0] + self.crop_rect[2]) * 0.5 * dw
        cy = oy + (self.crop_rect[1] + self.crop_rect[3]) * 0.5 * dh
        
        # 将鼠标坐标逆向旋转，转换到未旋转的裁切框局部空间进行碰撞检测
        ca, sa = math.cos(-self.rotation), math.sin(-self.rotation)
        dx, dy = mx - cx, my - cy
        rmx = cx + dx * ca - dy * sa
        rmy = cy + dx * sa + dy * ca

        cx0 = ox + self.crop_rect[0] * dw
        cy0 = oy + self.crop_rect[1] * dh
        cx1 = ox + self.crop_rect[2] * dw
        cy1 = oy + self.crop_rect[3] * dh
        
        # 1. 优先检查是否点击了外侧的旋转手柄
        rot_offset = 20
        rot_hit_size = 15
        rot_local_pts = [
            (cx0 - rot_offset, cy0 - rot_offset),
            (cx1 + rot_offset, cy0 - rot_offset),
            (cx1 + rot_offset, cy1 + rot_offset),
            (cx0 - rot_offset, cy1 + rot_offset)
        ]
        for rx, ry in rot_local_pts:
            if abs(rmx - rx) < rot_hit_size and abs(rmy - ry) < rot_hit_size:
                return 'ROTATE'
        
        # 2. 检查缩放手柄和内部移动
        margin = 10
        if cx0 - margin < rmx < cx1 + margin and cy0 - margin < rmy < cy1 + margin:
            l = abs(rmx - cx0) < margin
            r = abs(rmx - cx1) < margin
            b = abs(rmy - cy0) < margin
            t = abs(rmy - cy1) < margin
            
            if l and t: return 'LT'
            if r and t: return 'RT'
            if l and b: return 'LB'
            if r and b: return 'RB'
            if l: return 'L'
            if r: return 'R'
            if t: return 'T'
            if b: return 'B'
            return 'MOVE'
            
        return 'NONE'


    def handle_mouse_move(self, event):
        if self._drag_mode == 'NONE':
            return False

        mx, my = event.mouse_region_x, event.mouse_region_y
        dx = mx - self._drag_start_pos[0]
        dy = my - self._drag_start_pos[1]
        
        ox, oy, dw, dh = self._img_rect

        if self._drag_mode == 'ROTATE':
            # 计算中心点
            cx = ox + (self._drag_start_rect[0] + self._drag_start_rect[2]) * 0.5 * dw
            cy = oy + (self._drag_start_rect[1] + self._drag_start_rect[3]) * 0.5 * dh
            
            # 计算当前鼠标相对于中心点的角度
            current_angle = math.atan2(my - cy, mx - cx)
            angle_diff = current_angle - self._drag_start_angle
            
            raw_rot = self._drag_start_rot + angle_diff
            
            if event.shift:
                # 按住 Shift 时，吸附到 5 度的整数倍
                deg = math.degrees(raw_rot)
                deg_snapped = round(deg / 5.0) * 5.0
                self.rotation = math.radians(deg_snapped)
            else:
                self.rotation = raw_rot
            
        elif self._drag_mode == 'MOVE':
            ndx = dx / dw if dw > 0 else 0
            ndy = dy / dh if dh > 0 else 0
            
            box_w = self._drag_start_rect[2] - self._drag_start_rect[0]
            box_h = self._drag_start_rect[3] - self._drag_start_rect[1]
            
            new_xmin = self._drag_start_rect[0] + ndx
            new_ymin = self._drag_start_rect[1] + ndy
            
            self.crop_rect[0] = new_xmin
            self.crop_rect[1] = new_ymin
            self.crop_rect[2] = new_xmin + box_w
            self.crop_rect[3] = new_ymin + box_h
            
        else:
            # 1. 计算初始的中心点和半宽高（屏幕像素）
            start_cx = (self._drag_start_rect[0] + self._drag_start_rect[2]) * 0.5 * dw
            start_cy = (self._drag_start_rect[1] + self._drag_start_rect[3]) * 0.5 * dh
            start_hw = (self._drag_start_rect[2] - self._drag_start_rect[0]) * 0.5 * dw
            start_hh = (self._drag_start_rect[3] - self._drag_start_rect[1]) * 0.5 * dh
            
            # 2. 将鼠标位移逆向旋转，转换到裁切框的局部像素坐标系
            ca, sa = math.cos(-self._drag_start_rot), math.sin(-self._drag_start_rot)
            rdx = dx * ca - dy * sa
            rdy = dx * sa + dy * ca
            
            # 3. 定义局部坐标系下的四条边
            l = -start_hw
            r = start_hw
            b = -start_hh
            t = start_hh
            
            # 4. 根据拖拽模式更新对应的边
            if 'L' in self._drag_mode: l += rdx
            if 'R' in self._drag_mode: r += rdx
            if 'B' in self._drag_mode: b += rdy
            if 'T' in self._drag_mode: t += rdy
            
            # 5. 限制最小尺寸（防止反向交叉），最小限制为 10 个像素
            min_px = 10.0
            if r - l < min_px:
                if 'L' in self._drag_mode: l = r - min_px
                if 'R' in self._drag_mode: r = l + min_px
            if t - b < min_px:
                if 'B' in self._drag_mode: b = t - min_px
                if 'T' in self._drag_mode: t = b + min_px
                
            # 6. 计算新的局部中心点和新的半宽高
            new_local_cx = (l + r) * 0.5
            new_local_cy = (b + t) * 0.5
            new_hw = (r - l) * 0.5
            new_hh = (t - b) * 0.5
            
            # 7. 将局部中心点顺向旋转回全局屏幕坐标系
            ca_rot, sa_rot = math.cos(self._drag_start_rot), math.sin(self._drag_start_rot)
            global_cx = start_cx + (new_local_cx * ca_rot - new_local_cy * sa_rot)
            global_cy = start_cy + (new_local_cx * sa_rot + new_local_cy * ca_rot)
            
            # 8. 转换回归一化的 crop_rect
            if dw > 0 and dh > 0:
                # 修复：先计算出当前的归一化坐标
                new_xmin = (global_cx - new_hw) / dw
                new_xmax = (global_cx + new_hw) / dw
                new_ymin = (global_cy - new_hh) / dh
                new_ymax = (global_cy + new_hh) / dh
                
                # 新增：按住 Shift 时吸附到 0.1 的倍数
                if event.shift:
                    if 'L' in self._drag_mode: new_xmin = round(new_xmin * 10.0) / 10.0
                    if 'R' in self._drag_mode: new_xmax = round(new_xmax * 10.0) / 10.0
                    if 'B' in self._drag_mode: new_ymin = round(new_ymin * 10.0) / 10.0
                    if 'T' in self._drag_mode: new_ymax = round(new_ymax * 10.0) / 10.0
                    
                    # 再次确保吸附后不会反向交叉，最小间隔为 0.1
                    if new_xmax <= new_xmin:
                        if 'L' in self._drag_mode: new_xmin = new_xmax - 0.1
                        if 'R' in self._drag_mode: new_xmax = new_xmin + 0.1
                    if new_ymax <= new_ymin:
                        if 'B' in self._drag_mode: new_ymin = new_ymax - 0.1
                        if 'T' in self._drag_mode: new_ymax = new_ymin + 0.1

                # 统一赋值给 crop_rect
                self.crop_rect[0] = new_xmin
                self.crop_rect[2] = new_xmax
                self.crop_rect[1] = new_ymin
                self.crop_rect[3] = new_ymax

                
        return True

    def handle_mouse_release(self, event):
        self._drag_mode = 'NONE'

    def save_as_copy(self):
        try:
            full_np = blimg_2_npimg(self.original_image)
            result = CropTool.process(full_np, self.crop_rect, self.rotation, self.padding_mode)
            new_name = self.original_image.name + "_cropped"
            npimg_2_blimg(result, new_name, True)
            bpy.ops.ed.undo_push(message="自由裁切另存")
        except Exception as e:
            print(f"[自由裁切] 另存失败: {e}")

    def apply_to_original(self):
        try:
            full_np = blimg_2_npimg(self.original_image)
            # 直接使用类属性中的填充模式
            result = CropTool.process(full_np, self.crop_rect, self.rotation, self.padding_mode)
            
            h, w = result.shape[:2]
            self.original_image.scale(w, h)
            self.original_image.pixels.foreach_set(result.ravel())
            self.original_image.update()
        except Exception as e:
            print(f"裁切应用失败: {e}")
        finally:
            self.cleanup()



    def cleanup(self):
        super().cleanup()

# ==========================================
# 操作器与面板
# ==========================================

class IMAGE_OT_free_crop_modal(WarpModalBase):
    bl_idname = "image_editor_tools.free_crop_modal"
    bl_label = "自由裁切"
    engine_class = CropEngine
    tool_key = 'warp:自由裁切'
    tool_label = '自由裁切'
    status_text = "自由裁切: 拖拽边角/旋转 | P填色 | F重置 | Enter应用 | Esc取消"
    _drag_attr = '_drag_mode'
    _drag_none = 'NONE'

    def _custom_keys(self, context, event, engine):
        if event.type == 'P' and event.value == 'PRESS':
            engine.cycle_padding_mode()
            return True
        if event.type == 'F' and event.value == 'PRESS':
            engine.reset_transform()
            return True
        return False


        return {'FINISHED'}

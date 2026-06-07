import bpy
import gpu


class BaseEngine:
    _active_instance = None
    _engine_type = None

    def __init__(self, context, image):
        self.context = context
        self.original_image = image

        self._draw_handler = None
        self.should_exit = False
        self._cached_orig_tex = None
        self._cached_orig_image = None
        self._img_rect = (0.0, 0.0, 1.0, 1.0)

        self.__class__._active_instance = self
        self._on_init_before_draw()
        self._start_drawing()

    def _on_init_before_draw(self):
        """子类可在绘制开始前执行额外初始化"""
        pass

    def _report_error(self, msg):
        print(f"[图像编辑工具集] {msg}")

    def _start_drawing(self):
        engine = self

        def draw_callback():
            try:
                engine._draw()
            except Exception as e:
                engine._report_error(f"绘制错误: {e}")

        self._draw_handler = bpy.types.SpaceImageEditor.draw_handler_add(
            draw_callback, (), 'WINDOW', 'POST_PIXEL'
        )

    def _get_image_rect(self):
        """计算并缓存图像在区域中的绘制矩形，返回 (offset_x, offset_y, disp_w, disp_h) 或 None"""
        if self.original_image is None:
            return None

        context = bpy.context
        region = context.region
        if region is None or region.type != 'WINDOW':
            return None

        img_w, img_h = self.original_image.size
        if img_w <= 0 or img_h <= 0:
            return None

        zoom_x, zoom_y = context.space_data.zoom[0], context.space_data.zoom[1]
        disp_w = float(img_w) * zoom_x
        disp_h = float(img_h) * zoom_y

        view2d = region.view2d
        bl = view2d.view_to_region(0.0, 0.0, clip=False)
        offset_x, offset_y = bl[0], bl[1]

        self._img_rect = (offset_x, offset_y, disp_w, disp_h)

        if self._cached_orig_image is not self.original_image:
            self._cached_orig_tex = gpu.texture.from_image(self.original_image)
            self._cached_orig_image = self.original_image

        return offset_x, offset_y, disp_w, disp_h

    def _draw(self):
        raise NotImplementedError

    def handle_mouse_press(self, event):
        return False

    def handle_mouse_move(self, event):
        return False

    def handle_mouse_release(self, event):
        pass

    def apply_to_original(self):
        raise NotImplementedError

    def save_as_copy(self):
        raise NotImplementedError

    def cleanup(self):
        self.should_exit = True
        if self._draw_handler is not None:
            bpy.types.SpaceImageEditor.draw_handler_remove(self._draw_handler, 'WINDOW')
            self._draw_handler = None
        self._cached_orig_tex = None
        self._cached_orig_image = None
        self.__class__._active_instance = None

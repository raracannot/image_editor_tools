import bpy


class BaseTool:
    tool_id: str = ''
    label: str = ''
    modal_operator: str = 'image.tool_start'

    @staticmethod
    def get_properties() -> dict:
        return {}

    @staticmethod
    def draw_panel(layout: bpy.types.UILayout, props: bpy.types.PropertyGroup):
        pass

    @staticmethod
    def process(np_array, props: bpy.types.PropertyGroup):
        raise NotImplementedError

    @staticmethod
    def on_mouse_press(props, img_rect, mx, my):
        """返回 drag_state dict 表示已处理，返回 None 表示未处理"""
        return None

    @staticmethod
    def on_mouse_move(props, img_rect, mx, my, drag_state):
        """拖拽中每帧调用，返回 True 表示已处理"""
        return False

    @staticmethod
    def on_mouse_release(props):
        """拖拽结束清理 (可选)"""
        pass

    @staticmethod
    def on_apply(full_np, props, save_mode):
        """返回 ndarray 则用它替代 process 结果写入原图;
           返回 None 则走默认 process 逻辑。
           save_mode: 'original' | 'copy'"""
        return None

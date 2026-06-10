from .base import BaseTool


class RebuildAlphaTool(BaseTool):
    tool_id = 'rebuild_alpha'
    label = '黑底抠图'

    @staticmethod
    def get_properties():
        return {}

    @staticmethod
    def draw_panel(layout, props):
        layout.label(text="alpha = max(R,G,B) 重建透明通道")

    @staticmethod
    def process(np_array, props):
        from ..utils.np_img_utils import np_rebuild_alpha
        return np_rebuild_alpha(np_array)

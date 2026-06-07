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

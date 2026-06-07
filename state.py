import bpy

current_tool = 'NONE'

_SRGB_SHADER_VERSION = (4, 3, 0)
_HAS_SRGB_SHADER = bpy.app.version >= _SRGB_SHADER_VERSION


def get_display_shader():
    import gpu
    shader = 'IMAGE_SCENE_LINEAR_TO_REC709_SRGB' if _HAS_SRGB_SHADER else 'IMAGE'
    return gpu.shader.from_builtin(shader)

import bpy

current_tool = 'NONE'


_SRGB_SHADER_VERSION = (4, 3, 0)
_HAS_SRGB_SHADER = bpy.app.version >= _SRGB_SHADER_VERSION
_cached_shader = None
def get_display_shader():
    import gpu
    global _cached_shader
    shader = 'IMAGE_SCENE_LINEAR_TO_REC709_SRGB' if _HAS_SRGB_SHADER else 'IMAGE'
    if _cached_shader is None:
        _cached_shader = gpu.shader.from_builtin(shader)
    return _cached_shader

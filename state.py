import bpy

current_tool = 'NONE'

_cached_shader = None
_HAS_SRGB_SHADER = False
def get_display_shader():
    global _cached_shader,_HAS_SRGB_SHADER
    if _cached_shader is None:
        import gpu
        try:
            _cached_shader = gpu.shader.from_builtin('IMAGE_SCENE_LINEAR_TO_REC709_SRGB')
            _HAS_SRGB_SHADER = True
        except Exception:
            _cached_shader = gpu.shader.from_builtin('IMAGE')
            _HAS_SRGB_SHADER = False
    return _cached_shader

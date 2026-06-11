import bpy
from ..engine_base import BaseEngine
from .. import state
from .bezier_warp import IMAGE_OT_mesh_warp_modal, MeshWarpEngine
from .perspective_warp import IMAGE_OT_perspective_warp_modal, PerspectiveEngine
from .free_crop import IMAGE_OT_free_crop_modal, CropEngine
from .place_image import IMAGE_OT_place_image_modal, PlaceImageEngine
from .slice_image import IMAGE_OT_slice_image_modal, SliceImageEngine


def _get_active_engine():
    for engine_cls in BaseEngine.__subclasses__():
        if getattr(engine_cls, '_engine_type', None) == 'warp':
            eng = engine_cls._active_instance
            if eng is not None:
                return eng
    return None


class IMAGE_OT_warp_cancel(bpy.types.Operator):
    bl_idname = "image_editor_tools.warp_cancel"
    bl_label = "取消"

    def execute(self, context):
        eng = _get_active_engine()
        if eng is not None:
            eng.cleanup()
            state.current_tool = 'NONE'
        return {'FINISHED'}


class IMAGE_OT_warp_apply(bpy.types.Operator):
    bl_idname = "image_editor_tools.warp_apply"
    bl_label = "应用"

    def execute(self, context):
        eng = _get_active_engine()
        if eng is not None:
            eng.apply_to_original()
            state.current_tool = 'NONE'
        return {'FINISHED'}


class IMAGE_OT_warp_save_as(bpy.types.Operator):
    bl_idname = "image_editor_tools.warp_save_as"
    bl_label = "另存"

    def execute(self, context):
        eng = _get_active_engine()
        if eng is not None:
            eng.save_as_copy()
            state.current_tool = 'NONE'
        return {'FINISHED'}


classes = [
    IMAGE_OT_mesh_warp_modal,
    IMAGE_OT_perspective_warp_modal,
    IMAGE_OT_free_crop_modal,
    IMAGE_OT_place_image_modal,
    IMAGE_OT_slice_image_modal,
    IMAGE_OT_warp_cancel,
    IMAGE_OT_warp_apply,
    IMAGE_OT_warp_save_as,
]


def register():
    for cls in classes:
        bpy.utils.register_class(cls)


def unregister():
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass

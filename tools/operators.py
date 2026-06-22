import bpy
from .. import state
from ..translation import pget_tmpl
from .preview_engine import PreviewEngine


class IMAGE_OT_tool_start(bpy.types.Operator):
    bl_idname = "image_editor_tools.tool_start"
    bl_label = "启动编辑工具"
    bl_description = "启动图像编辑模态"

    tool_id: bpy.props.StringProperty()

    @classmethod
    def poll(cls, context):
        return (
            context.space_data.type == 'IMAGE_EDITOR'
            and context.space_data.image is not None
        )

    def execute(self, context):
        from . import TOOLS
        tool = TOOLS.get(self.tool_id)
        if tool is None:
            return {'CANCELLED'}

        if PreviewEngine._active_instance is not None:
            PreviewEngine._active_instance.cleanup()

        self._prev_ui_mode = str(context.space_data.ui_mode)
        if context.area.ui_type != 'IMAGE_EDITOR':
            context.area.ui_type = 'IMAGE_EDITOR'
        context.space_data.ui_mode = 'VIEW'

        image = context.space_data.image
        engine = PreviewEngine(context, image, tool)

        props = context.scene.image_editor_tools
        state.current_tool = self.tool_id

        self._tool = tool
        self._tool_drag = None

        bpy.ops.ed.undo_push(message=pget_tmpl("进入{label}模态", label=tool.label))

        wm = context.window_manager
        self._timer = wm.event_timer_add(0.1, window=context.window)
        try:
            wm.modal_handler_add(self)
            self._engine = engine
            return {'RUNNING_MODAL'}
        except Exception:
            wm.event_timer_remove(self._timer)
            self._timer = None
            engine.cleanup()
            raise

    def modal(self, context, event):
        if state.current_tool == 'NONE':
            return self._finish(context)

        engine = getattr(self, '_engine', None)
        if engine is None or engine.should_exit:
            return self._finish(context)

        if event.type == 'TIMER':
            try:
                _ = engine.original_image.name
            except ReferenceError:
                return self._finish(context)
            for window in context.window_manager.windows:
                for area in window.screen.areas:
                    if area.type == 'IMAGE_EDITOR':
                        area.tag_redraw()
            return {'PASS_THROUGH'}

        if event.type in {'ESC', 'Z'}:
            if event.type == 'Z' and not event.ctrl:
                return {'PASS_THROUGH'}
            engine.cleanup()
            return self._finish(context)

        if event.type in {'LEFTMOUSE', 'MOUSEMOVE'}:
            region = context.region
            if region is None:
                return {'PASS_THROUGH'}

            mx = event.mouse_region_x
            my = event.mouse_region_y
            props = context.scene.image_editor_tools
            tool = getattr(self, '_tool', None)

            if event.type == 'LEFTMOUSE':
                if event.value == 'PRESS':
                    if engine.handle_mouse_press(event):
                        return {'RUNNING_MODAL'}
                    if tool is not None:
                        ds = tool.on_mouse_press(props, engine._img_rect, mx, my)
                        if ds is not None:
                            self._tool_drag = ds
                            return {'RUNNING_MODAL'}
                elif event.value == 'RELEASE':
                    engine.handle_mouse_release(event)
                    if self._tool_drag is not None and tool is not None:
                        tool.on_mouse_release(props)
                        self._tool_drag = None
                    return {'PASS_THROUGH'}

            elif event.type == 'MOUSEMOVE':
                if engine._dragging:
                    if engine.handle_mouse_move(event):
                        return {'RUNNING_MODAL'}
                elif self._tool_drag is not None and tool is not None:
                    if tool.on_mouse_move(props, engine._img_rect, mx, my, self._tool_drag):
                        return {'RUNNING_MODAL'}

        return {'PASS_THROUGH'}

    def _finish(self, context):
        self._tool_drag = None
        engine = getattr(self, '_engine', None)
        if engine is not None:
            props = context.scene.image_editor_tools
            state.current_tool = 'NONE'
            if not engine.should_exit:
                engine.cleanup()
            self._engine = None

        timer = getattr(self, '_timer', None)
        if timer is not None:
            context.window_manager.event_timer_remove(timer)
            self._timer = None

        prev_ui_mode = getattr(self, '_prev_ui_mode', None)
        if prev_ui_mode is not None:
            try:
                area = context.area
                if area is not None and area.type == 'IMAGE_EDITOR':
                    sp = area.spaces.active
                    if sp is not None:
                        sp.ui_mode = prev_ui_mode
            except Exception:
                pass

        return {'CANCELLED'}


class IMAGE_OT_tool_cancel(bpy.types.Operator):
    bl_idname = "image_editor_tools.tool_cancel"
    bl_label = "取消"
    bl_description = "取消当前工具操作"

    def execute(self, context):
        engine = PreviewEngine._active_instance
        if engine is not None:
            engine.cleanup()
            state.current_tool = 'NONE'
        return {'FINISHED'}


class IMAGE_OT_tool_save(bpy.types.Operator):
    bl_idname = "image_editor_tools.tool_save"
    bl_label = "保存"
    bl_description = "以当前参数编辑原图"

    def execute(self, context):
        engine = PreviewEngine._active_instance
        if engine is not None:
            engine.apply_to_original()
            state.current_tool = 'NONE'
        return {'FINISHED'}


class IMAGE_OT_tool_save_as(bpy.types.Operator):
    bl_idname = "image_editor_tools.tool_save_as"
    bl_label = "另存"
    bl_description = "复制原图后以当前参数对副本进行编辑"

    def execute(self, context):
        engine = PreviewEngine._active_instance
        if engine is not None:
            engine.apply_to_new()
            state.current_tool = 'NONE'
        return {'FINISHED'}




classes = [
    IMAGE_OT_tool_start,
    IMAGE_OT_tool_cancel,
    IMAGE_OT_tool_save,
    IMAGE_OT_tool_save_as,
]

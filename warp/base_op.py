import bpy
from .. import state


class WarpModalBase(bpy.types.Operator):
    bl_options = {'REGISTER', 'UNDO'}
    engine_class = None
    tool_key = ''
    tool_label = ''
    status_text = ''
    _drag_attr = '_drag_idx'
    _drag_none = -1

    @classmethod
    def poll(cls, context):
        return (
            context.space_data is not None
            and context.space_data.type == 'IMAGE_EDITOR'
            and context.space_data.image is not None
        )

    def _custom_keys(self, context, event, engine):
        return False

    def _on_mouse_press(self, event, engine):
        if engine.handle_mouse_press(event):
            return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    def _on_mouse_release(self, event, engine):
        engine.handle_mouse_release(event)
        return {'PASS_THROUGH'}

    def _on_mouse_move(self, event, engine):
        val = getattr(engine, self._drag_attr, None)
        if val is not None and val != self._drag_none:
            if engine.handle_mouse_move(event):
                return {'RUNNING_MODAL'}
        return {'PASS_THROUGH'}

    def modal(self, context, event):
        engine = self.engine_class._active_instance
        if not engine or engine.should_exit:
            return self._finish(context)
        context.area.tag_redraw()

        if event.type in {'RIGHTMOUSE', 'ESC'}:
            engine.cleanup()
            return self._finish(context)
        elif event.type in {'RET', 'NUMPAD_ENTER'}:
            engine.apply_to_original()
            return self._finish(context)

        if self._custom_keys(context, event, engine):
            return {'RUNNING_MODAL'}

        if event.type == 'LEFTMOUSE':
            if event.value == 'PRESS':
                return self._on_mouse_press(event, engine)
            elif event.value == 'RELEASE':
                return self._on_mouse_release(event, engine)
        elif event.type == 'MOUSEMOVE':
            return self._on_mouse_move(event, engine)

        return {'PASS_THROUGH'}

    def invoke(self, context, event):
        if context.space_data is None or context.space_data.type != 'IMAGE_EDITOR' or not context.space_data.image:
            self.report({'WARNING'}, "请在图像编辑器中打开一张图片")
            return {'CANCELLED'}
        if self.engine_class._active_instance:
            self.engine_class._active_instance.cleanup()

        self._prev_ui_mode = str(context.space_data.ui_mode)
        if context.area.ui_type != 'IMAGE_EDITOR':
            context.area.ui_type = 'IMAGE_EDITOR'
        context.space_data.ui_mode = 'VIEW'

        bpy.ops.ed.undo_push(message=self.tool_label)
        state.current_tool = self.tool_key
        self.engine_class(context, context.space_data.image)
        context.window_manager.modal_handler_add(self)
        context.workspace.status_text_set(self.status_text)
        return {'RUNNING_MODAL'}

    def _finish(self, context):
        state.current_tool = 'NONE'
        prev = getattr(self, '_prev_ui_mode', None)
        if prev is not None:
            try:
                area = context.area
                if area is not None and area.type == 'IMAGE_EDITOR':
                    sp = area.spaces.active
                    if sp is not None:
                        sp.ui_mode = prev
            except Exception:
                pass
        context.workspace.status_text_set(None)
        return {'FINISHED'}

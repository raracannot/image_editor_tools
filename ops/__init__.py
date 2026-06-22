import bpy


_ADDON_NAME = __package__.rsplit('.', 1)[0]


def _get_current_node_tree(context):
    try:
        space = context.space_data
        tree = space.node_tree
        if hasattr(space, "path") and space.path:
            tree = space.path[-1].node_tree
        return tree
    except Exception:
        return None


def _get_image_from_selected_node(context):
    """扫描选中节点的属性/输入/输出, 提取所有 bpy.types.Image 对象.
    :return: [[node, property_path, image], ...]
    """
    node_image_list = []
    node_tree = _get_current_node_tree(context)
    if not node_tree:
        return node_image_list

    selected_nodes = [n for n in node_tree.nodes if n.select]
    if not selected_nodes:
        return node_image_list

    for node in selected_nodes:
        try:
            for prop_name in dir(node):
                if prop_name.startswith(('_', 'bl_', 'rna_', 'type')):
                    continue
                try:
                    val = getattr(node, prop_name, None)
                    if isinstance(val, bpy.types.Image):
                        node_image_list.append([node, prop_name, val])
                except (AttributeError, TypeError, RuntimeError):
                    continue
        except Exception:
            continue

        if hasattr(node, 'inputs') and node.inputs:
            for i, sock in enumerate(node.inputs):
                try:
                    dv = getattr(sock, 'default_value', None)
                    if isinstance(dv, bpy.types.Image):
                        node_image_list.append([node, f'inputs[{i}]', dv])
                except (AttributeError, TypeError, RuntimeError):
                    continue

        if hasattr(node, 'outputs') and node.outputs:
            for i, sock in enumerate(node.outputs):
                try:
                    dv = getattr(sock, 'default_value', None)
                    if isinstance(dv, bpy.types.Image):
                        node_image_list.append([node, f'outputs[{i}]', dv])
                except (AttributeError, TypeError, RuntimeError):
                    continue

    return node_image_list


class IMAGEEDITOR_TOOLS_OT_open_in_blender_editor(bpy.types.Operator):
    bl_idname = "image_editor_tools.open_in_blender_editor"
    bl_label = "在内部编辑器打开图像"
    bl_description = "在新窗口中用图像编辑器打开选中节点的贴图"
    bl_options = {'REGISTER', 'UNDO'}

    def execute(self, context):
        image_list = _get_image_from_selected_node(context)
        if not image_list:
            self.report({'ERROR'}, "未在选中节点中找到任何图像对象")
            return {'CANCELLED'}

        _node, _prop_name, image = image_list[0]

        render = context.scene.render
        view = context.preferences.view
        old_res_x = render.resolution_x
        old_res_y = render.resolution_y
        old_res_percent = render.resolution_percentage
        old_display_type = view.render_display_type

        try:
            img_w, img_h = image.size
            win_w = context.window.width
            win_h = context.window.height

            target_area = (win_w * win_h) / 4.0
            img_area = img_w * img_h
            scale = (target_area / img_area) ** 0.5
            scale = min(scale, (win_w * 0.8) / img_w, (win_h * 0.8) / img_h)

            render.resolution_x = int(img_w * scale)
            render.resolution_y = int(img_h * scale)
            render.resolution_percentage = 100
            view.render_display_type = 'WINDOW'

            bpy.ops.render.view_show('INVOKE_DEFAULT')

            new_win = context.window_manager.windows[-1]
            area = next((a for a in new_win.screen.areas if a.type == 'IMAGE_EDITOR'), None)
            if area:
                space = area.spaces.active
                space.ui_mode = 'VIEW'
                space.image = image
                space.show_region_toolbar = True
                try:
                    space.show_region_ui = True
                except AttributeError:
                    pass

                for region in area.regions:
                    if region.type == 'WINDOW':
                        with context.temp_override(window=new_win, area=area, region=region):
                            bpy.ops.image.view_all(fit_view=True)
                            bpy.ops.image.view_zoom_out(location=(0.5, 0.5))
                        break

                ui_region = next((r for r in area.regions if r.type == 'UI'), None)
                if ui_region is not None:
                    try:
                        bpy.ops.wm.context_set_string(
                            context.temp_override(window=new_win, area=area, region=ui_region),
                            data_path="space_data.active_panel_category",
                            value="Tool",
                        )
                    except Exception:
                        pass

        except Exception as e:
            self.report({'WARNING'}, f"打开图像编辑器失败: {e}")
        finally:
            render.resolution_x = old_res_x
            render.resolution_y = old_res_y
            render.resolution_percentage = old_res_percent
            view.render_display_type = old_display_type

        return {'FINISHED'}


class IMAGEEDITOR_TOOLS_OT_clipboard_paste_to_prop(bpy.types.Operator):
    bl_idname = "image_editor_tools.clipboard_paste_to_prop"
    bl_label = ""
    bl_description = "从剪贴板导入图像"
    bl_options = {'REGISTER', 'UNDO'}

    target_prop: bpy.props.StringProperty()

    def execute(self, context):
        try:
            from ..utils.clipboard_image import import_image_from_clipboard
            img = import_image_from_clipboard()
            if img is None:
                self.report({'WARNING'}, "剪贴板中无有效图像")
                return {'CANCELLED'}
            setattr(context.scene.image_editor_tools, self.target_prop, img)
            self.report({'INFO'}, f"已导入: {img.name}")
        except Exception as e:
            self.report({'ERROR'}, f"剪贴板导入失败: {e}")
            return {'CANCELLED'}
        return {'FINISHED'}


def _menu_draw(self, context):
    addon = context.preferences.addons.get(_ADDON_NAME)
    if addon and addon.preferences and not addon.preferences.show_node_menu:
        return
    image_list = _get_image_from_selected_node(context)
    if not image_list:
        return
    self.layout.operator(
        "image_editor_tools.open_in_blender_editor",
        text="在内部图像编辑器中编辑",
        icon='COLOR',
    )


def register():
    bpy.utils.register_class(IMAGEEDITOR_TOOLS_OT_open_in_blender_editor)
    bpy.utils.register_class(IMAGEEDITOR_TOOLS_OT_clipboard_paste_to_prop)
    bpy.types.NODE_MT_context_menu.append(_menu_draw)


def unregister():
    try:
        bpy.types.NODE_MT_context_menu.remove(_menu_draw)
    except Exception:
        pass
    try:
        bpy.utils.unregister_class(IMAGEEDITOR_TOOLS_OT_clipboard_paste_to_prop)
    except Exception:
        pass
    try:
        bpy.utils.unregister_class(IMAGEEDITOR_TOOLS_OT_open_in_blender_editor)
    except Exception:
        pass

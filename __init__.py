# image_editor_master

bl_info = {
    "name": "Image Editor Master [图像超级工具]",
    "author": "RARA[来一点咖啡吗]",
    "version": (0, 3, 7),
    "blender": (4, 2, 0),
    'doc_url': 'https://space.bilibili.com/27284213',
    "location": "Image Editor > Sidebar(N-Panel) > Tool",
    "description": "Super Image Toolbox (color, filters, texture optimization)",
    "category": "Tool",
}

import bpy
import traceback
from . import warp
from . import ops
from . import state
from . import translation
from .tools import TOOLS, operator_classes, PreviewEngine
from .utils.blend_modes import BLEND_MODE_ITEMS
from .utils import gpu_img_utils as giu


class IMAGEEDITOR_TOOLS_PG_Properties(bpy.types.PropertyGroup):
    place_img_fg: bpy.props.PointerProperty(
        name="前景图", type=bpy.types.Image,
        description="用于置入叠加的前景图像",
    )
    place_img_mode: bpy.props.EnumProperty(
        name="混合模式",
        items=BLEND_MODE_ITEMS,
        default='MIX',
    )
    place_img_opacity: bpy.props.FloatProperty(
        name="不透明度", default=1.0, min=0.0, max=1.0,
        soft_min=0.0, soft_max=1.0, subtype='FACTOR',
    )
    place_text_content: bpy.props.StringProperty(
        name="文字内容", default="文字",
        description="要置入的文字内容",
    )
    place_text_font_path: bpy.props.StringProperty(
        name="字体文件",
        subtype='FILE_PATH',
        default="",
        description="自定义字体文件路径，留空使用默认字体",
    )
    place_text_font_size: bpy.props.IntProperty(
        name="字号", default=120, min=10, max=2000,
        description="文字大小（像素）",
    )
    place_text_color: bpy.props.FloatVectorProperty(
        name="文字颜色",
        subtype='COLOR',
        size=4,
        default=(1.0, 1.0, 1.0, 1.0),
        min=0.0, max=1.0,
        description="文字颜色 (RGBA)",
    )
    place_text_letter_spacing: bpy.props.FloatProperty(
        name="字间距", default=0.0, min=-50.0, max=500.0,
        description="字符之间的额外间距（像素）",
    )
    place_text_italic_angle: bpy.props.FloatProperty(
        name="倾斜角度", default=0.0, min=-45.0, max=45.0,
        description="文字倾斜角度（度）",
    )
    place_text_padding: bpy.props.FloatProperty(
        name="出血", default=0.2, min=0.0, max=5.0,
        description="基于字高的出血比例，实际出血 = 系数 × 字号",
    )
    place_text_mode: bpy.props.EnumProperty(
        name="混合模式",
        items=BLEND_MODE_ITEMS,
        default='MIX',
    )
    place_text_opacity: bpy.props.FloatProperty(
        name="不透明度", default=1.0, min=0.0, max=1.0,
        soft_min=0.0, soft_max=1.0, subtype='FACTOR',
    )
    slice_cols: bpy.props.IntProperty(
        name="纵向分割数", default=2, min=0, max=12,
        description="纵向分割线数量",
    )
    slice_rows: bpy.props.IntProperty(
        name="横向分割数", default=2, min=0, max=12,
        description="横向分割线数量",
    )


class IMAGEEDITOR_TOOLS_PT_MainPanel(bpy.types.Panel):
    bl_label = "图像编辑工具集"
    bl_idname = "IMAGEEDITOR_TOOLS_PT_MainPanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Tool"

    @classmethod
    def poll(cls, context):
        return (
            context.space_data.image is not None
            and state.current_tool == 'NONE'
        )

    def draw(self, context):
        layout = self.layout
        props = context.scene.image_editor_tools
        prefs = _get_prefs(context)

        # ── 色调调整 ──
        box = layout.box()
        row = box.row(align=True)
        row.prop(prefs, "ui_color", text="色调调整", icon='TRIA_DOWN' if prefs.ui_color else 'TRIA_RIGHT', emboss=False)
        if prefs.ui_color:
            col = box.column(align=True)
            op = col.operator("image_editor_tools.tool_start", text="基础调色器", icon='COLOR')
            op.tool_id = 'color'
            op = col.operator("image_editor_tools.tool_start", text="通道混合器", icon='COLOR')
            op.tool_id = 'channel_mixer'
            op = col.operator("image_editor_tools.tool_start", text="颜色混色器", icon='COLOR')
            op.tool_id = 'mixer'
            op = col.operator("image_editor_tools.tool_start", text="色阶", icon='CON_ACTION')
            op.tool_id = 'levels'
            op = col.operator("image_editor_tools.tool_start", text="色彩迁徙", icon='COLOR')
            op.tool_id = 'color_transfer'
            op = col.operator("image_editor_tools.tool_start", text="色调分离", icon='MOD_MASK')
            op.tool_id = 'posterize'
            op = col.operator("image_editor_tools.tool_start", text="色深优化", icon='RENDER_RESULT')
            op.tool_id = 'depth'
            op = col.operator("image_editor_tools.tool_start", text="反相", icon='IMAGE_RGB')
            op.tool_id = 'invert'

        # ── 滤镜效果 ──
        box = layout.box()
        row = box.row(align=True)
        row.prop(prefs, "ui_filter", text="滤镜效果", icon='TRIA_DOWN' if prefs.ui_filter else 'TRIA_RIGHT', emboss=False)
        if prefs.ui_filter:
            col = box.column(align=True)
            op = col.operator("image_editor_tools.tool_start", text="模糊", icon='MATSHADERBALL')
            op.tool_id = 'blur'
            op = col.operator("image_editor_tools.tool_start", text="高反差保留", icon='MOD_DISPLACE')
            op.tool_id = 'high_pass'
            op = col.operator("image_editor_tools.tool_start", text="USM锐化", icon='OUTLINER_OB_LIGHTPROBE')
            op.tool_id = 'sharpen'
            op = col.operator("image_editor_tools.tool_start", text="噪点", icon='MOD_PARTICLES')
            op.tool_id = 'noise'
            op = col.operator("image_editor_tools.tool_start", text="边缘检测", icon='OUTLINER_OB_LIGHTPROBE')
            op.tool_id = 'edge_detect'
            op = col.operator("image_editor_tools.tool_start", text="位移", icon='MOD_UVPROJECT')
            op.tool_id = 'offset'
            op = col.operator("image_editor_tools.tool_start", text="马赛克", icon='MESH_GRID')
            op.tool_id = 'mosaic'
            op = col.operator("image_editor_tools.tool_start", text="晶格化", icon='OUTLINER_OB_POINTCLOUD')
            op.tool_id = 'crystallize'
            op = col.operator("image_editor_tools.tool_start", text="浮雕", icon='MOD_BUILD')
            op.tool_id = 'emboss'
            op = col.operator("image_editor_tools.tool_start", text="减少杂色", icon='MOD_SOFT')
            op.tool_id = 'denoise'
            op = col.operator("image_editor_tools.tool_start", text="色彩半调", icon='SHADING_SOLID')
            op.tool_id = 'halftone'

        # ── 贴图处理 ──
        box = layout.box()
        row = box.row(align=True)
        row.prop(prefs, "ui_texture", text="贴图处理", icon='TRIA_DOWN' if prefs.ui_texture else 'TRIA_RIGHT', emboss=False)
        if prefs.ui_texture:
            col = box.column(align=True)
            op = col.operator("image_editor_tools.tool_start", text="法线生成", icon='NORMALS_FACE')
            op.tool_id = 'normal'
            op = col.operator("image_editor_tools.tool_start", text="法线还原高度", icon='MESH_GRID')
            op.tool_id = 'height_from_normal'
            op = col.operator("image_editor_tools.tool_start", text="法线转换", icon='NORMALS_FACE')
            op.tool_id = 'normal_convert'
            op = col.operator("image_editor_tools.tool_start", text="高度→AO", icon='SHADING_SOLID')
            op.tool_id = 'height_to_ao'
            op = col.operator("image_editor_tools.tool_start", text="曲率图", icon='CURVE_DATA')
            op.tool_id = 'curvature'

            op = col.operator("image_editor_tools.tool_start", text="贴图无缝化", icon='MOD_TRIANGULATE')
            op.tool_id = 'seamless'
            op = col.operator("image_editor_tools.tool_start", text="黑底抠图", icon='IMAGE_ALPHA')
            op.tool_id = 'rebuild_alpha'
            op = col.operator("image_editor_tools.tool_start", text="图像合成", icon='NODE_COMPOSITING')
            op.tool_id = 'composite'
            op = col.operator("image_editor_tools.tool_start", text="均匀光影", icon='LIGHT')
            op.tool_id = 'relight'
            op = col.operator("image_editor_tools.tool_start", text="通道修改", icon='TEXTURE')
            op.tool_id = 'channel'            


        # ── 几何变形 ──
        box = layout.box()
        row = box.row(align=True)
        row.prop(prefs, "ui_warp", text="几何变形", icon='TRIA_DOWN' if prefs.ui_warp else 'TRIA_RIGHT', emboss=False)
        if prefs.ui_warp:
            col = box.column(align=True)
            op = col.operator("image_editor_tools.mesh_warp_modal", text="贝塞尔扭曲", icon='MOD_LATTICE')
            op = col.operator("image_editor_tools.perspective_warp_modal", text="透视形变", icon='MESH_GRID')
            op = col.operator("image_editor_tools.free_crop_modal", text="自由裁切", icon='BORDERMOVE')
            op = col.operator("image_editor_tools.place_image_modal", text="置入图像", icon='IMAGE_DATA')
            op = col.operator("image_editor_tools.place_text_modal", text="置入文字", icon='FONT_DATA')
            op = col.operator("image_editor_tools.slice_image_modal", text="切分图像", icon='GRID')


class IMAGEEDITOR_TOOLS_PT_ToolPanel(bpy.types.Panel):
    bl_label = "工具面板"
    bl_idname = "IMAGEEDITOR_TOOLS_PT_ToolPanel"
    bl_space_type = 'IMAGE_EDITOR'
    bl_region_type = 'UI'
    bl_category = "Tool"

    @classmethod
    def poll(cls, context):
        return (
            context.space_data.image is not None
            and state.current_tool != 'NONE'
        )

    def draw_header(self, context):
        tool = TOOLS.get(state.current_tool)
        if tool:
            self.layout.label(text=tool.label)
        elif state.current_tool.startswith('warp:'):
            self.layout.label(text=state.current_tool[5:], icon='MOD_LATTICE')

    def draw(self, context):
        layout = self.layout
        props = context.scene.image_editor_tools
        if state.current_tool.startswith('warp:'):
            if state.current_tool == 'warp:置入图像':
                row = layout.row(align=True)
                row.prop(props, "place_img_fg", text="前景图")
                op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
                op.target_prop = "place_img_fg"
                if props.place_img_fg is not None:
                    layout.prop(props, "place_img_mode", text="模式")
                    layout.prop(props, "place_img_opacity", text="不透明度", slider=True)
                else:
                    layout.label(text="请选择一幅前景图", icon='INFO')
                layout.separator()
            elif state.current_tool == 'warp:置入文字':
                layout.prop(props, "place_text_content", text="文字")
                layout.prop(props, "place_text_font_path", text="字体")
                row = layout.row(align=True)
                row.prop(props, "place_text_font_size", text="字号")
                row.prop(props, "place_text_color", text="")
                row = layout.row(align=True)
                row.prop(props, "place_text_letter_spacing", text="字间距")
                row.prop(props, "place_text_italic_angle", text="倾斜")
                layout.prop(props, "place_text_padding", text="出血", slider=True)
                if props.place_text_content:
                    layout.prop(props, "place_text_mode", text="模式")
                    layout.prop(props, "place_text_opacity", text="不透明度", slider=True)
                else:
                    layout.label(text="请输入文字内容", icon='INFO')
                layout.separator()
            elif state.current_tool == 'warp:切分图像':
                layout.prop(props, "slice_cols", text="纵向")
                layout.prop(props, "slice_rows", text="横向")
                layout.separator()
            row = layout.row(align=True)
            row.operator("image_editor_tools.warp_cancel", text="取消", icon='X')
            row.operator("image_editor_tools.warp_apply", text="应用", icon='CHECKMARK')
            row.operator("image_editor_tools.warp_save_as", text="另存", icon='DUPLICATE')
            return
        tool = TOOLS.get(state.current_tool)
        if tool is None:
            return
        tool.draw_panel(layout, props)
        layout.separator()
        row = layout.row(align=True)
        row.operator("image_editor_tools.tool_cancel", text="取消", icon='X')
        row.operator("image_editor_tools.tool_save", text="保存", icon='CHECKMARK')
        row.operator("image_editor_tools.tool_save_as", text="另存", icon='DUPLICATE')


class IMAGEEDITOR_TOOLS_Preferences(bpy.types.AddonPreferences):
    bl_idname = __package__

    ui_color: bpy.props.BoolProperty(
        name="色调调整", default=True,
        description="在工具面板中展开色调调整分类",
    )
    ui_filter: bpy.props.BoolProperty(
        name="滤镜效果", default=True,
        description="在工具面板中展开滤镜效果分类",
    )
    ui_texture: bpy.props.BoolProperty(
        name="贴图处理", default=True,
        description="在工具面板中展开贴图处理分类",
    )
    ui_warp: bpy.props.BoolProperty(
        name="几何变形", default=True,
        description="在工具面板中展开几何变形分类",
    )
    show_node_menu: bpy.props.BoolProperty(
        name="节点编辑器右键菜单",
        description="当在节点编辑器里，选中图像节点时，右键菜单会增加一个快速跳转并修改图像的按钮",
        default=True,
    )

    def draw(self, context):
        layout = self.layout
        layout.label(text="面板分类（默认展开）", icon='COLLAPSEMENU')
        row = layout.row(align=True)
        row.prop(self, "ui_color")
        row.prop(self, "ui_filter")
        row = layout.row(align=True)
        row.prop(self, "ui_texture")
        row.prop(self, "ui_warp")
        layout.separator()
        row = layout.row(align=True)
        row.prop(self, "show_node_menu")
        row.label(text="当在节点编辑器里，选中图像节点时，右键菜单会增加一个快速跳转并修改图像的按钮")


def _get_prefs(context):
    return context.preferences.addons[__package__].preferences


classes = [
    IMAGEEDITOR_TOOLS_PG_Properties,
    IMAGEEDITOR_TOOLS_Preferences,
    IMAGEEDITOR_TOOLS_PT_MainPanel,
    IMAGEEDITOR_TOOLS_PT_ToolPanel,
] + operator_classes


def register():
    try:
        translation.register()
        state.current_tool = 'NONE'
        for tool_id, tool_cls in TOOLS.items():
            for name, prop in tool_cls.get_properties().items():
                IMAGEEDITOR_TOOLS_PG_Properties.__annotations__[name] = prop
        for cls in classes:
            bpy.utils.register_class(cls)
        bpy.types.Scene.image_editor_tools = bpy.props.PointerProperty(
            type=IMAGEEDITOR_TOOLS_PG_Properties
        )
        warp.register()
        ops.register()
    except Exception:
        traceback.print_exc()
        raise


def unregister():
    state.current_tool = 'NONE'
    try:
        translation.unregister()
    except Exception:
        pass
    try:
        ops.unregister()
    except Exception:
        pass
    try:
        warp.unregister()
    except Exception:
        pass
    try:
        engine = PreviewEngine._active_instance
        if engine is not None:
            engine.cleanup()
    except Exception:
        pass
    try:
        giu.clear_shader_cache()
        for _nm in ('._tool_preview_temp', '._place_text_temp', '._place_text_composite'):
            if _nm in bpy.data.images:
                bpy.data.images.remove(bpy.data.images[_nm])
    except Exception:
        pass
    try:
        del bpy.types.Scene.image_editor_tools
    except AttributeError:
        pass
    for cls in reversed(classes):
        try:
            bpy.utils.unregister_class(cls)
        except Exception:
            pass


if __name__ == "__main__":
    register()

import bpy
import numpy as np
from .base import BaseTool
from . import _on_param_update

CH_SRC_ITEMS = [
    ('R', "R 通道", ""),
    ('G', "G 通道", ""),
    ('B', "B 通道", ""),
    ('A', "A 通道", ""),
    ('RGB', "RGB 转灰度", ""),
]


class ChannelTool(BaseTool):
    tool_id = 'channel'
    label = '通道修改'

    @staticmethod
    def get_properties():
        return {
            'ch_mode': bpy.props.EnumProperty(
                name="模式", items=[
                    ('SPLIT', "拆分通道", ""),
                    ('MERGE', "合并通道", ""),
                ], default='SPLIT', update=_on_param_update,
            ),
            'ch_view': bpy.props.EnumProperty(
                name="预览通道", items=[
                    ('R', "红", ""), ('G', "绿", ""),
                    ('B', "蓝", ""), ('A', "阿尔法", ""),
                ], default='R', update=_on_param_update,
            ),
            'ch_img_r': bpy.props.PointerProperty(
                name="图 R", type=bpy.types.Image, update=_on_param_update,
            ),
            'ch_img_g': bpy.props.PointerProperty(
                name="图 G", type=bpy.types.Image, update=_on_param_update,
            ),
            'ch_img_b': bpy.props.PointerProperty(
                name="图 B", type=bpy.types.Image, update=_on_param_update,
            ),
            'ch_img_a': bpy.props.PointerProperty(
                name="图 A", type=bpy.types.Image, update=_on_param_update,
            ),
            'ch_src_r': bpy.props.EnumProperty(
                name="R 源", items=CH_SRC_ITEMS, default='R', update=_on_param_update,
            ),
            'ch_src_g': bpy.props.EnumProperty(
                name="G 源", items=CH_SRC_ITEMS, default='G', update=_on_param_update,
            ),
            'ch_src_b': bpy.props.EnumProperty(
                name="B 源", items=CH_SRC_ITEMS, default='B', update=_on_param_update,
            ),
            'ch_src_a': bpy.props.EnumProperty(
                name="A 源", items=CH_SRC_ITEMS, default='A', update=_on_param_update,
            ),
        }

    @staticmethod
    def draw_panel(layout, props):
        layout.prop(props, "ch_mode")
        if props.ch_mode == 'SPLIT':
            layout.prop(props, "ch_view", text="预览")
        else:
            for label, img_key, src_key in [
                ("R 通道", "ch_img_r", "ch_src_r"),
                ("G 通道", "ch_img_g", "ch_src_g"),
                ("B 通道", "ch_img_b", "ch_src_b"),
                ("A 通道", "ch_img_a", "ch_src_a"),
            ]:
                box = layout.box()
                row = box.row(align=True)
                row.prop(props, img_key, text=label)
                op = row.operator("image_editor_tools.clipboard_paste_to_prop", text="", icon='PASTEDOWN')
                op.target_prop = img_key
                box.prop(props, src_key, text="来源")

    @staticmethod
    def process(np_array, props):
        if props.ch_mode == 'SPLIT':
            return _split_preview(np_array, props.ch_view)
        return _merge_preview(np_array, props)

    @staticmethod
    def on_apply(full_np, props, save_mode):
        if props.ch_mode == 'SPLIT':
            _do_split(full_np, props)
            return full_np
        return None


def _split_preview(np_array, ch):
    idx = {'R': 0, 'G': 1, 'B': 2, 'A': 3}[ch]
    val = np_array[:, :, idx].copy()
    result = np.zeros_like(np_array)
    if ch == 'A':
        result[:, :, 0] = val
        result[:, :, 1] = val
        result[:, :, 2] = val
        result[:, :, 3] = val
    else:
        result[:, :, idx] = val
        result[:, :, 3] = 1.0
    return result


def _merge_preview(np_array, props):
    img_w, img_h = np_array.shape[:2]
    out = np.zeros_like(np_array)

    for ch_idx, img_key, src_key in [
        (0, 'ch_img_r', 'ch_src_r'),
        (1, 'ch_img_g', 'ch_src_g'),
        (2, 'ch_img_b', 'ch_src_b'),
        (3, 'ch_img_a', 'ch_src_a'),
    ]:
        src_img = getattr(props, img_key)
        src_mode = getattr(props, src_key)
        arr = _get_channel_data(np_array, src_img, src_mode, img_w, img_h)
        out[:, :, ch_idx] = arr if ch_idx < 3 else arr

    rgb = out[:, :, :3]
    has_alpha = (rgb[:, :, 0] != 0) | (rgb[:, :, 1] != 0) | (rgb[:, :, 2] != 0) | (out[:, :, 3] != 0)
    out[:, :, 3] = np.where(has_alpha, out[:, :, 3], 1.0)
    return out


def _get_channel_data(self_np, src_img, src_mode, target_w, target_h):
    if src_img is None:
        arr = self_np
    else:
        from ..utils.np_img_utils import blimg_2_npimg, np_resize_img
        arr = blimg_2_npimg(src_img)
        sh, sw = arr.shape[:2]
        if sh != target_h or sw != target_w:
            arr = np_resize_img(arr, target_w, target_h)

    if src_mode == 'RGB':
        return (arr[:, :, 0] * 0.299 + arr[:, :, 1] * 0.587 + arr[:, :, 2] * 0.114)
    idx = {'R': 0, 'G': 1, 'B': 2, 'A': 3}[src_mode]
    return arr[:, :, idx].copy()


def _do_split(np_array, props):
    from ..utils.np_img_utils import npimg_2_blimg
    base = bpy.context.space_data.image
    if base is None:
        return
    base_name = base.name
    for ch, name_suffix in [('R', 'R'), ('G', 'G'), ('B', 'B'), ('A', 'A')]:
        ch_np = _split_preview(np_array, ch)
        npimg_2_blimg(ch_np, f"{base_name}_{name_suffix}", True)

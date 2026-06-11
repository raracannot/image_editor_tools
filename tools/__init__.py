from .preview_engine import PreviewEngine


def _on_param_update(self, context):
    engine = PreviewEngine._active_instance
    if engine is None:
        return
    engine._process_preview()


from .blur import BlurTool
from .seamless import SeamlessTool
from .normal import NormalTool
from .sharpen import SharpenTool
from .noise import NoiseTool
from .color import ColorTool
from .depth import DepthTool
from .height_from_normal import HeightFromNormalTool
from .color_transfer import ColorTransferTool
from .edge_detect import EdgeDetectTool
from .posterize import PosterizeTool
from .composite import CompositeTool
from .rebuild_alpha import RebuildAlphaTool
from .offset import OffsetTool
from .mosaic import MosaicTool
from .emboss import EmbossTool
from .denoise import DenoiseTool
from .halftone import HalftoneTool
from .mixer import MixerTool
from .high_pass import HighPassTool
from .crystallize import CrystallizeTool
from .invert import InvertTool
from .levels import LevelsTool
from .channel_mixer import ChannelMixerTool
from .channel import ChannelTool
from .color_replace import ColorReplaceTool
from .operators import classes as operator_classes

TOOLS = {
    'blur': BlurTool,
    'seamless': SeamlessTool,
    'normal': NormalTool,
    'sharpen': SharpenTool,
    'noise': NoiseTool,
    'color': ColorTool,
    'depth': DepthTool,
    'height_from_normal': HeightFromNormalTool,
    'color_transfer': ColorTransferTool,
    'edge_detect': EdgeDetectTool,
    'posterize': PosterizeTool,
    'composite': CompositeTool,
    'rebuild_alpha': RebuildAlphaTool,
    'offset': OffsetTool,
    'mosaic': MosaicTool,
    'emboss': EmbossTool,
    'denoise': DenoiseTool,
    'halftone': HalftoneTool,
    'mixer': MixerTool,
    'high_pass': HighPassTool,
    'crystallize': CrystallizeTool,
    'invert': InvertTool,
    'levels': LevelsTool,
    'channel_mixer': ChannelMixerTool,
    'channel': ChannelTool,
    'color_replace': ColorReplaceTool,
}

#!/usr/bin/env python3
"""
[Blender超级技术交流社] 开源图像辅助处理工具库
作者：来一点咖啡吗（RARA）
https://space.bilibili.com/27284213
https://extensions.blender.org/author/58486/
欢迎任意修改，任意使用

 - 版本号：v1.2.0
 - 更新时间：2026-05-09

适用于Blender的轻量级图像处理工具集。
内部计算统一使用 float32 [0,1] 空间，uint8 用户需手动调用 np_u8_to_f32() / np_f32_to_u8()。

适配环境:
- Blender版本：2.80+ (兼容所有新版Blender)
- Python版本：3.7+ (Blender内置Python环境)
- 依赖：Python标准库 + Blender内置库（NumPy, OpenImageIO），无第三方依赖

导入使用示例:
import bpy
from np_img_utils import blimg_2_npimg, npimg_2_blimg
from np_img_utils import np_zoom, np_rotate, np_gaussian_filter, np_sobel_edge
"""

from __future__ import annotations

import bpy
import numpy as np
import math
import os
import OpenImageIO as oiio
# 本库所有计算函数只接受/返回 float32 [0,1]，
# uint8 用户通过这两个函数显式转换：
# ══════════════════════════════════════════════════════════════════════════════

def np_u8_to_f32(img: np.ndarray) -> np.ndarray:
    """uint8 [0,255] → float32 [0,1]，已是 float32 则原样返回"""
    if img.dtype == np.float32:
        return img
    return img.astype(np.float32) / 255.0


def np_f32_to_u8(img: np.ndarray) -> np.ndarray:
    """float32 [0,1] → uint8 [0,255]（含四舍五入），已是 uint8 则原样返回"""
    if img.dtype == np.uint8:
        return img
    # +0.5 实现四舍五入（而非截断），避免系统性偏暗
    return np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)


# ══════════════════════════════════════════════════════════════════════════════
# Blender 图像 ↔ Numpy 数组互转（仅 float32 RGBA）
# ══════════════════════════════════════════════════════════════════════════════

def blimg_2_npimg(blender_image: bpy.types.Image) -> np.ndarray:
    """将 Blender Image 对象转为 numpy RGBA 数组 (H, W, 4)，float32 [0,1]"""
    if not isinstance(blender_image, bpy.types.Image):
        raise TypeError("输入必须是Blender图像对象（bpy.types.Image）")
    img_w, img_h = blender_image.size
    pixel_flat = np.empty(img_w * img_h * 4, dtype=np.float32)
    blender_image.pixels.foreach_get(pixel_flat)  # 高效批量读取
    return pixel_flat.reshape((img_h, img_w, 4))


def npimg_2_blimg(np_image: np.ndarray, image_name: str, overwrite: bool = True, pack: bool = True,) -> bpy.types.Image: 
    # 校验
    if np_image.dtype != np.float32:
        np_image = np_image.astype(np.float32)
    if np_image.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道！当前通道数：{np_image.shape[-1]}（要求：4）")
    h, w = np_image.shape[:2]
    np_image = np.clip(np_image, 0.0, 1.0)
    
    if (image_name in bpy.data.images) and overwrite:
        # 如果已经存在且开启覆写
        bl_image = bpy.data.images[image_name]
        # 仅尺寸不同才需要resize
        if bl_image.size[0] != w or bl_image.size[1] != h:
            bl_image.scale(w, h)
    else:
        #如果不存在或者没开启覆写则新建，（新建会自动避让旧名称无需额外处理）
        bl_image = bpy.data.images.new(name=image_name, width=w, height=h, alpha=True)
    
    bl_image.pixels.foreach_set(np_image.ravel()) # 高效批量写入
    bl_image.update()
    if pack:
        bl_image.pack()
    return bl_image


# ══════════════════════════════════════════════════════════════════════════════
# RGB ↔ HSV 色彩空间转换（仅 float32 [0,1]）
# ══════════════════════════════════════════════════════════════════════════════

def np_rgb_to_hsv(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """float32 RGB (H,W,3) [0,1] → (H, S, V)，各分量独立 ndarray"""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    max_c = np.maximum(np.maximum(r, g), b)
    min_c = np.minimum(np.minimum(r, g), b)
    delta = max_c - min_c
    delta = np.maximum(delta, np.float32(1e-10))
    max_c = np.maximum(max_c, np.float32(1e-10))
    v = max_c
    s = np.where(max_c > 1e-6, delta / max_c, 0.0)
    h = np.zeros_like(max_c)
    mask = delta > 1e-6
    r_mask = mask & (max_c == r)
    g_mask = mask & (max_c == g) & ~r_mask
    b_mask = mask & ~r_mask & ~g_mask
    h = np.where(r_mask, 60.0 * (((g - b) / delta) % 6.0), h)
    h = np.where(g_mask, 60.0 * ((b - r) / delta + 2.0), h)
    h = np.where(b_mask, 60.0 * ((r - g) / delta + 4.0), h)
    return (h / 360.0) % 1.0, s, v


def np_hsv_to_rgb(h: np.ndarray, s: np.ndarray, v: np.ndarray) -> np.ndarray:
    """(H, S, V) → float32 RGB (H,W,3) [0,1]"""
    h6 = (h % 1.0) * 6.0
    i = np.floor(h6).astype(np.int32) % 6
    f = h6 - np.floor(h6)
    p = v * (1.0 - s)
    q = v * (1.0 - f * s)
    t = v * (1.0 - (1.0 - f) * s)
    r = np.where((i == 0) | (i == 5), v,
         np.where((i == 1), q,
         np.where((i == 2) | (i == 3), p, t)))
    g = np.where((i == 1) | (i == 2), v,
         np.where((i == 3), q,
         np.where((i == 4) | (i == 5), p, t)))
    b = np.where((i == 3) | (i == 4), v,
         np.where((i == 5), q,
         np.where((i == 0) | (i == 1), p, t)))
    return np.stack([r, g, b], axis=-1)


# ══════════════════════════════════════════════════════════════════════════════
# OIIO 本地图像文件 IO（内部处理 uint8 ↔ float32，对外统一 float32 [0,1]）
# ══════════════════════════════════════════════════════════════════════════════

def lcimg_2_npimg(filepath: str) -> np.ndarray:
    """
    从本地文件加载图像到 float32 [0,1] numpy 数组 (H,W,3)。
    内部通过 OIIO 以 uint8 读取后自动转换。
    单通道自动扩展为 3 通道灰度，4+ 通道取前 3 通道。
    """
    buf = oiio.ImageBuf(filepath)
    if buf.has_error:
        raise IOError(f"无法打开图像: {buf.geterror()}")
    img = buf.get_pixels(oiio.TypeDesc("uint8"))
    if img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)  # 灰度 → RGB
    elif img.shape[2] > 3:
        img = img[..., :3]  # 丢弃 Alpha 及额外通道
    return img.astype(np.float32) / 255.0


def npimg_2_lcimg(img: np.ndarray, filepath: str) -> None:
    """
    将 float32 [0,1] numpy 数组保存为本地图像文件。
    内部自动转换为 uint8 后通过 OIIO 写入。
    """
    img_u8 = np.clip(img * 255.0 + 0.5, 0, 255).astype(np.uint8)
    h, w = img_u8.shape[:2]
    c = img_u8.shape[2] if img_u8.ndim == 3 else 1
    spec = oiio.ImageSpec(w, h, c, oiio.TypeDesc("uint8"))
    buf = oiio.ImageBuf(spec)
    buf.set_pixels(oiio.ROI(), img_u8)
    out_dir = os.path.dirname(filepath)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    buf.write(filepath)


# ══════════════════════════════════════════════════════════════════════════════
# 缩放系列（仅 float32 RGBA）
# 实现：双线性插值，纯 NumPy 向量化
# ══════════════════════════════════════════════════════════════════════════════

def np_resize_img(
    img_data: np.ndarray,
    target_w: float,
    target_h: float
) -> np.ndarray:
    """
    双线性插值缩放到指定尺寸。
    输入/输出：float32 RGBA (H,W,4)，值域 [0,1]
    """
    if img_data.ndim != 3 or img_data.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道数组！当前形状：{img_data.shape}（要求：(H,W,4)）")
    if img_data.dtype != np.float32:
        img_data = img_data.astype(np.float32)

    h_ori, w_ori = img_data.shape[:2]
    target_h = int(round(target_h))
    target_w = int(round(target_w))

    # 生成目标坐标网格，映射到原始图的浮点坐标
    y_target = np.linspace(0, h_ori - 1, target_h, dtype=np.float32)
    x_target = np.linspace(0, w_ori - 1, target_w, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(x_target, y_target)

    # 四邻域像素坐标
    x0 = np.floor(x_grid).astype(np.int32)
    x1 = np.minimum(x0 + 1, w_ori - 1)
    y0 = np.floor(y_grid).astype(np.int32)
    y1 = np.minimum(y0 + 1, h_ori - 1)

    # 插值权重
    dx = (x_grid - x0).astype(np.float32)
    dy = (y_grid - y0).astype(np.float32)

    # 四邻域颜色值
    val00 = img_data[y0, x0, :]
    val01 = img_data[y0, x1, :]
    val10 = img_data[y1, x0, :]
    val11 = img_data[y1, x1, :]

    # 双线性插值：先水平后垂直
    val_x0 = val00 * (1 - dx[..., np.newaxis]) + val01 * dx[..., np.newaxis]
    val_x1 = val10 * (1 - dx[..., np.newaxis]) + val11 * dx[..., np.newaxis]
    resized = val_x0 * (1 - dy[..., np.newaxis]) + val_x1 * dy[..., np.newaxis]

    return np.clip(resized, 0.0, 1.0)


def np_clamp_image_size(np_image: np.ndarray, max_side: int = 1024) -> np.ndarray:
    """
    将图像长边钳制到 max_side 内（等比缩放）。
    长边 ≤ max_side 时返回原图副本。
    输入/输出：float32 RGBA (H,W,4)
    """
    if np_image.ndim != 3 or np_image.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道数组！当前形状：{np_image.shape}（要求：(H,W,4)）")
    if np_image.dtype != np.float32:
        np_image = np_image.astype(np.float32)

    max_side = max(int(round(max_side)), 1)
    h_ori, w_ori = np_image.shape[:2]

    if max(h_ori, w_ori) <= max_side:
        return np_image.copy()

    scale_ratio = max_side / max(h_ori, w_ori)
    target_w = int(round(w_ori * scale_ratio))
    target_h = int(round(h_ori * scale_ratio))
    return np_resize_img(np_image, target_w, target_h)


def np_resize_img_by_scale(img_data: np.ndarray, scale_factor: float) -> np.ndarray:
    """按比例缩放图像。scale_factor > 0，1.0 时返回原图副本"""
    if scale_factor <= 0:
        raise ValueError(f"缩放系数必须大于0！当前值：{scale_factor}")
    if img_data.ndim != 3 or img_data.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道数组！当前形状：{img_data.shape}（要求：(H,W,4)）")
    if img_data.dtype != np.float32:
        img_data = img_data.astype(np.float32)

    if scale_factor == 1.0:
        return img_data.copy()

    h_ori, w_ori = img_data.shape[:2]
    target_w = max(1, int(round(w_ori * scale_factor)))
    target_h = max(1, int(round(h_ori * scale_factor)))
    return np_resize_img(img_data, target_w, target_h)


# ══════════════════════════════════════════════════════════════════════════════
# 盒式模糊算法（多线程，仅 float32 RGBA）
# 使用滑动窗口积分图（前缀和）实现 O(1) 半径复杂度
# 原理：对 padded 数组做二维前缀和，任意窗口和 = 四角前缀和差值，
#       然后除以窗口面积得到均值，再按亮度比缩放保持整体亮度。
# ══════════════════════════════════════════════════════════════════════════════

def _process_single_channel(
    channel_data: np.ndarray,
    radius: int,
    mode: str
) -> np.ndarray:
    """
    单通道模糊处理：亮度保持型盒式模糊（内部函数，由线程池调用）。
    核心原理：
    1. 积分图实现 O(1) 窗口均值
    2. 模糊后按亮度比缩放，防止平均降采样导致的整体变暗
    """
    channel_mean = np.mean(channel_data)
    channel_mean = channel_mean if channel_mean != 0 else 1e-8

    # 边界填充（不同模式对应不同无缝策略）
    if mode == "edge":
        padded = np.pad(channel_data, radius, mode="edge")
    elif mode == "wrap":
        padded = np.pad(channel_data, radius, mode="wrap")
    elif mode == "constant":
        # 根据通道亮度智能选择填充值：亮通道填 1.0，暗通道填均值
        fill_val = 1.0 if channel_mean > 0.9 else channel_mean
        padded = np.pad(channel_data, radius, mode="constant", constant_values=fill_val)
    else:
        padded = np.pad(channel_data, radius, mode="edge")

    # 二维积分图（先水平前缀和 → 再垂直前缀和）
    # 步骤：行方向前缀和 → 相邻两列相减得行窗口和 → 列方向前缀和 → 相邻两行相减得最终窗口和
    row_cumsum = np.cumsum(padded, axis=1)
    row_sum = row_cumsum[:, 2 * radius:] - row_cumsum[:, :-2 * radius]
    col_cumsum = np.cumsum(row_sum, axis=0)
    col_sum = col_cumsum[2 * radius:, :] - col_cumsum[:-2 * radius, :]

    window_area = (2 * radius + 1) ** 2
    blurred_channel = (col_sum / window_area).astype(np.float32)

    # 亮度还原：模糊前后的均值比，防止平均化导致的亮度损失
    blur_channel_mean = np.mean(blurred_channel)
    blur_channel_mean = blur_channel_mean if blur_channel_mean != 0 else 1e-8
    brightness_scale = channel_mean / blur_channel_mean
    return np.clip(blurred_channel * brightness_scale, 0.0, 1.0)


def np_blur_img(
    image_array: np.ndarray,
    blur_percent: float = 1.0,
    mode: str = "edge"
) -> np.ndarray:
    """
    多线程盒式模糊（通道级并行）。
    blur_percent: 0~100，基于图像对角线比例计算模糊半径
    mode: edge / wrap / constant（边界填充模式）
        - edge:   边缘像素重复（默认，适合大多数场景）
        - wrap:   循环平铺（适合无缝贴图预处理）
        - constant: 智能填充（亮图填白，暗图填均值）
    输入/输出：float32 RGBA (H,W,C) 或 (H,W)，值域 [0,1]
    """
    if not (0.0 <= blur_percent <= 100.0):
        raise ValueError("模糊百分比必须在0~100之间")
    if mode not in ["edge", "wrap", "constant"]:
        raise ValueError(f"不支持的填充模式：{mode}")
    if blur_percent == 0.0:
        return image_array.copy()

    is_single_channel = len(image_array.shape) == 2
    if is_single_channel:
        img = image_array[..., np.newaxis].astype(np.float32)
    else:
        # 避免不必要的 copy + astype
        if image_array.dtype == np.float32:
            img = image_array.copy()
        else:
            img = image_array.astype(np.float32)

    H, W, C = img.shape

    # 半径 = 对角线/4 * blur_percent/100
    # 对角线/4 作为基准长度，保证 100% 时半径为对角线的四分之一（适度模糊）
    diagonal = math.hypot(W, H)
    base_length = diagonal / 4
    radius_float = base_length * blur_percent / 100
    radius = max(0, min(int(round(radius_float)), int(base_length)))
    if radius == 0:
        return image_array.copy()

    # 通道顺序处理
    blurred_img = np.zeros_like(img)
    for c in range(C):
        try:
            blurred_img[..., c] = _process_single_channel(img[..., c], radius, mode)
        except Exception as e:
            print(f"通道{c}计算失败：{str(e)}")
            raise e

    if is_single_channel:
        blurred_img = blurred_img[..., 0]
    return blurred_img


def np_stacked_box_blur(image_array: np.ndarray, blur_percent: float = 1.0,
                        mode: str = "edge", passes: int = 4) -> np.ndarray:
    """堆叠盒式模糊：多次盒式模糊叠加，逼近高斯质量。
    passes 次积分图盒式模糊，每次半径递增但面积守恒。
    输入/输出：float32 [0,1]"""
    if passes <= 1 or blur_percent <= 0:
        return np_blur_img(image_array, blur_percent, mode)
    radii = []
    factor = 3.0 / (passes * (passes + 1))
    for k in range(1, passes + 1):
        radii.append(blur_percent * k * factor)
    result = image_array
    for r in radii:
        result = np_blur_img(result, r, mode)
    return result


def np_kawase_blur(image_array: np.ndarray, blur_percent: float = 1.0,
                   mode: str = "edge", iterations: int = 5) -> np.ndarray:
    """Kawase 模糊：多轮小半径迭代，产生柔光发光效果。
    每轮半径固定为 1 像素，叠加 iterations 轮。
    输入/输出：float32 [0,1]"""
    if iterations <= 0 or blur_percent <= 0:
        return image_array.copy()
    iters = min(iterations, max(1, int(blur_percent)))
    result = image_array
    for _ in range(iters):
        result = np_blur_img(result, 1.0, mode)
    return result


def np_directional_blur(image_array: np.ndarray, angle_deg: float = 0.0,
                        blur_percent: float = 1.0, mode: str = "wrap") -> np.ndarray:
    """方向模糊：沿指定角度平移全图 N 次后取平均，通道级并行。
    angle_deg: 模糊方向角度 (-180~180)，0=水平，90=垂直。
    mode: 'wrap' 循环补边 / 'edge' 边缘重复 / 'constant' 填黑。
    输入/输出：float32 [0,1]
    """
    if blur_percent <= 0:
        return image_array.copy()
    img = np.ascontiguousarray(image_array, dtype=np.float32)
    h, w = img.shape[:2]
    angle_rad = math.radians(angle_deg)
    dx = math.cos(angle_rad)
    dy = -math.sin(angle_rad)

    diagonal = math.hypot(w, h)
    radius = max(1, int(diagonal * blur_percent / 400))
    weight = np.float32(1.0 / (2 * radius + 1))

    shifts = []
    for i in range(1, radius + 1):
        sdy = int(round(dy * i))
        sdx = int(round(dx * i))
        if sdy == 0 and sdx == 0:
            continue
        shifts.append((sdy, sdx))

    if mode == 'wrap':
        def _shift(ch, sy, sx):
            return np.roll(ch, (sy, sx), axis=(0, 1))
    else:
        rows = np.arange(h)
        cols = np.arange(w)
        fill = np.float32(0) if mode == 'constant' else None

        def _shift(ch, sy, sx):
            src_y = rows - sy
            src_x = cols - sx
            if mode == 'edge':
                src_y = np.clip(src_y, 0, h - 1)
                src_x = np.clip(src_x, 0, w - 1)
            elif mode == 'constant':
                valid_y = (src_y >= 0) & (src_y < h)
                valid_x = (src_x >= 0) & (src_x < w)
                src_y = np.clip(src_y, 0, h - 1)
                src_x = np.clip(src_x, 0, w - 1)
                result = np.full_like(ch, fill)
                result[np.ix_(valid_y, valid_x)] = ch[src_y[:, None], src_x[None, :]][np.ix_(valid_y, valid_x)]
                return result
            return ch[src_y[:, None], src_x[None, :]]

    def _blur_channel(ch):
        r = ch.astype(np.float32, copy=True)
        for sdy, sdx in shifts:
            r += _shift(ch, sdy, sdx)
            r += _shift(ch, -sdy, -sdx)
        r *= weight
        return r

    C = img.shape[-1]
    result = np.empty_like(img)
    for c in range(C):
        result[..., c] = _blur_channel(img[..., c])
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 颜色差值计算（仅 float32 RGBA）
# ══════════════════════════════════════════════════════════════════════════════

def np_contrast_img(
    blurred_npimg: np.ndarray,
    original_npimg: np.ndarray,
    normalize: bool = True
) -> np.ndarray:
    """
    计算 模糊色 - 原始色 的像素级差异。
    normalize=True 时将差值归一化到 [0,1] 便于可视化（灰色=0差异）。
    Alpha 通道保持原值不变。
    输入/输出：float32 RGBA (H,W,4)
    """
    if blurred_npimg.shape != original_npimg.shape:
        raise ValueError(
            f"模糊图像和原始图像形状不匹配！模糊图：{blurred_npimg.shape}，原始图：{original_npimg.shape}"
        )
    if blurred_npimg.shape[-1] != 4 or original_npimg.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道！当前形状：{blurred_npimg.shape}")

    diff_npimg = blurred_npimg - original_npimg

    if normalize:
        # 取所有像素各通道的最大绝对值作为归一化分母
        max_abs_diff = np.max(np.abs(diff_npimg))
        if max_abs_diff > 0:
            # 映射到 [0,1]：0 表示 -max_abs_diff（最大负差），1 表示 +max_abs_diff（最大正差）
            diff_npimg = (diff_npimg / max_abs_diff + 1.0) / 2.0
        else:
            # 完全无差异：返回全 0.5 灰色（仅 RGB 三通道）
            diff_npimg[..., :3] = 0.5

    # 保留原始 Alpha 通道
    diff_npimg[..., 3] = original_npimg[..., 3]
    return np.clip(diff_npimg, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Alpha 通道重建
# ══════════════════════════════════════════════════════════════════════════════

def np_rebuild_alpha(rgba_data: np.ndarray) -> np.ndarray:
    """
    基于 RGB 最大值重建 Alpha：alpha = max(R, G, B)。
    适用于去背图的重建透明度场景（如 PNG 抠图后 Alpha 丢失恢复）。
    输入/输出：float32 RGBA (H,W,4)
    """
    if rgba_data.ndim != 3 or rgba_data.shape[-1] != 4:
        raise ValueError(f"仅支持RGBA四通道数组！当前形状：{rgba_data.shape}（要求：(H,W,4)）")
    if rgba_data.dtype != np.float32:
        rgba_data = rgba_data.astype(np.float32)

    new_alpha = np.maximum(np.maximum(rgba_data[..., 0], rgba_data[..., 1]), rgba_data[..., 2])
    new_alpha = np.clip(new_alpha, 0.0, 1.0)

    new_rgba = rgba_data.copy()
    new_rgba[..., 3] = new_alpha
    return new_rgba


# ══════════════════════════════════════════════════════════════════════════════
# 高度图 → 法线贴图  /  法线贴图 → 高度图
# ══════════════════════════════════════════════════════════════════════════════

def np_height_to_normal(
    height_np: np.ndarray,
    strength: float = 2.0,
    invert: bool = False
) -> np.ndarray:
    """
    纯 NumPy 向量化实现：高度图 → 法线贴图。
    使用 Sobel 算子计算 x/y 方向梯度，构建法线向量后归一化到 [0,1]。
    输出 (H,W,4) RGBA float32 [0,1] 法线贴图（Alpha 恒为 1.0）。
    支持 2D / 3D 输入，uint8 / float32 自动转换。
    """
    if len(height_np.shape) not in [2, 3]:
        raise ValueError(f"输入数组维度必须是2D(H,W)或3D(H,W,C)，当前：{height_np.shape}")

    if height_np.dtype == np.uint8:
        gray_arr = height_np.astype(np.float32) / 255.0
    else:
        gray_arr = np.clip(height_np.astype(np.float32), 0.0, 1.0)

    # 多通道转灰度（标准 BT.601 亮度权重）
    if len(gray_arr.shape) == 3 and gray_arr.shape[-1] > 1:
        gray_arr = np.dot(gray_arr[..., :3], [0.299, 0.587, 0.114])
    elif len(gray_arr.shape) == 3 and gray_arr.shape[-1] == 1:
        gray_arr = gray_arr[..., 0]

    h, w = gray_arr.shape
    # Sobel 3×3 卷积核
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    # 滑动窗口同时计算所有像素的 3×3 邻域
    arr_padded = np.pad(gray_arr, ((1, 1), (1, 1)), mode='edge')
    window = np.lib.stride_tricks.sliding_window_view(arr_padded, (3, 3), axis=(0, 1))
    dx = np.sum(window * sobel_x, axis=(2, 3))
    dy = np.sum(window * sobel_y, axis=(2, 3))

    # 法线向量：nx = -dx*S, ny = -dy*S, nz = 1
    # 负号使得法线朝向光源方向（传统约定）
    nx = -dx * strength
    ny = -dy * strength
    nz = np.ones_like(gray_arr)
    norm = np.sqrt(nx**2 + ny**2 + nz**2)
    norm[norm == 0] = 1.0  # 防止除零（平坦区域）
    nx /= norm
    ny /= norm
    nz /= norm

    if invert:
        ny = -ny

    # [-1,1] → [0,1] 映射到法线贴图颜色空间
    normal_r = ((nx + 1.0) * 0.5).clip(0.0, 1.0)
    normal_g = ((ny + 1.0) * 0.5).clip(0.0, 1.0)
    normal_b = ((nz + 1.0) * 0.5).clip(0.0, 1.0)
    normal_a = np.full_like(normal_r, 1.0, dtype=np.float32)

    return np.stack([normal_r, normal_g, normal_b, normal_a], axis=-1).astype(np.float32)


def np_normal_to_height_fft(normal_img: np.ndarray, flip_g: bool = False) -> np.ndarray:
    """
    法线贴图 → 高度图（FFT 求解 Poisson 方程）。
    专为【四边连续/无缝】法线贴图设计，使用周期性边界条件。

    算法流程：
    1. 从法线解码梯度场 grad_x, grad_y
    2. 计算散度 divergence = ∂(grad_x)/∂x + ∂(grad_y)/∂y
    3. FFT 频域求解 Poisson 方程 ∇²Z = divergence
    4. 逆 FFT 还原高度图并归一化到 [0,1]
    
    参数：
    normal_img : float32 [0,1] 法线贴图 (H,W,3)
    flip_g     : bool, 是否翻转 G 通道 (默认 False)。
                 用于在 False/OpenGL (Y+) 和 True/DirectX (Y-) 法线格式之间切换。
    输出：
    float32 [0,1] 高度图 (H,W)
    """
    # 解码法线向量：[0,1] → [-1,1]
    nx = normal_img[..., 0].astype(np.float32) * 2.0 - 1.0
    ny = normal_img[..., 1].astype(np.float32) * 2.0 - 1.0
    nz = np.maximum(normal_img[..., 2].astype(np.float32) * 2.0 - 1.0, 1e-6)
    if flip_g:
        ny = -ny
    # 从法线恢复梯度场：grad_x = -nx/nz, grad_y = -ny/nz
    grad_x = -nx / nz
    grad_y = -ny / nz
    grad_x -= np.mean(grad_x)  # 去均值避免 DC 分量累积
    grad_y -= np.mean(grad_y)

    height, width = grad_x.shape

    # 周期性中心差分计算散度
    # 使用 np.roll 避免边界问题（周期边界条件）
    dpdx = (np.roll(grad_x, -1, axis=1) - np.roll(grad_x, 1, axis=1)) * 0.5
    dqdy = (np.roll(grad_y, -1, axis=0) - np.roll(grad_y, 1, axis=0)) * 0.5
    divergence = dpdx + dqdy

    # FFT 求解：在频域解 Poisson 方程 ∇²Z = div
    # k2 = kx² + ky² 为频域拉普拉斯算子特征值
    # 使用 fftfreq 直接得到 FFT 序的波数（避免 ifftshift）
    kx = np.fft.fftfreq(width, d=1.0).astype(np.float32) * np.float32(2 * np.pi)
    ky = np.fft.fftfreq(height, d=1.0).astype(np.float32) * np.float32(2 * np.pi)
    k2 = kx[np.newaxis, :]**2 + ky[:, np.newaxis]**2
    k2[0, 0] = 1.0  # DC 分量置 1 避免除零

    div_fft = np.fft.fft2(divergence)
    height_fft = -div_fft / k2
    height_fft[0, 0] = 0.0 + 0.0j  # 高度 DC 置零（绝对高度无意义）

    height_map = np.real(np.fft.ifft2(height_fft))

    # 归一化到 [0, 1]
    h_min, h_max = height_map.min(), height_map.max()
    return (height_map - h_min) / (h_max - h_min + np.float32(1e-6))



# ══════════════════════════════════════════════════════════════════════════════
# 无缝贴图生成（基础版：循环偏移 + 线性渐变混合）
# ══════════════════════════════════════════════════════════════════════════════

def np_offset(img_np: np.ndarray, x_offset: int, y_offset: int) -> np.ndarray:
    """
    纯 NumPy 图像循环偏移（无缝贴图辅助工具）。
    支持水平/垂直任意像素偏移，越界部分绕回（wrap-around）。
    相当于 Photoshop 的「位移」滤镜。
    输入/输出：float32 (H,W,C)
    """
    h, w = img_np.shape[:2]
    x_offset = x_offset % w
    if x_offset != 0:
        img_np = np.concatenate([img_np[:, -x_offset:, :], img_np[:, :-x_offset, :]], axis=1)
    y_offset = y_offset % h
    if y_offset != 0:
        img_np = np.concatenate([img_np[-y_offset:, :, :], img_np[:-y_offset, :, :]], axis=0)
    return img_np


def np_make_seamless_tile(
    img_np: np.ndarray,
    blend_ratio: float = 0.125
) -> np.ndarray:
    """
    纯 NumPy 无缝拼接：先水平再垂直，使用线性渐变蒙版混合边缘。
    原理：将图像沿中心偏移一半宽度/高度，与原图在边缘区域线性混合。
    blend_ratio: 混合带宽度占短边的比例，默认 0.125（12.5%）
    输入/输出：float32 RGBA (H,W,4)
    """
    if img_np.ndim != 3 or img_np.shape[-1] != 4:
        raise ValueError(f"输入必须是RGBA四通道数组！当前形状：{img_np.shape}")
    if not (0.0 < blend_ratio <= 1.0):
        raise ValueError(f"混合系数必须在(0, 1]范围内！当前值：{blend_ratio}")

    img_np = img_np.astype(np.float32, copy=False)
    h, w = img_np.shape[:2]
    short_side = min(w, h)
    blend_width = max(2, min(int(round(short_side * blend_ratio)), short_side // 2))
    bw = blend_width // 2

    def blend_images(base_img: np.ndarray, offset_img: np.ndarray, mask_direction: str) -> np.ndarray:
        """生成线性渐变蒙版并混合两图"""
        mask = np.zeros((h, w), dtype=np.float32)
        if bw <= 0:
            return base_img
        alpha_1 = 1.0 - np.arange(bw) / bw
        alpha_2 = np.arange(bw) / bw

        if mask_direction == "horizontal":
            mask[:, :bw] = alpha_1[np.newaxis, :]
            mask[:, w - bw:w] = alpha_2[np.newaxis, :]
        elif mask_direction == "vertical":
            mask[:bw, :] = alpha_1[:, np.newaxis]
            mask[h - bw:h, :] = alpha_2[:, np.newaxis]

        mask_4ch = np.repeat(mask[:, :, np.newaxis], 4, axis=2)
        return base_img * (1 - mask_4ch) + offset_img * mask_4ch

    # 水平混合：将图像水平偏移一半，与原图在左右边缘混合
    img_h_offset = np_offset(img_np.copy(), -w // 2, 0)
    img_B = blend_images(img_np, img_h_offset, "horizontal")
    # 垂直混合：再将结果垂直偏移一半，在上下边缘混合
    img_v_offset = np_offset(img_B.copy(), 0, -h // 2)
    img_C = blend_images(img_B, img_v_offset, "vertical")

    return np.clip(img_C, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 高级无缝贴图生成（Seam Carving + 双频融合 + 梯度惩罚）
# 所有函数统一使用 float32 [0,1]
# ══════════════════════════════════════════════════════════════════════════════

def np_local_color_transfer(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    局部色彩匹配：将 source 的色彩统计量（均值/方差）对齐到 target。
    用于消除重叠区域的微小色差，防止融合后发脏。
    使用全局均值/方差的线性变换：output = (source - μ_src) * σ_tar/σ_src + μ_tar
    输入/输出：float32 (H,W,C)
    """
    mu_s = np.mean(source, axis=(0, 1), keepdims=True).astype(np.float32)
    std_s = np.std(source, axis=(0, 1), keepdims=True).astype(np.float32)
    mu_t = np.mean(target, axis=(0, 1), keepdims=True).astype(np.float32)
    std_t = np.std(target, axis=(0, 1), keepdims=True).astype(np.float32)
    std_s = np.where(std_s == 0, 1e-5, std_s)  # 防止除零
    matched = (source - mu_s) * (std_t / std_s) + mu_t
    return np.clip(matched, 0.0, 1.0)


def np_compute_gradient_penalty(L: np.ndarray, R: np.ndarray) -> np.ndarray:
    """
    计算左右区域的梯度差异（SSD of Gradients）。
    边缘结构不一致处产生高惩罚值，迫使缝合线绕开这些区域。
    用灰度图计算梯度，避免色彩空间差异对结构判断的干扰。
    """
    L_gray = np.mean(L, axis=2).astype(np.float32)
    R_gray = np.mean(R, axis=2).astype(np.float32)
    grad_L_y, grad_L_x = np.gradient(L_gray)
    grad_R_y, grad_R_x = np.gradient(R_gray)
    ssd_grad = (
        (grad_R_x.astype(np.float32) - grad_L_x.astype(np.float32)) ** 2 +
        (grad_R_y.astype(np.float32) - grad_L_y.astype(np.float32)) ** 2
    )
    return ssd_grad


def np_fast_blur_padded(img: np.ndarray, iterations: int = 5) -> np.ndarray:
    """
    快速盒式模糊（多次迭代 + 边缘填充）。
    使用 rolling average 替代卷积，多次迭代近似高斯模糊（中心极限定理）。
    特点：半径由迭代次数隐式控制，适合双频融合的低频/蒙版生成。
    注意：对于非常大半径的模糊，积分图方案（见 np_blur_img）更高效。
    """
    pad_size = iterations * 2
    padded = np.pad(
        img,
        ((pad_size, pad_size), (pad_size, pad_size), (0, 0)),
        mode='edge'
    )
    res = padded.astype(np.float32)
    two = np.float32(2.0)
    four = np.float32(4.0)
    # 多次迭代近似高斯：每次 roll 水平+垂直，逐步扩散
    for _ in range(iterations):
        res = (np.roll(res, 1, axis=1) + two * res + np.roll(res, -1, axis=1)) / four
        res = (np.roll(res, 1, axis=0) + two * res + np.roll(res, -1, axis=0)) / four
    return res[pad_size:-pad_size, pad_size:-pad_size, :]


def np_find_vertical_seam(error_matrix: np.ndarray) -> np.ndarray:
    """
    动态规划寻找垂直方向最优缝合线（全部 NumPy 向量化实现）。

    算法：Seam Carving / 最小累积能量路径
    - cost[i][j] = error[i][j] + min(cost[i-1][j-1], cost[i-1][j], cost[i-1][j+1])
    - 从顶部到底部逐行计算累积代价，反向追踪得最优路径。

    error_matrix: (H,W) 像素级误差图，值越大表示该像素越不适合作为缝合线。
    返回: 长度为 H 的数组，每行最优缝合列索引。
    """
    h, w = error_matrix.shape
    cost = np.zeros_like(error_matrix, dtype=np.float32)
    paths = np.zeros_like(error_matrix, dtype=np.int8)
    cost[0, :] = error_matrix[0, :]

    # 逐行向量化计算：用 np.roll 同时获得左/上/右三个方向的上一行代价
    for i in range(1, h):
        prev = cost[i - 1]
        left = np.roll(prev, 1)
        left[0] = np.inf          # 最左侧无左邻居
        right = np.roll(prev, -1)
        right[-1] = np.inf        # 最右侧无右邻居
        up = prev

        # 三方向取最小值
        mins = np.minimum(np.minimum(left, up), right)
        cost[i] = error_matrix[i] + mins

        # 记录路径方向：-1=左, 0=上, 1=右
        paths[i] = np.where(mins == left, -1, np.where(mins == up, 0, 1))

    # 反向追踪最优缝合线
    seam = np.zeros(h, dtype=np.int32)
    seam[-1] = np.argmin(cost[-1])
    for i in range(h - 2, -1, -1):
        seam[i] = seam[i + 1] + paths[i + 1, seam[i + 1]]
    return seam


def np_two_band_blend(
    L: np.ndarray,
    R: np.ndarray,
    seam_mask: np.ndarray
) -> np.ndarray:
    """
    双频段融合：低频用软蒙版（渐变过渡），高频用硬蒙版（保留细节）。

    原理：
    - 低频分量（模糊后的图）承载光照/色彩渐变，用软蒙版过渡避免接缝
    - 高频分量（原图 - 低频）承载纹理细节，用硬蒙版保留锐利度
    - 合并两条路径得到既无缝又保留细节的结果

    适用于 Seam Carving 后的最终融合。
    """
    L_low = np_fast_blur_padded(L, iterations=5)
    R_low = np_fast_blur_padded(R, iterations=5)
    L_high = L.astype(np.float32) - L_low
    R_high = R.astype(np.float32) - R_low

    one = np.float32(1.0)
    # 高频：硬蒙版，保留纹理锐度
    high_blended = R_high * seam_mask + L_high * (one - seam_mask)
    # 低频：软蒙版（额外模糊 8 次），平滑光照过渡
    soft_mask = np_fast_blur_padded(seam_mask, iterations=8)
    low_blended = R_low * soft_mask + L_low * (one - soft_mask)

    result = low_blended + high_blended
    return np.clip(result, 0.0, 1.0)


def np_flatten_macro_gradient(
    img: np.ndarray,
    overlap_ratio: float = 0.2
) -> np.ndarray:
    """
    全局线性梯度补偿：拟合水平和垂直方向的线性光照模型，
    消除宏观光照断层（如左暗右亮），同时保持局部对比度和 AO。

    原理：
    1. 取图像左右/上下边缘像素的平均颜色
    2. 拟合水平/垂直方向的线性光照渐变
    3. 将整体光照向中心均值补偿，消除方向性色偏

    输入/输出：float32 [0,1]
    注意：所有中间计算使用 float32 避免精度提升。
    """
    img_float = img.astype(np.float32)
    H, W, C = img.shape
    overlap_w = int(W * overlap_ratio)
    overlap_h = int(H * overlap_ratio)
    eps = np.float32(1e-5)

    # 水平方向：左右边缘均值对齐
    mean_L = np.mean(img_float[:, :overlap_w], axis=(0, 1)).astype(np.float32)
    mean_R = np.mean(img_float[:, -overlap_w:], axis=(0, 1)).astype(np.float32)
    target_X = (mean_L + mean_R) * np.float32(0.5)  # 目标：左右平均色
    profile_X = np.linspace(0, 1, W, dtype=np.float32).reshape(1, W, 1)
    lighting_X = mean_L * (1 - profile_X) + mean_R * profile_X
    img_float = img_float * (target_X / (lighting_X + eps))

    # 垂直方向：上下边缘均值对齐
    mean_T = np.mean(img_float[:overlap_h, :], axis=(0, 1)).astype(np.float32)
    mean_B = np.mean(img_float[-overlap_h:, :], axis=(0, 1)).astype(np.float32)
    target_Y = (mean_T + mean_B) * np.float32(0.5)
    profile_Y = np.linspace(0, 1, H, dtype=np.float32).reshape(H, 1, 1)
    lighting_Y = mean_T * (1 - profile_Y) + mean_B * profile_Y
    img_float = img_float * (target_Y / (lighting_Y + eps))

    return np.clip(img_float, 0.0, 1.0)


def np_quilt_horizontal(
    img: np.ndarray,
    overlap_w: int,
    alpha: float = 1.0,
    beta: float = 5.0
) -> np.ndarray:
    """
    水平方向最优缝合线拼接（Seam Carving + 梯度惩罚 + 双频融合）。

    流程：
    1. 取左右重叠区域 → 局部色彩匹配消除色差
    2. 计算 RGB 色差 + 梯度惩罚 → 综合误差图
    3. 动态规划找最优缝合线
    4. 双频融合：低频渐变 + 高频硬切

    alpha: RGB 色差权重，beta: 梯度惩罚权重（越大越保护边缘结构）。
    """
    h, w, c = img.shape
    R = img[:, -overlap_w:, :].astype(np.float32)   # 右侧重叠区（原图右半）
    L = img[:, :overlap_w, :].astype(np.float32)     # 左侧重叠区（原图左半）

    # 将右侧色彩匹配到左侧，消除宏观色差
    R_matched = np_local_color_transfer(R, L)

    # 综合误差 = RGB像素差 + 梯度结构差异
    diff = R_matched - L
    ssd_rgb = np.sum(diff ** 2, axis=2).astype(np.float32)
    ssd_grad = np_compute_gradient_penalty(L, R_matched).astype(np.float32)
    error_matrix = np.float32(alpha) * ssd_rgb + np.float32(beta) * ssd_grad

    # 动态规划找最优缝合线
    seam = np_find_vertical_seam(error_matrix)

    # 生成硬蒙版：缝合线左侧=1（取左图），右侧=0（取右图）
    hard_mask = np.zeros((h, overlap_w, 1), dtype=np.float32)
    for i in range(h):
        hard_mask[i, :seam[i]] = 1.0

    # 双频融合
    stitched = np_two_band_blend(L, R_matched, hard_mask)
    middle = img[:, overlap_w:-overlap_w, :]  # 中间不重叠区域直接保留
    return np.concatenate([middle, stitched], axis=1)


def np_quilt_vertical(
    img: np.ndarray,
    overlap_h: int,
    alpha: float = 1.0,
    beta: float = 5.0
) -> np.ndarray:
    """垂直方向最优缝合线拼接（委托给水平函数，转置处理）"""
    transposed = np.transpose(img, (1, 0, 2))
    stitched = np_quilt_horizontal(transposed, overlap_h, alpha, beta)
    return np.transpose(stitched, (1, 0, 2))


def np_make_texture_seamless(
    img: np.ndarray,
    overlap_ratio: float = 0.2,
    beta: float = 5.0
) -> np.ndarray:
    """
    高级无缝贴图生成（完整管线）。

    流程：宏观梯度补偿 → 水平缝合 → 垂直缝合。
    输入/输出：float32 [0,1] RGB (H,W,3)。

    beta: 梯度惩罚权重（强纹理如砖块可调高到 10.0，平滑纹理如草地可降到 2.0）。

    重要提示：输出尺寸会因 overlap 裁剪而缩小！
    输入 (H,W) → 输出 (H - 2*overlap_h, W - 2*overlap_w)。
    如需保持尺寸，调用后自行用 np_resize_img 恢复。
    """
    img = img.astype(np.float32)
    h, w = img.shape[:2]
    overlap_w = int(w * overlap_ratio)
    overlap_h = int(h * overlap_ratio)

    img_corrected = np_flatten_macro_gradient(img, overlap_ratio)
    img_h = np_quilt_horizontal(img_corrected, overlap_w, alpha=1.0, beta=beta)
    final_img = np_quilt_vertical(img_h, overlap_h, alpha=1.0, beta=beta)

    return np.clip(final_img, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# Reinhard 色彩迁移（LAB 空间）
# 原理：在 CIE LAB 色彩空间中进行统计量匹配（均值/标准差），
#       将目标图像的色彩分布迁移到源图像风格。
# ══════════════════════════════════════════════════════════════════════════════

def np_rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """
    sRGB (H,W,4) → CIE LAB (H,W,3)
    包含 sRGB 线性化（伽马校正 2.4）和标准 D65 白点转换。

    色彩空间链：sRGB → 线性 RGB → XYZ(D65) → CIE LAB
    """
    if rgb.ndim != 3 or rgb.shape[-1] != 4:
        raise ValueError(f"输入必须是(H,W,4)的RGBA数组！当前形状：{rgb.shape}")
    rgb = rgb.astype(np.float32)
    rgb_only = np.clip(rgb[..., :3].copy(), 1e-6, 1.0)

    # sRGB → 线性 RGB（gamma 校正反变换）
    gamma_mask = rgb_only <= 0.04045
    rgb_linear = np.where(
        gamma_mask,
        rgb_only / 12.92,
        ((rgb_only + 0.055) / 1.055) ** 2.4
    )

    # 线性 RGB → XYZ（sRGB 标准转换矩阵，D65 白点）
    M_srgb2xyz = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041]
    ], dtype=np.float32)
    xyz = np.clip(np.dot(rgb_linear, M_srgb2xyz.T), 1e-6, None)

    # XYZ → LAB (D65 参考白点)
    xyz_ref_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    xyz_scaled = xyz / xyz_ref_white

    def f(t: np.ndarray) -> np.ndarray:
        """LAB 非线性压缩函数"""
        return np.where(t > (6 / 29) ** 3, t ** (1 / 3), (29 / 6) ** 2 * t / 3 + 4 / 29)

    f_xyz = f(xyz_scaled)
    L = 116 * f_xyz[..., 1] - 16
    a = 500 * (f_xyz[..., 0] - f_xyz[..., 1])
    b = 200 * (f_xyz[..., 1] - f_xyz[..., 2])

    return np.stack([L, a, b], axis=-1)


def np_lab_to_rgb(lab: np.ndarray) -> np.ndarray:
    """
    CIE LAB (H,W,3) → sRGB (H,W,3)，float32 [0,1]。
    包含 LAB → XYZ → 线性 RGB → sRGB（伽马校正 2.4）。
    对 LAB 值域进行裁剪防止色域越界。
    """
    if lab.ndim != 3 or lab.shape[-1] != 3:
        raise ValueError(f"输入必须是(H,W,3)的LAB数组！当前形状：{lab.shape}")
    lab = lab.astype(np.float32)
    L = np.clip(lab[..., 0], 0.0, 100.0)
    a = np.clip(lab[..., 1], -128.0, 128.0)
    b = np.clip(lab[..., 2], -128.0, 128.0)
    lab_clipped = np.stack([L, a, b], axis=-1)

    # LAB → XYZ
    xyz_ref_white = np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    f_y = (lab_clipped[..., 0] + 16) / 116
    f_x = lab_clipped[..., 1] / 500 + f_y
    f_z = f_y - lab_clipped[..., 2] / 200

    def f_inv(t: np.ndarray) -> np.ndarray:
        """LAB 非线性解压缩函数"""
        return np.where(
            t > 6 / 29,
            t ** 3,
            3 * (6 / 29) ** 2 * (t - 4 / 29)
        )

    x = xyz_ref_white[0] * f_inv(f_x)
    y = xyz_ref_white[1] * f_inv(f_y)
    z = xyz_ref_white[2] * f_inv(f_z)
    xyz = np.clip(np.stack([x, y, z], axis=-1), 1e-6, None)

    # XYZ → 线性 RGB（标准逆矩阵）
    M_xyz2srgb = np.array([
        [3.2404542, -1.5371385, -0.4985314],
        [-0.9692660, 1.8760108, 0.0415560],
        [0.0556434, -0.2040259, 1.0572252]
    ], dtype=np.float32)
    rgb_linear = np.clip(np.dot(xyz, M_xyz2srgb.T), 0.0, 1.0)

    # 线性 RGB → sRGB（伽马校正）
    gamma_mask = rgb_linear <= 0.0031308
    rgb_srgb = np.where(
        gamma_mask,
        rgb_linear * 12.92,
        1.055 * (rgb_linear ** (1 / 2.4)) - 0.055
    )
    return np.clip(rgb_srgb, 0.0, 1.0)


def np_reinhard_color_transfer(
    source_np: np.ndarray,
    target_np: np.ndarray,
    l_weight: float = 0.5,
    ab_weight: float = 1.0
) -> np.ndarray:
    """
    Reinhard 色彩迁移（LAB 统计量匹配）。
    将 target 图像的色彩特性迁移到 source 风格。

    L 通道单独柔化（保留目标高光细节），a/b 通道正常迁移（保证色彩还原）。
    l_weight=0.5 折中保留原目标亮度和迁移源亮度，避免高光变暗。
    输入/输出：float32 RGBA (H,W,4)
    """
    if source_np.shape != target_np.shape:
        raise ValueError(
            f"源/目标形状必须一致！源：{source_np.shape}，目标：{target_np.shape}"
        )
    if source_np.ndim != 3 or source_np.shape[-1] != 4:
        raise ValueError(f"输入必须是(H,W,4)的RGBA数组！")

    # 保留目标 Alpha
    alpha_channel = target_np[..., 3:4].copy()

    # 转换到 LAB 空间
    src_lab = np_rgb_to_lab(source_np)
    tar_lab = np_rgb_to_lab(target_np)

    src_L, src_a, src_b = src_lab[..., 0], src_lab[..., 1], src_lab[..., 2]
    tar_L, tar_a, tar_b = tar_lab[..., 0], tar_lab[..., 1], tar_lab[..., 2]

    # 统计量提取
    src_L_mean, src_L_std = src_L.mean(), src_L.std()
    src_a_mean, src_a_std = src_a.mean(), src_a.std()
    src_b_mean, src_b_std = src_b.mean(), src_b.std()
    tar_L_mean, tar_L_std = tar_L.mean(), tar_L.std()
    tar_a_mean, tar_a_std = tar_a.mean(), tar_a.std()
    tar_b_mean, tar_b_std = tar_b.mean(), tar_b.std()

    eps = 1e-6
    src_L_std = max(src_L_std, eps)
    src_a_std = max(src_a_std, eps)
    src_b_std = max(src_b_std, eps)
    tar_L_std = max(tar_L_std, eps)
    tar_a_std = max(tar_a_std, eps)
    tar_b_std = max(tar_b_std, eps)

    # L 通道：加权保留原目标亮度，防止高光变暗
    # 公式：tar_L_target = (tar_L - μ_tar) / σ_tar * σ_src + μ_src
    tar_L_trans = (tar_L - tar_L_mean) / tar_L_std * src_L_std + src_L_mean
    tar_L_final = (1 - l_weight) * tar_L + l_weight * tar_L_trans

    # a/b 通道：加权迁移源色彩分布
    # 公式：tar_a_target = (tar_a - μ_tar_a) / σ_tar_a * σ_src_a * w_ab + μ_src_a * w_ab + μ_tar_a * (1-w_ab)
    tar_a_final = (
        (tar_a - tar_a_mean) / tar_a_std * src_a_std * ab_weight
        + src_a_mean * ab_weight + tar_a_mean * (1 - ab_weight)
    )
    tar_b_final = (
        (tar_b - tar_b_mean) / tar_b_std * src_b_std * ab_weight
        + src_b_mean * ab_weight + tar_b_mean * (1 - ab_weight)
    )

    result_lab = np.stack([tar_L_final, tar_a_final, tar_b_final], axis=-1)
    result_rgb = np_lab_to_rgb(result_lab)
    result_rgba = np.concatenate([result_rgb, alpha_channel], axis=-1)
    return np.clip(result_rgba, 0.0, 1.0)


# 向后兼容别名（旧函数名仍然可用）
rgb_to_lab_np = np_rgb_to_lab
lab_to_rgb_np = np_lab_to_rgb
reinhard_color_transfer_np = np_reinhard_color_transfer


def np_linear_to_srgb(img: np.ndarray) -> np.ndarray:
    """线性→sRGB gamma 校正（IEC 61966-2-1 标准分段函数）。
    输入输出：float32 [0,1]"""
    c = np.clip(img[..., :3], 0, 1)
    lo = c <= 0.0031308
    c[lo] *= 12.92
    c[~lo] = 1.055 * np.power(c[~lo], 1.0 / 2.4) - 0.055
    img[..., :3] = c
    return img


def np_srgb_to_linear(img: np.ndarray) -> np.ndarray:
    """sRGB→线性 反 gamma 校正（IEC 61966-2-1 标准分段函数）。
    输入输出：float32 [0,1]"""
    c = np.clip(img[..., :3], 0, 1)
    lo = c <= 0.04045
    c[lo] /= 12.92
    c[~lo] = np.power((c[~lo] + 0.055) / 1.055, 2.4)
    img[..., :3] = c
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 几何变换（dtype 无关，直接操作数组视图/副本）
# ══════════════════════════════════════════════════════════════════════════════

def np_bilinear_interpolate(
    img: np.ndarray,
    y: np.ndarray,
    x: np.ndarray
) -> np.ndarray:
    """
    纯 NumPy 双线性插值（底层函数）。
    支持 2D 灰度 / 3D 多通道。
    y, x 是目标像素映射到源图的浮点坐标网格。
    """
    H, W = img.shape[:2]
    x0 = np.floor(x).astype(int)
    x1 = x0 + 1
    y0 = np.floor(y).astype(int)
    y1 = y0 + 1
    x0 = np.clip(x0, 0, W - 1)
    x1 = np.clip(x1, 0, W - 1)
    y0 = np.clip(y0, 0, H - 1)
    y1 = np.clip(y1, 0, H - 1)

    Ia = img[y0, x0]
    Ib = img[y1, x0]
    Ic = img[y0, x1]
    Id = img[y1, x1]

    # 双线性权重
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    if img.ndim == 3:
        wa = wa[..., np.newaxis]
        wb = wb[..., np.newaxis]
        wc = wc[..., np.newaxis]
        wd = wd[..., np.newaxis]

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def np_zoom(img: np.ndarray, scale: float) -> np.ndarray:
    """等比缩放（双线性插值）。scale > 0，1.0 返回原图副本"""
    if scale == 1.0:
        return img.copy()
    H, W = img.shape[:2]
    new_H, new_W = int(np.round(H * scale)), int(np.round(W * scale))
    y_target, x_target = np.mgrid[0:new_H, 0:new_W]
    return np_bilinear_interpolate(img, y_target / scale, x_target / scale)


def np_rotate(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """
    任意角度旋转（双线性插值）。
    输出画布自适应大小，超出部分填充黑色 (0.0)。
    angle_deg: 顺时针角度（度）
    """
    if angle_deg % 360 == 0:
        return img.copy()
    H, W = img.shape[:2]
    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    cy, cx = H / 2.0, W / 2.0

    # 计算旋转后四个角点，确定新画布尺寸
    corners = np.array([[-cx, -cy], [cx, -cy], [-cx, cy], [cx, cy]])
    rot_mat = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    new_corners = corners @ rot_mat.T
    new_W = int(np.ceil(new_corners[:, 0].max() - new_corners[:, 0].min()))
    new_H = int(np.ceil(new_corners[:, 1].max() - new_corners[:, 1].min()))
    new_cy, new_cx = new_H / 2.0, new_W / 2.0

    y_target, x_target = np.mgrid[0:new_H, 0:new_W]
    y_centered = y_target - new_cy
    x_centered = x_target - new_cx
    # 逆旋转矩阵：从目标坐标反算源坐标
    inv_rot_mat = rot_mat.T
    x_source = x_centered * inv_rot_mat[0, 0] + y_centered * inv_rot_mat[0, 1] + cx
    y_source = x_centered * inv_rot_mat[1, 0] + y_centered * inv_rot_mat[1, 1] + cy

    rotated_img = np_bilinear_interpolate(img, y_source, x_source)
    # 超出范围的像素填充黑色
    # 注意：NumPy 2.x 中布尔索引不再广播，需显式匹配形状
    mask = (x_source >= 0) & (x_source <= W - 1) & (y_source >= 0) & (y_source <= H - 1)
    if img.ndim == 3:
        mask = np.broadcast_to(mask[..., np.newaxis], rotated_img.shape)
    rotated_img[~mask] = 0.0
    return rotated_img


def np_flip_horizontal(img: np.ndarray) -> np.ndarray:
    """水平镜像翻转（沿垂直轴）"""
    return img[:, ::-1, ...]


def np_flip_vertical(img: np.ndarray) -> np.ndarray:
    """垂直镜像翻转（沿水平轴）"""
    return img[::-1, :, ...]


def np_rotate_90(img: np.ndarray, k: int = 1) -> np.ndarray:
    """
    90 度整数倍旋转（无损，无需插值）。
    k=1: 逆时针 90°, k=2: 180°, k=-1: 顺时针 90°
    """
    return np.rot90(img, k=k, axes=(0, 1))


def np_crop(img: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = box
    return img[top:bottom, left:right].copy()


def np_rotate_image(img: np.ndarray, angle_rad: float,
                    fill_mode: str = "透明") -> np.ndarray:
    """旋转图像（双线性插值），输出自动扩边以容纳全部内容。

    纯 numpy 向量化实现，无外部依赖。

    Args:
        img: (H, W, C) float32 RGBA 数组。
        angle_rad: 旋转角度（弧度），正值为逆时针。
        fill_mode: 超界填充策略 "透明" / "黑色" / "白色" / "重复"。

    Returns:
        np.ndarray: 旋转后的图像，尺寸可能改变。
    """
    if abs(angle_rad) < 1e-6:
        return img.copy()
    h, w = img.shape[:2]
    c = img.shape[2] if img.ndim == 3 else 1
    cos_a = abs(math.cos(angle_rad))
    sin_a = abs(math.sin(angle_rad))
    new_w = int(w * cos_a + h * sin_a)
    new_h = int(h * cos_a + w * sin_a)

    fill_map = {"透明": (0.0, 0.0, 0.0, 0.0), "黑色": (0.0, 0.0, 0.0, 1.0),
                "白色": (1.0, 1.0, 1.0, 1.0)}
    if fill_mode == "透明":
        fill = (0.0, 0.0, 0.0, 0.0)
        result = np.zeros((new_h, new_w, c) if c > 1 else (new_h, new_w), dtype=img.dtype)
    elif fill_mode == "黑色":
        result = np.zeros((new_h, new_w, c) if c > 1 else (new_h, new_w), dtype=img.dtype)
        if c > 1:
            result[..., :3] = 0.0
            result[..., 3] = 1.0
        else:
            result[...] = 0.0
    elif fill_mode == "白色":
        if c > 1:
            result = np.ones((new_h, new_w, c), dtype=img.dtype)
        else:
            result = np.ones((new_h, new_w), dtype=img.dtype)
    elif fill_mode == "重复":
        result = np.zeros((new_h, new_w, c) if c > 1 else (new_h, new_w), dtype=img.dtype)
    else:
        result = np.zeros((new_h, new_w, c) if c > 1 else (new_h, new_w), dtype=img.dtype)

    cx, cy = w / 2.0, h / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0
    rcos = math.cos(-angle_rad)
    rsin = math.sin(-angle_rad)
    yy, xx = np.mgrid[0:new_h, 0:new_w]
    ix = rcos * (xx - ncx) - rsin * (yy - ncy) + cx
    iy = rsin * (xx - ncx) + rcos * (yy - ncy) + cy

    if fill_mode == "重复":
        ix0 = np.mod(np.floor(ix).astype(np.int32), w)
        iy0 = np.mod(np.floor(iy).astype(np.int32), h)
        ix1 = np.mod(ix0 + 1, w)
        iy1 = np.mod(iy0 + 1, h)
        mask = np.ones((new_h, new_w), dtype=bool)
    else:
        ix0 = np.floor(ix).astype(np.int32)
        iy0 = np.floor(iy).astype(np.int32)
        ix1 = ix0 + 1
        iy1 = iy0 + 1
        mask = (ix0 >= 0) & (ix0 < w) & (iy0 >= 0) & (iy0 < h)
        ix0 = np.clip(ix0, 0, w - 1)
        iy0 = np.clip(iy0, 0, h - 1)
        ix1 = np.clip(ix1, 0, w - 1)
        iy1 = np.clip(iy1, 0, h - 1)

    fx = ix - ix0.astype(np.float64)
    fy = iy - iy0.astype(np.float64)
    if c > 1:
        for ch in range(c):
            v00 = img[iy0, ix0, ch]
            v10 = img[iy0, ix1, ch]
            v01 = img[iy1, ix0, ch]
            v11 = img[iy1, ix1, ch]
            result[..., ch] = mask * ((1 - fy) * ((1 - fx) * v00 + fx * v10)
                                       + fy * ((1 - fx) * v01 + fx * v11))
    else:
        v00 = img[iy0, ix0]
        v10 = img[iy0, ix1]
        v01 = img[iy1, ix0]
        v11 = img[iy1, ix1]
        result = mask * ((1 - fy) * ((1 - fx) * v00 + fx * v10)
                         + fy * ((1 - fx) * v01 + fx * v11))
    return result


def np_pad(
    img: np.ndarray,
    pad_width: int,
    color: tuple[float, ...] = (0.0, 0.0, 0.0)
) -> np.ndarray:
    """
    边缘填充（默认黑色）。
    color 值域应与图像 dtype 一致（float32 [0,1] 或 uint8 [0,255]）。
    示例：float32 黑色=(0.0, 0.0, 0.0)，uint8 黑色=(0, 0, 0)

    实现说明：3D 图像按通道分离→各自填充→重新合并。
    这避免了直接使用 numpy.pad 时 constant_values 维度与 pad_config 轴数不匹配的问题。
    """
    if img.ndim == 3:
        # 逐通道填充：每个通道使用自身颜色值作为填充常数
        channels = []
        for c in range(img.shape[2]):
            ch = img[..., c]
            padded_ch = np.pad(
                ch, pad_width,
                mode='constant',
                constant_values=color[c]
            )
            channels.append(padded_ch)
        return np.stack(channels, axis=-1)
    else:
        # 2D 单通道：constant_values 应为标量
        pad_config = ((pad_width, pad_width), (pad_width, pad_width))
        cv = color[0] if isinstance(color, tuple) else color
        return np.pad(img, pad_config, mode='constant', constant_values=cv)


# ══════════════════════════════════════════════════════════════════════════════
# 滤波器
# ══════════════════════════════════════════════════════════════════════════════

def np_gaussian_filter(img: np.ndarray, sigma: float, mode: str = 'edge') -> np.ndarray:
    """
    高斯模糊（分离卷积加速：先水平再垂直）。
    使用纯 NumPy 实现 1D 分离卷积（np.convolve），无第三方依赖。
    sigma 控制模糊程度，sigma ≤ 0 返回原图。
    mode: 'edge' / 'wrap' / 'constant' / 'reflect' 边界填充模式。
    """
    if sigma <= 0:
        return img.copy()
    radius = int(np.ceil(3 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()
    kw = len(kernel)
    pad_w = kw // 2

    if img.ndim == 2:
        H, W = img.shape
        padded_h = np.pad(img, ((0, 0), (pad_w, pad_w)), mode=mode)
        horz = np.empty_like(img)
        for i in range(H):
            horz[i, :] = np.convolve(padded_h[i, :], kernel, mode='valid')
        padded_v = np.pad(horz, ((pad_w, pad_w), (0, 0)), mode=mode)
        result = np.empty_like(img)
        for j in range(W):
            result[:, j] = np.convolve(padded_v[:, j], kernel, mode='valid')
    else:
        H, W, C = img.shape
        padded_h = np.pad(img, ((0, 0), (pad_w, pad_w), (0, 0)), mode=mode)
        horz = np.empty_like(img)
        for i in range(H):
            for c in range(C):
                horz[i, :, c] = np.convolve(padded_h[i, :, c], kernel, mode='valid')
        padded_v = np.pad(horz, ((pad_w, pad_w), (0, 0), (0, 0)), mode=mode)
        result = np.empty_like(img)
        for j in range(W):
            for c in range(C):
                result[:, j, c] = np.convolve(padded_v[:, j, c], kernel, mode='valid')

    return result


def np_unsharp_mask(
    img: np.ndarray,
    sigma: float = 1.0,
    amount: float = 1.5
) -> np.ndarray:
    """
    USM 锐化：原图 + amount * (原图 - 高斯模糊)。
    公式：sharpened = img + amount * (img - blurred)
    amount 控制锐化强度：1.0=轻微，2.0=中等，3.0+=强锐化。
    过大可能导致光晕（halo）效应。
    支持 2D 和 3D 输入。
    """
    blurred = np_gaussian_filter(img, sigma)
    return np.clip(img + amount * (img - blurred), 0.0, 1.0)


def np_high_pass(img: np.ndarray, sigma: float = 3.0, contrast: float = 1.0) -> np.ndarray:
    """
    高反差保留：提取图像高频细节，输出居中在 0.5 的灰度细节图。
    公式：result = (original - gaussian_blur(original)) * contrast + 0.5
    sigma 控制保留细节的尺度（越大越保留更大尺度的结构），
    contrast 控制细节加强倍率。
    仅处理 RGB 通道，Alpha 原样保留。
    """
    if sigma <= 0:
        return img.copy()
    rgb = img[..., :3].astype(np.float32)
    blurred = np_gaussian_filter(rgb, sigma)
    high_pass = (rgb - blurred) * contrast + 0.5
    result = np.zeros_like(img)
    result[..., :3] = np.clip(high_pass, 0.0, 1.0)
    result[..., 3] = img[..., 3]
    return result


def np_sobel_edge(gray_img: np.ndarray) -> np.ndarray:
    """
    Sobel 边缘检测，返回归一化到 [0,1] 的边缘强度图。
    输入：2D float32 [0,1] 灰度图
    输出：2D float32 [0,1] 边缘强度图
    """
    if gray_img.ndim != 2:
        raise ValueError(f"Sobel边缘检测仅支持2D灰度图输入！当前维度：{gray_img.ndim}")
    padded = np.pad(gray_img, pad_width=1, mode='edge')
    tl = padded[:-2, :-2]
    tc = padded[:-2, 1:-1]
    tr = padded[:-2, 2:]
    ml = padded[1:-1, :-2]
    mr = padded[1:-1, 2:]
    bl = padded[2:, :-2]
    bc = padded[2:, 1:-1]
    br = padded[2:, 2:]

    gx = (tr + 2 * mr + br) - (tl + 2 * ml + bl)
    gy = (bl + 2 * bc + br) - (tl + 2 * tc + tr)
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    mag_max = magnitude.max()
    if mag_max > 0:
        magnitude = magnitude / mag_max
    return magnitude


def np_dilate_3x3(mask: np.ndarray) -> np.ndarray:
    """
    3×3 膨胀操作（局部取最大值）。
    使用分期 pairwise maximum 替代 3D 堆叠，避免创建 (9, H, W) 临时数组。
    对于 4096² 图像可节省约 130MB 峰值内存。
    """
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=0)
    h, w = mask.shape[:2]
    result = padded[:-2, :-2].copy()
    # 依次与 8 个邻域取最大值（中位数已在 result 中）
    np.maximum(result, padded[1:-1, :-2], out=result)
    np.maximum(result, padded[1:-1, 2:], out=result)
    np.maximum(result, padded[:-2, 1:-1], out=result)
    np.maximum(result, padded[2:, 1:-1], out=result)
    np.maximum(result, padded[:-2, :-2], out=result)
    np.maximum(result, padded[:-2, 2:], out=result)
    np.maximum(result, padded[2:, :-2], out=result)
    np.maximum(result, padded[2:, 2:], out=result)
    np.maximum(result, padded[1:-1, 1:-1], out=result)
    return result


def np_erode_3x3(mask: np.ndarray) -> np.ndarray:
    """
    3×3 腐蚀操作（局部取最小值）。
    使用分期 pairwise minimum 替代 3D 堆叠，避免创建 (9, H, W) 临时数组。
    """
    padded = np.pad(mask, pad_width=1, mode='constant', constant_values=1)
    h, w = mask.shape[:2]
    result = padded[:-2, :-2].copy()
    np.minimum(result, padded[1:-1, :-2], out=result)
    np.minimum(result, padded[1:-1, 2:], out=result)
    np.minimum(result, padded[:-2, 1:-1], out=result)
    np.minimum(result, padded[2:, 1:-1], out=result)
    np.minimum(result, padded[:-2, :-2], out=result)
    np.minimum(result, padded[:-2, 2:], out=result)
    np.minimum(result, padded[2:, :-2], out=result)
    np.minimum(result, padded[2:, 2:], out=result)
    np.minimum(result, padded[1:-1, 1:-1], out=result)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 图像增强
# ══════════════════════════════════════════════════════════════════════════════

def np_histogram_equalization(gray_img: np.ndarray) -> np.ndarray:
    """
    直方图均衡化（自动对比度增强）。
    将像素值分布拉伸到全动态范围 [0,1]。
    输入/输出：float32 [0,1]
    """
    hist, _ = np.histogram(gray_img, bins=256, range=[0.0, 1.0])
    cdf = hist.cumsum()
    cdf_masked = np.ma.masked_equal(cdf, 0)
    # 归一化 CDF 到 [0,1]
    cdf_masked = (cdf_masked - cdf_masked.min()) / (cdf_masked.max() - cdf_masked.min())
    cdf_map = np.ma.filled(cdf_masked, 0.0).astype(np.float32)
    indices = np.clip((gray_img * 255).astype(np.int32), 0, 255)
    return cdf_map[indices]


def np_gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    伽马校正：output = input^(1/gamma)。
    gamma > 1 提亮暗部，gamma < 1 增强对比度。
    注意：此函数假定输入已在 sRGB 空间（非线性），
    与 sRGB→线性→LAB 管线中的伽马校正不同。
    """
    return np.power(np.clip(img, 0.0, 1.0), 1.0 / gamma)


def np_vignette(img: np.ndarray, intensity: float = 0.8) -> np.ndarray:
    """镜头暗角效果：中心亮边缘暗，intensity 控制暗角强度（0=无暗角, 1=全黑边缘）"""
    H, W = img.shape[:2]
    y, x = np.ogrid[0:H, 0:W]
    center_y, center_x = H / 2.0, W / 2.0
    max_dist = np.sqrt(center_x ** 2 + center_y ** 2)
    dist = np.sqrt((x - center_x) ** 2 + (y - center_y) ** 2)
    vmask = np.clip(1.0 - intensity * (dist / max_dist), 0.0, 1.0).astype(np.float32)
    if img.ndim == 3:
        vmask = vmask[..., np.newaxis]
    return img * vmask


def np_posterize(img: np.ndarray, levels: int = 4) -> np.ndarray:
    """色调分离（海报化）：将连续色调量化为 levels 个层级"""
    if levels < 2:
        levels = 2
    return np.floor(img * (levels - 1) + 0.5) / (levels - 1)


def np_adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
    """
    对比度调整。
    factor=1.0 不变，>1.0 增加对比度，<1.0 降低对比度。
    以 0.5（中性灰）为基准拉伸/压缩：output = (input - 0.5) * factor + 0.5
    """
    return np.clip((img - 0.5) * factor + 0.5, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 通道操作 + 色彩矩阵
# ══════════════════════════════════════════════════════════════════════════════

def np_apply_color_matrix(img: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """3×3 色彩矩阵变换，输入/输出 float32 RGB (H,W,3)"""
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("Image must be a 3-channel RGB array")
    return np.clip(np.dot(img, matrix.T), 0.0, 1.0)


def np_sepia_filter(img: np.ndarray) -> np.ndarray:
    """经典 Sepia（老照片/泛黄）滤镜"""
    sepia_matrix = np.array([
        [0.393, 0.769, 0.189],
        [0.349, 0.686, 0.168],
        [0.272, 0.534, 0.131]
    ], dtype=np.float32)
    return np_apply_color_matrix(img, sepia_matrix)


def np_rgb_to_gray(img: np.ndarray) -> np.ndarray:
    """
    RGB 转灰度（标准 BT.601 亮度权重：0.299R + 0.587G + 0.114B）。
    2D 输入直接返回副本。
    """
    if img.ndim == 2:
        return img.copy()
    weights = np.array([0.299, 0.587, 0.114], dtype=np.float32)
    return np.dot(img[..., :3], weights)


def np_split_channels(img: np.ndarray) -> tuple[np.ndarray, ...]:
    """分离多通道图像为各通道 2D 数组的元组"""
    if img.ndim != 3:
        raise ValueError("Image must have multiple channels")
    return tuple(img[..., i] for i in range(img.shape[-1]))


def np_merge_channels(*channels: np.ndarray) -> np.ndarray:
    """将多个 2D 通道数组合并为多通道图像"""
    return np.dstack(channels)


def np_invert(img: np.ndarray) -> np.ndarray:
    """图像反相。float32 输入返回 1.0 - img"""
    return 1.0 - img


# ══════════════════════════════════════════════════════════════════════════════
# 合成操作
# ══════════════════════════════════════════════════════════════════════════════

def np_alpha_composite(
    bg_rgba: np.ndarray,
    fg_rgba: np.ndarray
) -> np.ndarray:
    """
    RGBA 透明度合成（标准 Over 操作，Porter-Duff）。
    两张图尺寸必须相同，均为 RGBA 四通道。
    公式：a_out = a_fg + a_bg*(1-a_fg)
         rgb_out = (rgb_fg*a_fg + rgb_bg*a_bg*(1-a_fg)) / a_out
    """
    a_bg = bg_rgba[..., 3]
    a_fg = fg_rgba[..., 3]
    rgb_bg = bg_rgba[..., :3]
    rgb_fg = fg_rgba[..., :3]
    a_out = a_fg + a_bg * (1.0 - a_fg)
    a_out_safe = np.where(a_out == 0, 1.0, a_out)  # 防止除零
    rgb_out = (
        (rgb_fg * a_fg[..., np.newaxis] + rgb_bg * a_bg[..., np.newaxis] * (1.0 - a_fg)[..., np.newaxis])
        / a_out_safe[..., np.newaxis]
    )
    result = np.empty_like(bg_rgba)
    result[..., :3] = rgb_out
    result[..., 3] = a_out
    return result


def np_blend(
    img1: np.ndarray,
    img2: np.ndarray,
    alpha: float
) -> np.ndarray:
    """
    线性混合：result = img1 * (1-alpha) + img2 * alpha。
    alpha 范围 [0,1]，0=完全 img1，1=完全 img2。
    """
    if img1.shape != img2.shape:
        raise ValueError("Images must have the same dimensions")
    alpha = max(0.0, min(1.0, alpha))
    return img1 * (1.0 - alpha) + img2 * alpha


def np_paste_with_mask(
    background: np.ndarray,
    foreground: np.ndarray,
    mask: np.ndarray,
    position: tuple[int, int]
) -> np.ndarray:
    """
    带蒙版的图像粘贴（Alpha 混合）。
    background: 背景大图 (H_bg, W_bg, C)
    foreground: 前景小图 (H_fg, W_fg, C)
    mask: 蒙版 (H_fg, W_fg)，0=完全透明, 1=完全不透明
    position: (x, y) 前景左上角在背景中的坐标（允许负值/越界，自动裁剪）
    """
    bg_h, bg_w = background.shape[:2]
    fg_h, fg_w = foreground.shape[:2]
    x, y = int(position[0]), int(position[1])

    # 计算有效重叠区域（自动处理越界）
    x_min, x_max = max(0, x), min(bg_w, x + fg_w)
    y_min, y_max = max(0, y), min(bg_h, y + fg_h)
    if x_min >= x_max or y_min >= y_max:
        return background.copy()  # 无重叠区域

    fg_x_min = x_min - x
    fg_x_max = fg_x_min + (x_max - x_min)
    fg_y_min = y_min - y
    fg_y_max = fg_y_min + (y_max - y_min)

    result = background.copy()
    bg_roi = result[y_min:y_max, x_min:x_max]
    fg_roi = foreground[fg_y_min:fg_y_max, fg_x_min:fg_x_max]
    alpha_roi = mask[fg_y_min:fg_y_max, fg_x_min:fg_x_max]
    if background.ndim == 3:
        alpha_roi = alpha_roi[..., np.newaxis]

    result[y_min:y_max, x_min:x_max] = fg_roi * alpha_roi + bg_roi * (1.0 - alpha_roi)
    return result


def np_new_image(
    width: int,
    height: int,
    color: tuple[float, ...],
    mode: str = 'RGB',
    dtype: type = np.float32
) -> np.ndarray:
    """创建纯色图像。默认 float32 格式，支持 RGB/RGBA 模式，通过 dtype 参数可选 uint8"""
    channels = 4 if mode.upper() == 'RGBA' else 3
    if len(color) != channels:
        raise ValueError(f"Color tuple length must be {channels} for {mode} mode")
    return np.full((height, width, channels), color, dtype=dtype)


# ══════════════════════════════════════════════════════════════════════════════
# 位深转换优化：仿色（Dithering）与去条带（Debanding）
# 在量化/反量化时注入 TPDF 三角分布噪声，掩盖色阶断层。
#
# TPDF（Triangular Probability Density Function）噪声：
# 由两个独立均匀分布相减得到，分布为 [-1,1] 的三角形。
# 相比高斯噪声，TPDF 在量化去相关方面更优：
# - 消除量化误差的统计相关性（不引入 DC 偏移）
# - 噪声功率集中在高频（人眼不敏感）
# - 等效于 1-bit 增量（高斯噪声需要 1.5-bit）
# ══════════════════════════════════════════════════════════════════════════════

def np_dither_tpdf(img: np.ndarray, target_bits: int = 8) -> np.ndarray:
    """
    TPDF 仿色（Dithering）：高→低位深转换时注入三角分布噪声，
    掩盖量化产生的色阶断层（Banding）。
    img: float32 [0,1]，任意形状 (H,W) 或 (H,W,C)
    target_bits: 目标位深，默认 8
    返回: float32 [0,1]，已仿色（注入 ±1 量化步长的噪声后四舍五入量化）
    """
    levels = float((1 << target_bits) - 1)
    # 两个独立均匀分布相减 → TPDF 三角分布噪声
    noise1 = np.random.rand(*img.shape).astype(np.float32)
    noise2 = np.random.rand(*img.shape).astype(np.float32)
    tpdf_noise = noise1 - noise2  # 范围 [-1, 1]（在量化空间）
    dithered = np.round(img * levels + tpdf_noise) / levels
    return np.clip(dithered, 0.0, 1.0)


def np_deband(img: np.ndarray, source_bits: int = 8) -> np.ndarray:
    """
    去条带（Debanding）：低→高位深转换时注入微量三角分布噪声，
    打破已固化的量化边界，为后续调色提供缓冲空间。
    img: float32 [0,1]，任意形状
    source_bits: 源图像的实际位深，默认 8
    返回: float32 [0,1]，已注入噪声
    """
    step = 1.0 / float((1 << source_bits) - 1)
    noise1 = np.random.rand(*img.shape).astype(np.float32)
    noise2 = np.random.rand(*img.shape).astype(np.float32)
    tpdf_noise = (noise1 - noise2) * step * 1.5  # 1.5x 步长确保充分打破量化边界
    return np.clip(img + tpdf_noise, 0.0, 1.0)


# ══════════════════════════════════════════════════════════════════════════════
# 阈值 + 噪声
# ══════════════════════════════════════════════════════════════════════════════

def np_threshold(gray_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    二值化：大于 threshold 输出 1.0，否则 0.0。
    threshold 默认为 0.5（float32 [0,1] 空间）。
    """
    return np.where(gray_img > threshold, 1.0, 0.0).astype(np.float32)


def np_threshold_fast(gray_img: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """
    二值化（布尔运算版）。
    直接使用比较运算 + astype，略快于 np.where（节省一次分支）。
    """
    return (gray_img > threshold).astype(np.float32)


def np_add_gaussian_noise(
    img: np.ndarray,
    mean: float = 0.0,
    std: float = 0.05
) -> np.ndarray:
    """添加高斯噪声。std=0.05 在 float32 空间中对应约 12.75 级 uint8 噪声"""
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    return np.clip(img + noise, 0.0, 1.0)


def np_add_salt_pepper_noise(img: np.ndarray, amount: float = 0.02) -> np.ndarray:
    """添加椒盐噪声（随机黑白点）。amount 控制噪点密度（0~1）"""
    noisy = img.copy()
    rand_matrix = np.random.rand(*img.shape[:2])
    # 对称分配：一半噪点为黑（0），一半为白（1）
    noisy[rand_matrix < (amount / 2.0)] = 0.0
    noisy[rand_matrix > (1.0 - amount / 2.0)] = 1.0
    return noisy


def np_voronoi_crystallize_spatial(img: np.ndarray, num_cells: int) -> np.ndarray:
    """晶格化 空间哈希版：网格分区 + 3x3 邻域搜索 + 多线程并行。
    每个像素只搜索周围格子内的种子而非全局，大图友好。
    复杂度 ≈ O(H×W×K) 其中 K ≈ 3x3 格内平均种子数（通常 < 10）。
    num_cells: 晶格数量，越多越接近原图。
    返回: float32 RGBA (H,W,4) [0,1]
    """
    h, w = img.shape[:2]
    num_cells = max(1, int(num_cells))

    seed_hash = (h * 31 + w) * 17 + num_cells
    rng = np.random.RandomState(seed_hash & 0x7FFFFFFF)
    seeds_y = rng.randint(0, h, num_cells).astype(np.int32)
    seeds_x = rng.randint(0, w, num_cells).astype(np.int32)
    seed_colors = img[seeds_y, seeds_x].astype(np.float32)

    cell_size = max(24, int(np.sqrt(h * w / num_cells)))
    grid_h = (h + cell_size - 1) // cell_size
    grid_w = (w + cell_size - 1) // cell_size

    grid = [[[] for _ in range(grid_w)] for _ in range(grid_h)]
    for i in range(num_cells):
        gy = min(seeds_y[i] // cell_size, grid_h - 1)
        gx = min(seeds_x[i] // cell_size, grid_w - 1)
        grid[gy][gx].append(i)

    indices = np.empty((h, w), dtype=np.int32)

    def _process_rows(y_start, y_end):
        for gy in range(y_start, y_end):
            y0 = gy * cell_size
            y1 = min(y0 + cell_size, h)
            if y1 <= y0:
                continue
            for gx in range(grid_w):
                x0 = gx * cell_size
                x1 = min(x0 + cell_size, w)
                if x1 <= x0:
                    continue

                nearby = []
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        ny, nx = gy + dy, gx + dx
                        if 0 <= ny < grid_h and 0 <= nx < grid_w:
                            nearby.extend(grid[ny][nx])

                if not nearby:
                    sidx = np.arange(num_cells, dtype=np.int32)
                else:
                    sidx = np.array(nearby, dtype=np.int32)

                yg, xg = np.mgrid[y0:y1, x0:x1]
                cy = yg.ravel().astype(np.float32)
                cx = xg.ravel().astype(np.float32)
                sy = seeds_y[sidx].astype(np.float32)
                sx = seeds_x[sidx].astype(np.float32)
                d2 = (cy[:, np.newaxis] - sy) ** 2 + (cx[:, np.newaxis] - sx) ** 2
                local = np.argmin(d2, axis=-1).reshape(y1 - y0, x1 - x0)
                indices[y0:y1, x0:x1] = sidx[local]

    num_strips = min(4, grid_h)
    strip_size = (grid_h + num_strips - 1) // num_strips
    for t in range(num_strips):
        y0 = t * strip_size
        y1 = min(y0 + strip_size, grid_h)
        if y0 < y1:
            _process_rows(y0, y1)

    result = np.empty_like(img)
    for c in range(img.shape[-1]):
        result[:, :, c] = seed_colors[indices, c]
    return result

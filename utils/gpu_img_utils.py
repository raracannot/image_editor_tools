import bpy
import gpu
import blf
import os
import math
import numpy as np
from mathutils import Matrix
from gpu_extras.batch import batch_for_shader

# ══════════════════════════════════════════════════════════════════════════════
# gpu_img_utils —— 基于 GLSL/GPUOffScreen 的图像处理工具集
#
# 与 np_img_utils 对应：np_img_utils 走 CPU(numpy)，本模块走 GPU(着色器)。
#
# 统一约定（与 np_img_utils 完全一致，均经 Blender 实测确认）：
#   · 图像数组：RGBA、float32、范围 [0,1]、形状 (H, W, 4)
#   · 像素顺序：行优先、底到顶（Blender / OpenGL 原点在左下角），无需翻转
#   · 回读：gpu.types.Buffer 直接 np.array 后按 (H,W,4) 索引会错位，
#     np.frombuffer 也会因非连续报错。实测 np.array(buffer).flatten('F')
#     恰好等于正确的 foreach_set 顺序（与 to_list().ravel() 逐字节相同，
#     但快约 480 倍）；要 (H,W,4) 时再对该 flat 做 reshape。无需翻转。
#   · 采样：from_image 会按图像 colorspace 处理——sRGB 图会被线性化（0.5→0.214），
#     与 np_img_utils 的裸值不一致。需裸值时用 tex_from_image_raw（foreach_get 读
#     原始像素 + Buffer 直接上传 GPUTexture），它**不修改源图 colorspace**，可安全
#     在 draw handler 中调用（早期"临时改 Non-Color"的做法会在绘制回调里损坏未
#     保存的生成图）。
#   · 离屏缓冲默认 RGBA32F，保住 float 图精度。
# ══════════════════════════════════════════════════════════════════════════════
#
# ❗使用上下文（核心约定，务必遵守）
#
#   核心坑：在 POST_PIXEL 绘制回调（SpaceImageEditor.draw_handler_add 的回调）里做
#   "离屏渲染 + 采样纹理 + 回读"不可靠——前景纹理可能采样为空（合成丢内容），而同一
#   段代码在事件/操作符上下文执行却正常。与单 pass / 多趟无关，只要是离屏渲染都中招。
#
#   ▶ 离屏计算类（创建 GPUOffScreen + 回读，**只能在非绘制回调上下文调用**：
#     操作符 execute、modal 事件、mouse_release、bpy.app.timers 等）：
#       run_to_buffer / render_offscreen
#       composite_transform_to_image / _to_npimg
#       gpu_blur_to_image / _to_npimg
#       gpu_blur_npimg
#       gpu_resize_to_image / _to_npimg
#       gpu_canvas_resize_to_image / _to_npimg
#       gpu_text_to_image / _to_npimg
#       gpu_crystallize_npimg
#       gpu_gaussian_npimg / gpu_ao_npimg / gpu_height_to_normal_npimg / gpu_laplacian_npimg
#
#   ▶ 屏幕绘制类（**可在绘制回调里安全使用**，零离屏零回读）：
#       get_composite_shader + bind_composite + 自绘 quad —— 用于实时预览。
#
#   ▶ 模态预览推荐架构：绘制回调里用 bind_composite + from_image 纹理直接画到屏幕；
#     需要落地像素（apply/save）时，再在事件上下文里调离屏计算类函数回读。
#     (参见 warp/place_text.py 的 _draw_composite vs apply_to_original。)
# ══════════════════════════════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════════════════════════════
# 1. 全屏四边形 + 着色器缓存基础设施
# ══════════════════════════════════════════════════════════════════════════════

_FS_VERTS = ((-1.0, -1.0), (1.0, -1.0), (1.0, 1.0), (-1.0, 1.0))
_FS_UVS = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
_FS_INDICES = ((0, 1, 2), (2, 3, 0))

_SHADER_CACHE: dict = {}

# 通用顶点着色器：UV 透传 + MVP 变换
# （显示时用区域像素投影，离屏时载入单位矩阵 → 同一段代码两用）
FULLSCREEN_VS = """
    void main() {
        v_texCoord = texCoord;
        gl_Position = ModelViewProjectionMatrix * vec4(pos, 0.0, 1.0);
    }
"""


def get_cached_shader(key, builder):
    """按 key 缓存编译后的着色器，避免重复编译。builder 为无参构造函数。"""
    shader = _SHADER_CACHE.get(key)
    if shader is None:
        shader = builder()
        _SHADER_CACHE[key] = shader
    return shader


def clear_shader_cache():
    """卸载插件 / 热重载时清空着色器缓存。"""
    _SHADER_CACHE.clear()


def fullscreen_batch(shader):
    """构造覆盖整个 NDC 的全屏四边形 batch（texCoord 0..1）。"""
    return batch_for_shader(
        shader, 'TRIS',
        {"pos": _FS_VERTS, "texCoord": _FS_UVS},
        indices=_FS_INDICES,
    )


def _load_identity_matrices():
    gpu.matrix.load_matrix(Matrix.Identity(4))
    gpu.matrix.load_projection_matrix(Matrix.Identity(4))


def ortho_matrix(left, right, bottom, top, near=-1.0, far=1.0):
    """像素坐标正交投影矩阵（blf 文字绘制等按像素定位时用）。"""
    mat = Matrix.Identity(4)
    mat[0][0] = 2.0 / (right - left)
    mat[1][1] = 2.0 / (top - bottom)
    mat[2][2] = -2.0 / (far - near)
    mat[0][3] = -(right + left) / (right - left)
    mat[1][3] = -(top + bottom) / (top - bottom)
    mat[2][3] = -(far + near) / (far - near)
    return mat


def run_to_buffer(width, height, shader, draw_fn=None,
                  fmt='RGBA32F', clear=(0.0, 0.0, 0.0, 0.0)):
    """创建离屏 → 单位矩阵下绘制 → 回读 FLOAT Buffer → 释放离屏。

    draw_fn(shader)：在已 bind 离屏、已载单位矩阵后调用，负责 bind 着色器、
                     设 uniform 并 draw。为 None 时默认画一个全屏 quad。
    返回 gpu.types.Buffer，形状 (height, width, 4)。
    """
    offscreen = gpu.types.GPUOffScreen(width, height, format=fmt)
    try:
        with offscreen.bind():
            fb = gpu.state.active_framebuffer_get()
            if hasattr(fb, "clear"):
                fb.clear(color=clear)
            else:
                gpu.state.clear_color_and_depth(clear, 1.0)
            gpu.state.blend_set('NONE')
            gpu.matrix.push()
            gpu.matrix.push_projection()
            _load_identity_matrices()
            if draw_fn is None:
                shader.bind()
                fullscreen_batch(shader).draw(shader)
            else:
                draw_fn(shader)
            gpu.matrix.pop_projection()
            gpu.matrix.pop()
            buffer = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
    finally:
        offscreen.free()
    return buffer


# ══════════════════════════════════════════════════════════════════════════════
# 2. 纹理 / 缓冲 / 图像 互转
# ══════════════════════════════════════════════════════════════════════════════

def tex_from_image(image):
    """bpy.types.Image → GPUTexture（按图像 colorspace 采样，用于屏幕显示）。"""
    return gpu.texture.from_image(image)


def tex_from_image_raw(image):
    """以"裸值"方式采样图像：用 pixels.foreach_get 读取原始存储像素，再经
    gpu.types.Buffer 直接上传成 GPUTexture。**不修改源图 colorspace/pack**，可安全在
    draw handler 中调用（早期"临时改 Non-Color"的做法会损坏未保存的生成图）。
    foreach_get 读取的就是裸值（与 np_img_utils.blimg_2_npimg 一致）。返回 GPUTexture。"""
    w, h = image.size
    buf = np.empty(w * h * 4, dtype=np.float32)
    image.pixels.foreach_get(buf)
    gbuf = gpu.types.Buffer('FLOAT', w * h * 4, buf)
    return gpu.types.GPUTexture((w, h), format='RGBA32F', data=gbuf)


def tex_from_npimg(np_array, fmt='RGBA32F'):
    """numpy (H, W, 4) float32 → GPUTexture（裸值直传，无任何色彩管理）。
    与 tex_from_image_raw 同语义，但输入是 numpy 而非 bpy.Image。"""
    h, w = np_array.shape[:2]
    flat = np.ascontiguousarray(np_array, dtype=np.float32).ravel()
    buf = gpu.types.Buffer('FLOAT', w * h * 4, flat)
    return gpu.types.GPUTexture((w, h), format=fmt, data=buf)


def get_or_create_image(name, width, height):
    """按名取图，尺寸不符则缩放；不存在则新建（带 alpha）。"""
    if name in bpy.data.images:
        img = bpy.data.images[name]
        if img.size[0] != width or img.size[1] != height:
            img.scale(width, height)
    else:
        img = bpy.data.images.new(name, width, height, alpha=True)
    return img


def buffer_to_npimg(buffer, width, height) -> np.ndarray:
    """回读 Buffer → numpy (H, W, 4) float32，与 blimg_2_npimg 同约定。
    注意：gpu.types.Buffer 直接 np.array 后按 (H,W,4) 索引会错位；
    实测 np.array(buffer).flatten('F') 恰好等于正确的 foreach_set 顺序
    （与 to_list().ravel() 逐字节相同，但快约 480 倍），再 reshape 回 (H,W,4)。"""
    flat = np.array(buffer, copy=False, dtype=np.float32).flatten('F')
    return flat.reshape((height, width, 4))


def write_buffer_to_image(buffer, width, height, name):
    """回读 Buffer 直接写入（或新建）Blender 图像，返回该图像。"""
    img = get_or_create_image(name, width, height)
    flat = np.array(buffer, copy=False, dtype=np.float32).flatten('F')
    img.pixels.foreach_set(flat)
    img.update()
    return img


# ══════════════════════════════════════════════════════════════════════════════
# 3. GLSL 片段库（可复用着色器源码常量）
# ══════════════════════════════════════════════════════════════════════════════

# 与 utils/blend_modes.py 的 apply_blend_mode 逐条对齐（裸值，不做色彩转换）
BLEND_MODE_INDEX = {
    'MIX': 0,
    'DARKEN': 1,
    'MULTIPLY': 2,
    'BURN': 3,
    'LIGHTEN': 4,
    'SCREEN': 5,
    'DODGE': 6,
    'ADD': 7,
    'OVERLAY': 8,
    'SOFT_LIGHT': 9,
    'LINEAR_LIGHT': 10,
    'DIFFERENCE': 11,
    'EXCLUSION': 12,
    'SUBTRACT': 13,
    'DIVIDE': 14,
}

# imgtools_blend(B, L, mode)：B=背景, L=前景；返回混合后的 RGB（未含 alpha）
BLEND_MODE_GLSL = """
    vec3 imgtools_blend(vec3 B, vec3 L, int m) {
        const float eps = 1e-5;
        if (m == 1)  return min(B, L);                                  // DARKEN
        if (m == 2)  return B * L;                                      // MULTIPLY
        if (m == 3)  return clamp(1.0 - (1.0 - B) / (L + eps), 0.0, 1.0); // BURN
        if (m == 4)  return max(B, L);                                  // LIGHTEN
        if (m == 5)  return 1.0 - (1.0 - B) * (1.0 - L);                // SCREEN
        if (m == 6)  return clamp(B / (1.0 - L + eps), 0.0, 1.0);       // DODGE
        if (m == 7)  return clamp(B + L, 0.0, 1.0);                     // ADD
        if (m == 8)  return mix(1.0 - 2.0 * (1.0 - B) * (1.0 - L),
                                2.0 * B * L, step(B, vec3(0.5)));       // OVERLAY
        if (m == 9)  return mix(sqrt(max(B, 0.0)) * (2.0 * L - 1.0) + 2.0 * B * (1.0 - L),
                                2.0 * B * L + B * B * (1.0 - 2.0 * L),
                                step(L, vec3(0.5)));                    // SOFT_LIGHT
        if (m == 10) return clamp(B + 2.0 * L - 1.0, 0.0, 1.0);         // LINEAR_LIGHT
        if (m == 11) return abs(B - L);                                 // DIFFERENCE
        if (m == 12) return B + L - 2.0 * B * L;                        // EXCLUSION
        if (m == 13) return clamp(B - L, 0.0, 1.0);                     // SUBTRACT
        if (m == 14) return clamp(B / (L + eps), 0.0, 1.0);            // DIVIDE
        return L;                                                       // MIX(0)
    }
"""


# ══════════════════════════════════════════════════════════════════════════════
# 4. 高层操作：变换合成（前景按 平移/缩放/旋转 叠加到背景 + 混合 + alpha-over）
#
#    替代 place_text 中 numpy 版 _composite：反变换采样交给硬件双线性，
#    混合 + alpha-over 全在片元着色器完成。显示与 apply 复用同一着色器，
#    保证预览与落地结果一致。
# ══════════════════════════════════════════════════════════════════════════════

def _build_composite_shader():
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "ModelViewProjectionMatrix")
    info.push_constant('VEC2', "img_size")
    info.push_constant('VEC2', "fg_size")
    info.push_constant('VEC2', "center")
    info.push_constant('FLOAT', "scl")
    info.push_constant('FLOAT', "cos_a")
    info.push_constant('FLOAT', "sin_a")
    info.push_constant('FLOAT', "opacity")
    info.push_constant('INT', "blend_mode")
    info.push_constant('INT', "srgb_out")
    info.sampler(0, 'FLOAT_2D', "bg_image")
    info.sampler(1, 'FLOAT_2D', "fg_image")

    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_in(1, 'VEC2', "texCoord")
    iface = gpu.types.GPUStageInterfaceInfo("imgtools_composite_iface")
    iface.smooth('VEC2', "v_texCoord")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "fragColor")

    info.vertex_source(FULLSCREEN_VS)
    info.fragment_source(BLEND_MODE_GLSL + """
        vec3 imgtools_srgb_to_lin(vec3 c) {
            bvec3 lo = lessThanEqual(c, vec3(0.04045));
            return mix(pow((c + 0.055) / 1.055, vec3(2.4)), c / 12.92, lo);
        }
        void main() {
            vec2 px = v_texCoord * img_size;
            vec2 d = px - center;
            // 反旋转 + 反缩放，映射回前景像素坐标
            vec2 src = vec2(d.x * cos_a - d.y * sin_a,
                            d.x * sin_a + d.y * cos_a) / scl + fg_size * 0.5;
            vec2 fuv = src / fg_size;

            vec4 bg = texture(bg_image, v_texCoord);
            vec4 fg = vec4(0.0);
            if (fuv.x >= 0.0 && fuv.x <= 1.0 && fuv.y >= 0.0 && fuv.y <= 1.0) {
                fg = texture(fg_image, fuv);
            }

            vec3 blended = imgtools_blend(bg.rgb, fg.rgb, blend_mode);
            float a = fg.a * opacity;
            float oa = a + bg.a * (1.0 - a);
            vec3 orgb = (blended * a + bg.rgb * bg.a * (1.0 - a)) / max(oa, 1e-5);
            // srgb_out=1: 屏幕预览用——把裸值结果转线性，使帧缓冲的 线性→sRGB 编码后正好显示裸值
            if (srgb_out == 1) orgb = imgtools_srgb_to_lin(clamp(orgb, 0.0, 1.0));
            fragColor = vec4(orgb, oa);
        }
    """)
    return gpu.shader.create_from_info(info)


def get_composite_shader():
    """获取（缓存的）变换合成着色器。"""
    return get_cached_shader('composite_transform', _build_composite_shader)


def bind_composite(shader, bg_tex, fg_tex, img_w, img_h, fg_w, fg_h,
                   center_x, center_y, scale, rotation,
                   blend_mode='MIX', opacity=1.0, srgb_out=0):
    """bind 合成着色器并设置全部 uniform。

    调用方随后自行 draw 几何体：
      · 显示：在区域像素坐标下画覆盖图像矩形的 quad（texCoord 0..1）
      · 落地：用 fullscreen_batch 画进图像分辨率的离屏

    center_x/center_y：前景中心在背景图像中的像素坐标
    scale：相对缩放系数（与 place_text 的 self.scale 同义）
    rotation：弧度
    srgb_out：1=屏幕预览（输出端 srgb_to_linear，使帧缓冲编码后显示裸值；
              要求 bg_tex/fg_tex 为裸值纹理 tex_from_image_raw，混合空间才与落地一致）；
              0=落地（裸值输出，回读后存储）。
    """
    fw = max(int(fg_w), 1)
    fh = max(int(fg_h), 1)
    rel_sx = scale * (img_w / fw)
    rel_sy = scale * (img_h / fh)
    scl = max(min(rel_sx, rel_sy), 1e-6)
    cos_a = math.cos(-rotation)
    sin_a = math.sin(-rotation)
    bmi = BLEND_MODE_INDEX.get(blend_mode, 0)

    shader.bind()
    shader.uniform_sampler("bg_image", bg_tex)
    shader.uniform_sampler("fg_image", fg_tex)
    shader.uniform_float("img_size", (float(img_w), float(img_h)))
    shader.uniform_float("fg_size", (float(fw), float(fh)))
    shader.uniform_float("center", (float(center_x), float(center_y)))
    shader.uniform_float("scl", float(scl))
    shader.uniform_float("cos_a", float(cos_a))
    shader.uniform_float("sin_a", float(sin_a))
    shader.uniform_float("opacity", float(opacity))
    shader.uniform_int("blend_mode", bmi)
    shader.uniform_int("srgb_out", int(srgb_out))


def _composite_run(bg_image, fg_image, center_x, center_y, scale, rotation,
                   blend_mode, opacity, fmt):
    """内部：以裸值纹理采样 bg/fg，渲进背景分辨率离屏，返回 (buffer, w, h)。"""
    img_w, img_h = bg_image.size
    fg_w, fg_h = fg_image.size
    shader = get_composite_shader()
    bg_tex = tex_from_image_raw(bg_image)
    fg_tex = tex_from_image_raw(fg_image)

    def _draw(sh):
        bind_composite(sh, bg_tex, fg_tex, img_w, img_h, fg_w, fg_h,
                       center_x, center_y, scale, rotation, blend_mode, opacity)
        fullscreen_batch(sh).draw(sh)
    buffer = run_to_buffer(img_w, img_h, shader, draw_fn=_draw, fmt=fmt)
    return buffer, img_w, img_h


def composite_transform_to_image(bg_image, fg_image, out_name,
                                 center_x, center_y, scale, rotation,
                                 blend_mode='MIX', opacity=1.0, fmt='RGBA32F'):
    """变换合成"落地"：前景按 平移/缩放/旋转 + 混合 + alpha-over 叠到背景，
    渲进背景分辨率离屏 → 回读 → 写入 out_name 图像，返回该图像。

    bg_image / fg_image：bpy.types.Image（内部以 Non-Color 裸值采样并自动还原）
    center_x/center_y：前景中心在背景图像中的像素坐标
    scale：相对缩放系数（与 place_text 的 self.scale 同义）
    rotation：弧度
    """
    buffer, w, h = _composite_run(bg_image, fg_image, center_x, center_y,
                                  scale, rotation, blend_mode, opacity, fmt)
    return write_buffer_to_image(buffer, w, h, out_name)


def composite_transform_to_npimg(bg_image, fg_image,
                                 center_x, center_y, scale, rotation,
                                 blend_mode='MIX', opacity=1.0, fmt='RGBA32F') -> np.ndarray:
    """同 composite_transform_to_image，但直接返回 numpy (H, W, 4)，不建数据块。"""
    buffer, w, h = _composite_run(bg_image, fg_image, center_x, center_y,
                                  scale, rotation, blend_mode, opacity, fmt)
    return buffer_to_npimg(buffer, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# 5. 通用绘制 / 着色器构造辅助
#
#   说明：以下提取自参考实现（Dynamic Sum Node 的 blur/scaler），并按本库约定
#   适配：① 去掉 fragment 里的 linear_to_srgb（我们用 Non-Color 裸值采样，
#   不能再做一次色彩转换）；② 离屏统一 RGBA32F；③ 共享 get_cached_shader；
#   ④ 回读走 write_buffer_to_image / buffer_to_npimg 的 flatten('F') 快路。
#   ⑤ 这些都是"离屏渲染"函数，禁止在 POST_PIXEL 绘制回调里调用（见模块顶部"使用上下文"）。
# ══════════════════════════════════════════════════════════════════════════════

def _bind_draw(uniforms_float=None, uniforms_int=None, samplers=None):
    """生成 draw_fn(shader)：bind + 设 uniform + 画全屏 quad。
    uniforms_float 的值可为标量或元组（uniform_float 通用）。"""
    def _fn(shader):
        shader.bind()
        if uniforms_float:
            for k, v in uniforms_float.items():
                shader.uniform_float(k, v)
        if uniforms_int:
            for k, v in uniforms_int.items():
                shader.uniform_int(k, v)
        if samplers:
            for k, v in samplers.items():
                shader.uniform_sampler(k, v)
        fullscreen_batch(shader).draw(shader)
    return _fn


def _draw_into(offscreen, shader, draw_fn, clear=(0.0, 0.0, 0.0, 0.0)):
    """在已存在的离屏里渲染（单位矩阵 + 清屏 + draw_fn）。供 ping-pong 复用离屏。"""
    with offscreen.bind():
        fb = gpu.state.active_framebuffer_get()
        if hasattr(fb, "clear"):
            fb.clear(color=clear)
        else:
            gpu.state.clear_color_and_depth(clear, 1.0)
        gpu.state.blend_set('NONE')
        gpu.matrix.push()
        gpu.matrix.push_projection()
        _load_identity_matrices()
        draw_fn(shader)
        gpu.matrix.pop_projection()
        gpu.matrix.pop()


def render_offscreen(width, height, shader, draw_fn, fmt='RGBA32F',
                     clear=(0.0, 0.0, 0.0, 0.0)):
    """渲进一个新建离屏并返回该离屏（调用方负责 free）；中间结果取 .texture_color。
    用于多趟 pass 链式渲染。

    ⚠️ 离屏函数，禁止在 POST_PIXEL 绘制回调里调用（见模块顶部"使用上下文"）。"""
    off = gpu.types.GPUOffScreen(width, height, format=fmt)
    _draw_into(off, shader, draw_fn, clear)
    return off


def _make_shader(tag, fragment_src, floats=(), ints=(), vecs=(), samplers=("image",)):
    """按统一顶点着色器 + 全屏 quad 约定构造一个着色器。tag 保证 iface 名唯一。"""
    info = gpu.types.GPUShaderCreateInfo()
    info.push_constant('MAT4', "ModelViewProjectionMatrix")
    for n in vecs:
        info.push_constant('VEC2', n)
    for n in floats:
        info.push_constant('FLOAT', n)
    for n in ints:
        info.push_constant('INT', n)
    for i, n in enumerate(samplers):
        info.sampler(i, 'FLOAT_2D', n)
    info.vertex_in(0, 'VEC2', "pos")
    info.vertex_in(1, 'VEC2', "texCoord")
    iface = gpu.types.GPUStageInterfaceInfo("iface_" + tag)
    iface.smooth('VEC2', "v_texCoord")
    info.vertex_out(iface)
    info.fragment_out(0, 'VEC4', "fragColor")
    info.vertex_source(FULLSCREEN_VS)
    info.fragment_source(fragment_src)
    return gpu.shader.create_from_info(info)


# 边缘采样辅助：mode 0=edge(半像素 clamp), 1=wrap(循环平铺)
SAMPLE_MODE_INDEX = {'edge': 0, 'wrap': 1}
SAMPLE_GLSL = """
    vec4 imgtools_sample(sampler2D tex, vec2 uv, vec2 res, int mode) {
        if (mode == 1) { return texture(tex, fract(uv)); }
        vec2 hp = vec2(0.5) / res;
        return texture(tex, clamp(uv, hp, vec2(1.0) - hp));
    }
"""


def _get_passthrough_shader():
    return get_cached_shader('passthrough', lambda: _make_shader(
        'passthrough', "void main(){ fragColor = texture(image, v_texCoord); }"))


# ══════════════════════════════════════════════════════════════════════════════
# 6. 缩放 / 画布
# ══════════════════════════════════════════════════════════════════════════════

def _get_box_downsample_shader():
    frag = SAMPLE_GLSL + """
        void main() {
            vec4 c = vec4(0.0);
            for (int x = 0; x < 4; x++)
                for (int y = 0; y < 4; y++) {
                    vec2 o = (vec2(float(x), float(y)) - 1.5) / src_resolution;
                    c += imgtools_sample(image, v_texCoord + o, src_resolution, blur_mode);
                }
            fragColor = c / 16.0;
        }
    """
    return get_cached_shader('box_downsample', lambda: _make_shader(
        'box_downsample', frag, ints=('blur_mode',), vecs=('src_resolution',)))


def _resize_buffer(image, target_w, target_h, fmt):
    src_w, src_h = image.size
    tex = tex_from_image_raw(image)
    inter = []
    try:
        if target_w < src_w or target_h < src_h:
            # 降采样：2x 箱式金字塔逼近，最后双线性到精确目标
            cur_tex, cw, ch = tex, src_w, src_h
            ds = _get_box_downsample_shader()
            while cw // 2 >= target_w and ch // 2 >= target_h and max(cw, ch) > 16:
                nw, nh = cw // 2, ch // 2
                off = render_offscreen(nw, nh, ds, _bind_draw(
                    uniforms_float={"src_resolution": (float(cw), float(ch))},
                    samplers={"image": cur_tex}), fmt=fmt)
                inter.append(off)
                cur_tex, cw, ch = off.texture_color, nw, nh
            buffer = run_to_buffer(target_w, target_h, _get_passthrough_shader(),
                                   draw_fn=_bind_draw(samplers={"image": cur_tex}), fmt=fmt)
        else:
            # 升采样 / 等分辨率：单趟双线性
            buffer = run_to_buffer(target_w, target_h, _get_passthrough_shader(),
                                   draw_fn=_bind_draw(samplers={"image": tex}), fmt=fmt)
    finally:
        for o in inter:
            o.free()
    return buffer, target_w, target_h


def gpu_resize_to_image(image, target_w, target_h, out_name, fmt='RGBA32F'):
    """GPU 缩放：缩小走箱式金字塔，放大走双线性。写入 out_name 图像并返回。"""
    buffer, w, h = _resize_buffer(image, int(target_w), int(target_h), fmt)
    return write_buffer_to_image(buffer, w, h, out_name)


def gpu_resize_to_npimg(image, target_w, target_h, fmt='RGBA32F') -> np.ndarray:
    """同 gpu_resize_to_image，返回 numpy (H, W, 4)。"""
    buffer, w, h = _resize_buffer(image, int(target_w), int(target_h), fmt)
    return buffer_to_npimg(buffer, w, h)


CANVAS_ANCHORS = ('TOP_LEFT', 'TOP_CENTER', 'TOP_RIGHT',
                  'MID_LEFT', 'CENTER', 'MID_RIGHT',
                  'BOT_LEFT', 'BOT_CENTER', 'BOT_RIGHT')


def _anchor_offset(anchor, sw, sh, tw, th):
    """原图左下角相对新画布左下角的像素偏移。"""
    if anchor in ('TOP_LEFT', 'MID_LEFT', 'BOT_LEFT'):
        ox = 0.0
    elif anchor in ('TOP_RIGHT', 'MID_RIGHT', 'BOT_RIGHT'):
        ox = float(tw - sw)
    else:
        ox = (tw - sw) / 2.0
    if anchor in ('BOT_LEFT', 'BOT_CENTER', 'BOT_RIGHT'):
        oy = 0.0
    elif anchor in ('TOP_LEFT', 'TOP_CENTER', 'TOP_RIGHT'):
        oy = float(th - sh)
    else:
        oy = (th - sh) / 2.0
    return ox, oy


def _get_canvas_shader():
    frag = """
        void main() {
            vec2 ps = v_texCoord * tgt_res - offset;
            vec2 uv = ps / src_res;
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
                fragColor = vec4(0.0);
            } else {
                fragColor = texture(image, uv);
            }
        }
    """
    return get_cached_shader('canvas_resize', lambda: _make_shader(
        'canvas_resize', frag, vecs=('src_res', 'tgt_res', 'offset')))


def _canvas_buffer(image, target_w, target_h, anchor, fmt):
    src_w, src_h = image.size
    ox, oy = _anchor_offset(anchor, src_w, src_h, target_w, target_h)
    tex = tex_from_image_raw(image)
    buffer = run_to_buffer(target_w, target_h, _get_canvas_shader(), draw_fn=_bind_draw(
        uniforms_float={"src_res": (float(src_w), float(src_h)),
                        "tgt_res": (float(target_w), float(target_h)),
                        "offset": (float(ox), float(oy))},
        samplers={"image": tex}), fmt=fmt)
    return buffer, target_w, target_h


def gpu_canvas_resize_to_image(image, target_w, target_h, out_name,
                               anchor='CENTER', fmt='RGBA32F'):
    """GPU 画布调整：保持像素大小，按 anchor 定位扩展/裁剪画布（外部透明）。"""
    buffer, w, h = _canvas_buffer(image, int(target_w), int(target_h), anchor, fmt)
    return write_buffer_to_image(buffer, w, h, out_name)


def gpu_canvas_resize_to_npimg(image, target_w, target_h,
                               anchor='CENTER', fmt='RGBA32F') -> np.ndarray:
    buffer, w, h = _canvas_buffer(image, int(target_w), int(target_h), anchor, fmt)
    return buffer_to_npimg(buffer, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# 7. 模糊
#
#   blur_type:
#     'box' / 'gaussian'      —— 朴素 O(r²)，支持 edge/wrap 边界
#     'radial' / 'motion'     —— 径向 / 动态（motion 用 angle 弧度）
#     'separable'             —— 可分离卷积 O(r)，水平+垂直两趟（box 核）
#     'gaussian_separable'    —— 可分离高斯 O(r)
#     'downsample'            —— 1/factor 箱式降采样 → 可分离高斯 → 双线性还原
#     'kawase'                —— 多通道对角采样，每趟 4 采样
# ══════════════════════════════════════════════════════════════════════════════

BLUR_TYPE_INDEX = {'box': 0, 'gaussian': 1, 'radial': 2, 'motion': 3}


def _get_blur_naive_shader():
    frag = SAMPLE_GLSL + """
        void main() {
            if (radius <= 0.0) {
                fragColor = imgtools_sample(image, v_texCoord, resolution, blur_mode);
                return;
            }
            vec4 c = vec4(0.0);
            if (blur_type == 0) {
                float n = 0.0;
                for (float x = -radius; x <= radius; x += 1.0)
                    for (float y = -radius; y <= radius; y += 1.0) {
                        c += imgtools_sample(image, v_texCoord + vec2(x, y) / resolution, resolution, blur_mode);
                        n += 1.0;
                    }
                c /= n;
            } else if (blur_type == 1) {
                float tw = 0.0; float sigma = max(radius / 2.0, 1.0); float s2 = 2.0 * sigma * sigma;
                for (float x = -radius; x <= radius; x += 1.0)
                    for (float y = -radius; y <= radius; y += 1.0) {
                        float w = exp(-(x * x + y * y) / s2);
                        c += imgtools_sample(image, v_texCoord + vec2(x, y) / resolution, resolution, blur_mode) * w;
                        tw += w;
                    }
                c /= tw;
            } else if (blur_type == 2) {
                float n = 0.0; vec2 ctr = vec2(0.5); vec2 dir = ctr - v_texCoord;
                float steps = max(radius * 2.0, 10.0); float inten = radius * 0.005;
                for (float i = 0.0; i < steps; i += 1.0) {
                    float t = i / steps;
                    c += imgtools_sample(image, v_texCoord + dir * t * inten, resolution, blur_mode);
                    n += 1.0;
                }
                c /= n;
            } else {
                float n = 0.0; vec2 dir = vec2(cos(angle), sin(angle));
                for (float i = -radius; i <= radius; i += 1.0) {
                    c += imgtools_sample(image, v_texCoord + (dir * i) / resolution, resolution, blur_mode);
                    n += 1.0;
                }
                c /= n;
            }
            fragColor = c;
        }
    """
    return get_cached_shader('blur_naive', lambda: _make_shader(
        'blur_naive', frag, floats=('radius', 'angle'),
        ints=('blur_mode', 'blur_type'), vecs=('resolution',)))


def _get_separable_shader():
    frag = SAMPLE_GLSL + """
        void main() {
            if (radius <= 0.0) { fragColor = imgtools_sample(image, v_texCoord, resolution, blur_mode); return; }
            vec4 c = vec4(0.0);
            if (blur_type == 0) {
                float n = 0.0;
                for (float t = -radius; t <= radius; t += 1.0) {
                    c += imgtools_sample(image, v_texCoord + dir * t / resolution, resolution, blur_mode);
                    n += 1.0;
                }
                c /= n;
            } else {
                float tw = 0.0; float sigma = max(radius / 2.0, 1.0); float s2 = 2.0 * sigma * sigma;
                for (float t = -radius; t <= radius; t += 1.0) {
                    float w = exp(-(t * t) / s2);
                    c += imgtools_sample(image, v_texCoord + dir * t / resolution, resolution, blur_mode) * w;
                    tw += w;
                }
                c /= tw;
            }
            fragColor = c;
        }
    """
    return get_cached_shader('blur_separable', lambda: _make_shader(
        'blur_separable', frag, floats=('radius',), ints=('blur_type', 'blur_mode'),
        vecs=('resolution', 'dir')))


def _get_kawase_shader():
    frag = SAMPLE_GLSL + """
        void main() {
            vec4 c = vec4(0.0); vec2 d = vec2(offset) / resolution;
            c += imgtools_sample(image, v_texCoord + d, resolution, blur_mode);
            c += imgtools_sample(image, v_texCoord - d, resolution, blur_mode);
            c += imgtools_sample(image, v_texCoord + vec2(d.x, -d.y), resolution, blur_mode);
            c += imgtools_sample(image, v_texCoord + vec2(-d.x, d.y), resolution, blur_mode);
            fragColor = c / 4.0;
        }
    """
    return get_cached_shader('blur_kawase', lambda: _make_shader(
        'blur_kawase', frag, floats=('offset',), ints=('blur_mode',), vecs=('resolution',)))


def _separable_pair(tex, w, h, radius, type_int, mode_int, fmt):
    """对给定纹理做 水平+垂直 两趟可分离模糊，返回最终 buffer。"""
    sep = _get_separable_shader()
    off_h = render_offscreen(w, h, sep, _bind_draw(
        uniforms_float={"radius": float(radius), "resolution": (float(w), float(h)), "dir": (1.0, 0.0)},
        uniforms_int={"blur_type": type_int, "blur_mode": mode_int}, samplers={"image": tex}), fmt=fmt)
    try:
        buffer = run_to_buffer(w, h, sep, draw_fn=_bind_draw(
            uniforms_float={"radius": float(radius), "resolution": (float(w), float(h)), "dir": (0.0, 1.0)},
            uniforms_int={"blur_type": type_int, "blur_mode": mode_int}, samplers={"image": off_h.texture_color}), fmt=fmt)
    finally:
        off_h.free()
    return buffer


def _blur_run(tex, w, h, radius, blur_type, blur_mode, angle, factor, fmt):
    """对给定纹理做模糊，返回 buffer。tex 由调用方提供（来自图像或 numpy 上传）。"""
    inter = []
    mode_int = SAMPLE_MODE_INDEX.get(blur_mode, 0)
    try:
        if blur_type in ('box', 'gaussian', 'radial', 'motion'):
            buffer = run_to_buffer(w, h, _get_blur_naive_shader(), draw_fn=_bind_draw(
                uniforms_float={"radius": float(radius), "resolution": (float(w), float(h)), "angle": float(angle)},
                uniforms_int={"blur_mode": SAMPLE_MODE_INDEX.get(blur_mode, 0),
                              "blur_type": BLUR_TYPE_INDEX.get(blur_type, 0)},
                samplers={"image": tex}), fmt=fmt)
        elif blur_type in ('separable', 'gaussian_separable'):
            ti = 1 if blur_type == 'gaussian_separable' else 0
            buffer = _separable_pair(tex, w, h, radius, ti, mode_int, fmt)
        elif blur_type == 'downsample':
            f = max(2, int(factor))
            sw, sh = max(1, w // f), max(1, h // f)
            sr = max(1, int(radius / f))
            ds = render_offscreen(sw, sh, _get_box_downsample_shader(), _bind_draw(
                uniforms_float={"src_resolution": (float(w), float(h))},
                uniforms_int={"blur_mode": mode_int},
                samplers={"image": tex}), fmt=fmt)
            inter.append(ds)
            sep = _get_separable_shader()
            off_h = render_offscreen(sw, sh, sep, _bind_draw(
                uniforms_float={"radius": float(sr), "resolution": (float(sw), float(sh)), "dir": (1.0, 0.0)},
                uniforms_int={"blur_type": 1, "blur_mode": mode_int}, samplers={"image": ds.texture_color}), fmt=fmt)
            inter.append(off_h)
            off_v = render_offscreen(sw, sh, sep, _bind_draw(
                uniforms_float={"radius": float(sr), "resolution": (float(sw), float(sh)), "dir": (0.0, 1.0)},
                uniforms_int={"blur_type": 1, "blur_mode": mode_int}, samplers={"image": off_h.texture_color}), fmt=fmt)
            inter.append(off_v)
            buffer = run_to_buffer(w, h, _get_passthrough_shader(),
                                   draw_fn=_bind_draw(samplers={"image": off_v.texture_color}), fmt=fmt)
        elif blur_type == 'kawase':
            offsets = []
            rem, k = float(radius), 1.0
            while rem > 0.5:
                step = min(k, rem)
                offsets.append(int(step))
                rem -= step
                k *= 2.0
            if not offsets:
                offsets = [1]
            kw = _get_kawase_shader()
            cur_tex = tex
            for amt in offsets:
                o = render_offscreen(w, h, kw, _bind_draw(
                    uniforms_float={"offset": float(amt), "resolution": (float(w), float(h))},
                    uniforms_int={"blur_mode": mode_int},
                    samplers={"image": cur_tex}), fmt=fmt)
                inter.append(o)
                cur_tex = o.texture_color
            buffer = run_to_buffer(w, h, _get_passthrough_shader(),
                                   draw_fn=_bind_draw(samplers={"image": cur_tex}), fmt=fmt)
        else:
            buffer = run_to_buffer(w, h, _get_blur_naive_shader(), draw_fn=_bind_draw(
                uniforms_float={"radius": float(radius), "resolution": (float(w), float(h)), "angle": 0.0},
                uniforms_int={"blur_mode": SAMPLE_MODE_INDEX.get(blur_mode, 0), "blur_type": 0},
                samplers={"image": tex}), fmt=fmt)
    finally:
        for o in inter:
            o.free()
    return buffer


def _blur_buffer(image, radius, blur_type, blur_mode, angle, factor, fmt):
    w, h = image.size
    tex = tex_from_image_raw(image)
    buffer = _blur_run(tex, w, h, radius, blur_type, blur_mode, angle, factor, fmt)
    return buffer, w, h


def gpu_blur_to_image(image, radius, out_name, blur_type='box', blur_mode='edge',
                      angle=0.0, factor=4, fmt='RGBA32F'):
    """GPU 模糊，写入 out_name 图像并返回。blur_type 见本节顶部说明。"""
    buffer, w, h = _blur_buffer(image, radius, blur_type, blur_mode, angle, factor, fmt)
    return write_buffer_to_image(buffer, w, h, out_name)


def gpu_blur_to_npimg(image, radius, blur_type='box', blur_mode='edge',
                      angle=0.0, factor=4, fmt='RGBA32F') -> np.ndarray:
    """同 gpu_blur_to_image，返回 numpy (H, W, 4)。"""
    buffer, w, h = _blur_buffer(image, radius, blur_type, blur_mode, angle, factor, fmt)
    return buffer_to_npimg(buffer, w, h)


def gpu_blur_npimg(np_array, radius, blur_type='box', blur_mode='edge',
                   angle=0.0, factor=4, fmt='RGBA32F') -> np.ndarray:
    """numpy (H,W,4) float32 进/出的 GPU 模糊（供 numpy 管线的工具直接调用）。
    blur_type 见本节顶部说明；'box'/'gaussian' 建议用 'separable'/'gaussian_separable'
    （大半径 O(r) 远快于朴素 O(r²)）。⚠️ 离屏多趟，只在非绘制回调上下文调用。"""
    h, w = np_array.shape[:2]
    tex = tex_from_npimg(np_array)
    buffer = _blur_run(tex, w, h, radius, blur_type, blur_mode, angle, factor, fmt)
    return buffer_to_npimg(buffer, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# 8. 文字贴图生成（blf + GPUOffScreen）
#
#   用 blf 直接把字绘进离屏，自动按 blf.dimensions 计算画布尺寸，支持字间距、
#   斜体错切、颜色。提取自 place_text._generate_text_image / 参考 text_texture_maker，
#   适配本库约定：离屏 RGBA32F、回读走 flatten('F') 快路。
#
#   注意：
#     · letter_spacing 为"像素值"（直接加到字间距，未按 font_size 缩放），
#       与 place_text 语义一致；调用方如需按字号缩放请自行乘 font_size。
#     · padding 为相对字号的比例（实际留白 = int(padding * font_size)）。
#     · italic_angle 为角度（度）。
#     · 文字本身是新绘制的裸值，不做 sRGB 转换；输出图按裸值存储，后续用
#       tex_from_image_raw 采样即可与 numpy 管线一致。
# ══════════════════════════════════════════════════════════════════════════════

def _text_buffer(text, font_path, font_size, color, letter_spacing,
                 italic_angle, padding, fmt, solid_rgb=True):
    """渲染文字到离屏并回读。返回 (npimg(H,W,4) float32, w, h)；text 为空时返回 (None, 16, 16)。

    solid_rgb=True（默认）：把 RGB 三通道整体写成纯色 color.rgb，文字形状只由
    alpha（= color.a · 覆盖率）表达。这样消除"黑底渗边"——blf 直接画出来的是
    预乘式结果（边缘 rgb≈color·覆盖率，越靠边越黑），被当直 alpha 合成时会在
    抗锯齿边缘露出黑边；纯色块 + alpha 蒙版可彻底根治。"""
    if not text:
        return None, 16, 16

    font_id = 0
    loaded_path = None
    if font_path and os.path.isfile(font_path):
        try:
            font_id = blf.load(font_path)
            loaded_path = font_path
        except Exception as e:
            print(f"[gpu_img_utils] 字体加载失败: {e}")
            font_id = 0

    try:
        blf.size(font_id, font_size)
        dim_x, dim_y = blf.dimensions(font_id, text)

        if letter_spacing != 0:
            text_width = sum(blf.dimensions(font_id, ch)[0] for ch in text) \
                + max(0, len(text) - 1) * letter_spacing
        else:
            text_width = dim_x

        shear_x = math.tan(math.radians(italic_angle))
        pad = max(1, int(padding * font_size))
        width = max(1, int(text_width + abs(shear_x) * dim_y + pad * 2))
        height = max(1, int(dim_y + pad * 2))

        off = gpu.types.GPUOffScreen(width, height, format=fmt)
        verts = ((0, 0), (width, 0), (width, height), (0, height))
        indices = ((0, 1, 2), (0, 2, 3))
        clear_shader = gpu.shader.from_builtin('UNIFORM_COLOR')
        clear_batch = batch_for_shader(clear_shader, 'TRIS', {"pos": verts}, indices=indices)

        with off.bind():
            gpu.state.viewport_set(0, 0, width, height)
            gpu.state.depth_test_set('NONE')
            with gpu.matrix.push_pop():
                gpu.matrix.load_projection_matrix(ortho_matrix(0, width, 0, height))
                gpu.matrix.load_matrix(Matrix.Identity(4))

                gpu.state.blend_set('NONE')
                clear_shader.bind()
                clear_shader.uniform_float("color", (0.0, 0.0, 0.0, 0.0))
                clear_batch.draw(clear_shader)

                if italic_angle != 0.0:
                    gpu.matrix.multiply_matrix(Matrix([
                        [1.0, shear_x, 0.0, 0.0],
                        [0.0, 1.0,     0.0, 0.0],
                        [0.0, 0.0,     1.0, 0.0],
                        [0.0, 0.0,     0.0, 1.0],
                    ]))

                gpu.state.blend_set('ALPHA')
                blf.color(font_id, color[0], color[1], color[2], color[3])

                base_x = float(pad)
                if shear_x < 0:
                    base_x += abs(shear_x) * dim_y
                start_x = base_x - shear_x * pad

                if letter_spacing == 0:
                    blf.position(font_id, start_x, float(pad), 0)
                    blf.draw(font_id, text)
                else:
                    cur_x = start_x
                    for ch in text:
                        blf.position(font_id, cur_x, float(pad), 0)
                        blf.draw(font_id, ch)
                        cur_x += blf.dimensions(font_id, ch)[0] + letter_spacing

            fb = gpu.state.active_framebuffer_get()
            buffer = fb.read_color(0, 0, width, height, 4, 0, 'FLOAT')
        off.free()
        gpu.state.blend_set('NONE')
    finally:
        if font_id != 0 and loaded_path is not None:
            try:
                blf.unload(loaded_path)
            except Exception:
                pass

    # buffer(底到顶, RGBA) → (H,W,4)。flatten('F') 即正确的 foreach_set 顺序
    npimg = np.array(buffer, copy=False, dtype=np.float32).flatten('F').reshape((height, width, 4))
    if solid_rgb:
        # RGB 填纯色，仅保留 alpha 作蒙版 → 消除边缘黑底渗出
        npimg[..., 0] = color[0]
        npimg[..., 1] = color[1]
        npimg[..., 2] = color[2]
    return npimg, width, height


def gpu_text_to_image(text, out_name, font_path='', font_size=64,
                      color=(1.0, 1.0, 1.0, 1.0), letter_spacing=0.0,
                      italic_angle=0.0, padding=0.2, fmt='RGBA32F', solid_rgb=True):
    """生成自动尺寸的文字贴图，写入 out_name 图像并返回。
    solid_rgb=True（默认）：RGB 纯色 + alpha 蒙版，合成无黑边（见 _text_buffer）。"""
    npimg, w, h = _text_buffer(text, font_path, font_size, color,
                               letter_spacing, italic_angle, padding, fmt, solid_rgb)
    img = get_or_create_image(out_name, w, h)
    if npimg is None:
        img.pixels.foreach_set(np.zeros(w * h * 4, dtype=np.float32))
    else:
        img.pixels.foreach_set(npimg.ravel())
    img.update()
    return img


def gpu_text_to_npimg(text, font_path='', font_size=64,
                      color=(1.0, 1.0, 1.0, 1.0), letter_spacing=0.0,
                      italic_angle=0.0, padding=0.2, fmt='RGBA32F', solid_rgb=True) -> np.ndarray:
    """同 gpu_text_to_image，返回 numpy (H, W, 4)。"""
    npimg, w, h = _text_buffer(text, font_path, font_size, color,
                               letter_spacing, italic_angle, padding, fmt, solid_rgb)
    if npimg is None:
        return np.zeros((h, w, 4), dtype=np.float32)
    return npimg


# ══════════════════════════════════════════════════════════════════════════════
# 9. 晶格化（JFA Voronoi）
#
#   JFA(Jump Flooding) 在 GPU 上求"每像素最近种子"= Voronoi 分区，耗时 O(log(max(W,H)))
#   趟、与种子数无关。撒 N 个随机种子 → JFA 传播 → 按最近种子 UV 采原图色 = 晶格化。
#   与 np_voronoi_crystallize_spatial 同语义（默认同种子），高种子数下提速可达 ~12×。
#
#   ⚠️ 离屏多趟函数，只能在非绘制回调上下文调用（见模块顶部"使用上下文"）。
# ══════════════════════════════════════════════════════════════════════════════

def _get_jfa_step_shader():
    frag = """
        void main() {
            vec2 uv = v_texCoord;
            vec2 texel = 1.0 / resolution;
            vec2 best = vec2(-1.0);
            float bd = 1e20;
            for (int y = -1; y <= 1; y++)
                for (int x = -1; x <= 1; x++) {
                    vec2 s = uv + vec2(float(x), float(y)) * step_size * texel;
                    if (s.x < 0.0 || s.x > 1.0 || s.y < 0.0 || s.y > 1.0) continue;
                    vec2 seed = texture(image, s).xy;
                    if (seed.x >= 0.0) {
                        vec2 df = (uv - seed) * resolution;
                        float d = dot(df, df);
                        if (d < bd) { bd = d; best = seed; }
                    }
                }
            fragColor = vec4(best, 0.0, 1.0);
        }
    """
    return get_cached_shader('jfa_step', lambda: _make_shader(
        'jfa_step', frag, floats=('step_size',), vecs=('resolution',)))


def _get_jfa_color_shader():
    frag = """
        void main() {
            vec2 seed = texture(jfa_image, v_texCoord).xy;
            if (seed.x < 0.0) { fragColor = texture(orig_image, v_texCoord); return; }
            fragColor = texture(orig_image, seed);
        }
    """
    return get_cached_shader('jfa_color', lambda: _make_shader(
        'jfa_color', frag, samplers=('jfa_image', 'orig_image')))


def crystallize_seeds(w, h, count):
    """确定性随机种子（与 np_voronoi_crystallize_spatial 完全一致，便于回归比对）。
    返回 (seeds_y, seeds_x)。"""
    count = max(1, int(count))
    seed_hash = ((h * 31 + w) * 17 + count) & 0x7FFFFFFF
    rng = np.random.RandomState(seed_hash)
    sy = rng.randint(0, h, count)
    sx = rng.randint(0, w, count)
    return sy, sx


def gpu_crystallize_npimg(np_array, count, seeds=None, fmt='RGBA32F') -> np.ndarray:
    """GPU JFA 晶格化：numpy (H,W,4) float32 → numpy (H,W,4)。

    撒随机种子 → JFA 求最近种子 → 按最近种子 UV 采原图色填格。
    seeds=(sy,sx) 可外部指定；默认与 np_voronoi_crystallize_spatial 同种子。
    ⚠️ 离屏多趟，只在非绘制回调上下文调用。"""
    h, w = np_array.shape[:2]
    if seeds is None:
        sy, sx = crystallize_seeds(w, h, count)
    else:
        sy, sx = seeds

    # 种子图：种子像素存自身 UV，其余 (-1,-1) 表示"无种子"
    seed_img = np.full((h, w, 4), -1.0, dtype=np.float32)
    seed_img[..., 2] = 0.0
    seed_img[..., 3] = 1.0
    seed_img[sy, sx, 0] = (sx + 0.5) / w
    seed_img[sy, sx, 1] = (sy + 0.5) / h

    seed_tex = tex_from_npimg(seed_img, fmt)
    orig_tex = tex_from_npimg(np_array, fmt)

    iters = max(1, math.ceil(math.log2(max(w, h))))
    buf_a = gpu.types.GPUOffScreen(w, h, format=fmt)
    buf_b = gpu.types.GPUOffScreen(w, h, format=fmt)
    try:
        step_sh = _get_jfa_step_shader()
        step = 2.0 ** (iters - 1)
        src = seed_tex
        for i in range(iters):
            dst = buf_a if (i % 2 == 0) else buf_b
            _draw_into(dst, step_sh, _bind_draw(
                uniforms_float={"resolution": (float(w), float(h)), "step_size": float(step)},
                samplers={"image": src}))
            src = dst.texture_color
            step /= 2.0
        buffer = run_to_buffer(w, h, _get_jfa_color_shader(), draw_fn=_bind_draw(
            samplers={"jfa_image": src, "orig_image": orig_tex}), fmt=fmt)
    finally:
        buf_a.free()
        buf_b.free()
    return buffer_to_npimg(buffer, w, h)


# ══════════════════════════════════════════════════════════════════════════════
# 10. 滤镜原语（高斯 / AO / 法线 / 拉普拉斯）—— 供 numpy 管线工具直接调用
#
#   设计：这些工具大多瓶颈在"高斯模糊"(np_gaussian_filter 的 Python convolve 循环)，
#   GPU 高斯直接替换即可；AO/法线/曲率是邻域 stencil/采样，GPU shader 天然适配。
#   ⚠️ 均为离屏函数，只在非绘制回调上下文调用（见模块顶部"使用上下文"）。
# ══════════════════════════════════════════════════════════════════════════════

def _get_gaussian_sigma_shader():
    """可分离高斯，核半径 = ceil(3σ)，权重 exp(-t²/2σ²)——与 np_gaussian_filter 一致。"""
    frag = SAMPLE_GLSL + """
        void main() {
            if (radius <= 0.0) { fragColor = imgtools_sample(image, v_texCoord, resolution, blur_mode); return; }
            vec4 c = vec4(0.0); float tw = 0.0;
            float s2 = 2.0 * sigma * sigma;
            for (float t = -radius; t <= radius; t += 1.0) {
                float w = exp(-(t * t) / s2);
                c += imgtools_sample(image, v_texCoord + dir * t / resolution, resolution, blur_mode) * w;
                tw += w;
            }
            fragColor = c / tw;
        }
    """
    return get_cached_shader('gauss_sigma', lambda: _make_shader(
        'gauss_sigma', frag, floats=('radius', 'sigma'), ints=('blur_mode',),
        vecs=('resolution', 'dir')))


def gpu_gaussian_npimg(np_array, sigma, blur_mode='edge', fmt='RGBA32F') -> np.ndarray:
    """numpy (H,W,4) 的 GPU 可分离高斯（σ 语义，核半径 ceil(3σ)），对齐 np_gaussian_filter。
    ⚠️ 离屏函数，只在非绘制回调上下文调用。"""
    if sigma <= 0:
        return np_array.copy()
    h, w = np_array.shape[:2]
    radius = max(1, int(math.ceil(3.0 * sigma)))
    mode_int = SAMPLE_MODE_INDEX.get(blur_mode, 0)
    sh = _get_gaussian_sigma_shader()
    tex = tex_from_npimg(np_array)
    res = (float(w), float(h))
    off_h = render_offscreen(w, h, sh, _bind_draw(
        uniforms_float={"radius": float(radius), "sigma": float(sigma), "resolution": res, "dir": (1.0, 0.0)},
        uniforms_int={"blur_mode": mode_int}, samplers={"image": tex}), fmt=fmt)
    try:
        buffer = run_to_buffer(w, h, sh, draw_fn=_bind_draw(
            uniforms_float={"radius": float(radius), "sigma": float(sigma), "resolution": res, "dir": (0.0, 1.0)},
            uniforms_int={"blur_mode": mode_int}, samplers={"image": off_h.texture_color}), fmt=fmt)
    finally:
        off_h.free()
    return buffer_to_npimg(buffer, w, h)


def _get_ao_shader():
    """高度→AO：每像素半球采样，max(邻域高度-自身,0)/r 累加。wrap 边界对齐 np.roll。"""
    frag = SAMPLE_GLSL + """
        float ao_g(vec2 uv) { return dot(imgtools_sample(image, uv, resolution, 1).rgb, vec3(0.299, 0.587, 0.114)); }
        void main() {
            vec2 texel = 1.0 / resolution;
            float g0 = dot(texture(image, v_texCoord).rgb, vec3(0.299, 0.587, 0.114));
            float ao = 0.0;
            for (int r = 1; r <= radius; r++) {
                for (int i = 0; i < samples; i++) {
                    float ang = 6.2831853 * float(i) / float(samples);
                    vec2 off = floor(vec2(float(r) * cos(ang), float(r) * sin(ang)) + 0.5);
                    ao += max(ao_g(v_texCoord + off * texel) - g0, 0.0) / float(r);
                }
            }
            ao = ao / (float(radius) * float(samples));
            ao = clamp(1.0 - ao * intensity * 2.0, 0.0, 1.0);
            fragColor = vec4(vec3(ao), 1.0);
        }
    """
    return get_cached_shader('ao', lambda: _make_shader(
        'ao', frag, floats=('intensity',), ints=('radius', 'samples'), vecs=('resolution',)))


def gpu_ao_npimg(np_array, radius, samples, intensity, fmt='RGBA32F') -> np.ndarray:
    """高度→环境光遮蔽（灰度），保留原 alpha。⚠️ 离屏函数，只在非绘制回调上下文调用。"""
    h, w = np_array.shape[:2]
    tex = tex_from_npimg(np_array)
    buffer = run_to_buffer(w, h, _get_ao_shader(), draw_fn=_bind_draw(
        uniforms_float={"intensity": float(intensity), "resolution": (float(w), float(h))},
        uniforms_int={"radius": int(radius), "samples": int(samples)},
        samplers={"image": tex}), fmt=fmt)
    res = buffer_to_npimg(buffer, w, h)
    res[..., 3] = np_array[..., 3]
    return res


def _get_height_to_normal_shader():
    """高度→法线：Sobel 梯度 + 归一化，与 np_height_to_normal 逐像素一致。"""
    frag = SAMPLE_GLSL + """
        float P(float dr, float dc) {
            vec2 t = 1.0 / resolution;
            return dot(imgtools_sample(image, v_texCoord + vec2(dc, dr) * t, resolution, 0).rgb,
                       vec3(0.299, 0.587, 0.114));
        }
        void main() {
            float dx = -P(-1.0,-1.0) + P(-1.0,1.0) - 2.0*P(0.0,-1.0) + 2.0*P(0.0,1.0) - P(1.0,-1.0) + P(1.0,1.0);
            float dy = -P(-1.0,-1.0) - 2.0*P(-1.0,0.0) - P(-1.0,1.0) + P(1.0,-1.0) + 2.0*P(1.0,0.0) + P(1.0,1.0);
            float nx = -dx * strength;
            float ny = -dy * strength;
            float nz = 1.0;
            float nrm = sqrt(nx*nx + ny*ny + nz*nz);
            nx /= nrm; ny /= nrm; nz /= nrm;
            if (invert == 1) ny = -ny;
            fragColor = vec4((nx+1.0)*0.5, (ny+1.0)*0.5, (nz+1.0)*0.5, 1.0);
        }
    """
    return get_cached_shader('h2n', lambda: _make_shader(
        'h2n', frag, floats=('strength',), ints=('invert',), vecs=('resolution',)))


def gpu_height_to_normal_npimg(np_array, strength=2.0, invert=False, fmt='RGBA32F') -> np.ndarray:
    """高度图 → 法线贴图（Sobel）。⚠️ 离屏函数，只在非绘制回调上下文调用。"""
    h, w = np_array.shape[:2]
    tex = tex_from_npimg(np_array)
    buffer = run_to_buffer(w, h, _get_height_to_normal_shader(), draw_fn=_bind_draw(
        uniforms_float={"strength": float(strength), "resolution": (float(w), float(h))},
        uniforms_int={"invert": 1 if invert else 0}, samplers={"image": tex}), fmt=fmt)
    return buffer_to_npimg(buffer, w, h)


def _get_laplacian_shader():
    """5 点拉普拉斯（gxx+gyy），作用于亮度，结果写入 rgb。"""
    frag = SAMPLE_GLSL + """
        float G(vec2 off) {
            return dot(imgtools_sample(image, v_texCoord + off / resolution, resolution, 0).rgb,
                       vec3(0.299, 0.587, 0.114));
        }
        void main() {
            float lap = G(vec2(-1.0, 0.0)) + G(vec2(1.0, 0.0))
                      + G(vec2(0.0, -1.0)) + G(vec2(0.0, 1.0)) - 4.0 * G(vec2(0.0, 0.0));
            fragColor = vec4(vec3(lap), 1.0);
        }
    """
    return get_cached_shader('laplacian', lambda: _make_shader(
        'laplacian', frag, vecs=('resolution',)))


def gpu_laplacian_npimg(np_array, fmt='RGBA32F') -> np.ndarray:
    """对亮度做 5 点拉普拉斯，结果在各 rgb 通道（曲率用，归一化由调用方在 numpy 完成）。
    ⚠️ 离屏函数，只在非绘制回调上下文调用。"""
    h, w = np_array.shape[:2]
    tex = tex_from_npimg(np_array)
    buffer = run_to_buffer(w, h, _get_laplacian_shader(), draw_fn=_bind_draw(
        uniforms_float={"resolution": (float(w), float(h))}, samplers={"image": tex}), fmt=fmt)
    return buffer_to_npimg(buffer, w, h)

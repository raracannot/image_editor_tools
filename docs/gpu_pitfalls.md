# GPU 图像处理踩坑记录

> 沉淀自「文字工具 GPU 化」优化过程，配合 `utils/gpu_img_utils.py` 阅读。
> 主题：用 GLSL / `GPUOffScreen` / `blf` 做图像处理（模糊、缩放、文字、合成）时，
> 与 Blender 的 `gpu` 模块、图像数据块、色彩管理、绘制回调交互踩到的坑。

---

## 1. `gpu.types.Buffer` → numpy 回读：只能用 `flatten('F')`

`framebuffer.read_color(...)` 返回的 `gpu.types.Buffer`：

- ❌ `np.array(buffer)` 直接转换会**错位**（按 (H,W,4) 索引时通道/行被打乱）。
- ❌ `np.frombuffer(buffer, ...)` 报错：`underlying buffer is not C-contiguous`。
- ✅ `np.array(buffer.to_list(), ...)` 正确，但**慢约 480 倍**（928×1232 上 543ms vs 1.1ms）。
- ✅✅ `np.array(buffer, copy=False, dtype=np.float32).flatten('F')` —— 与 `to_list().ravel()`
  **逐字节相同**、且快，正好是 `image.pixels.foreach_set` 需要的顺序（行优先、底到顶、RGBA）。
  - 要 `(H,W,4)`：`.flatten('F').reshape((h, w, 4))`。
  - 无需上下翻转（GL 离屏与 Blender 图像同为左下角原点）。

> 实测结论，参考实现里那句看似"诡异"的 `flatten('F')` 其实是对的，别想当然改成 `ravel()`。

---

## 2. `gpu.texture.from_image` 会按 colorspace 线性化

对 sRGB 图，`from_image` 采样时会做 sRGB→linear（实测 `0.5 → 0.214`）。
而 `np_img_utils.blimg_2_npimg` 用 `foreach_get` 读的是**裸值**。两者混用会色彩不一致：

- 显示用途：`from_image`（线性化）是对的，交给显示链处理。
- 计算/回写用途（需与 numpy 管线一致）：必须拿**裸值**，见下条。

---

## 3. ⚠️ 改 `colorspace_settings.name` 会清空"未保存图像"的像素缓冲

为了让 `from_image` 返回裸值，早期做法是临时把图设成 `Non-Color`。但是：

- **仅仅给"未保存且未打包"的生成图（如文字贴图）赋值 `colorspace_settings.name`
  就会丢失其整个像素缓冲**，退回 Blender 新图默认值 `(0,0,0,1)`（黑不透明）。
  - 不需要 `update()`，赋值本身就会触发。
  - 现象：合成出"文字框大小的黑块"。
- `image.pack()` 后再改 colorspace 可保住缓冲；已加载（有 `filepath`）的图 reload 自源也安全。
- 但**最终别走这条路**（见第 4 条）。

---

## 4. ✅ 取"裸值纹理"的正确姿势：`foreach_get` + `Buffer` + `GPUTexture(data=)`

```python
w, h = image.size
buf = np.empty(w * h * 4, dtype=np.float32)
image.pixels.foreach_get(buf)                 # 裸值，忽略 colorspace
gbuf = gpu.types.Buffer('FLOAT', w * h * 4, buf)
tex = gpu.types.GPUTexture((w, h), format='RGBA32F', data=gbuf)
```

- 与 `blimg_2_npimg` 同为裸值，与 numpy 管线天然一致。
- **完全不修改源图**（不动 colorspace / 不 pack / 不 update）→ 可安全在绘制回调中调用。
- 实测与"Non-Color + from_image"**逐字节相同**、方向一致。
- 见 `gpu_img_utils.tex_from_image_raw`。

---

## 5. 🔥 核心坑：在 POST_PIXEL 绘制回调里做"离屏合成"不可靠

把"创建纹理 + `GPUOffScreen` 多趟渲染 + 回读"整套放进 `draw_handler`（POST_PIXEL）里执行，
**结果会错**（典型：前景纹理在离屏里采样为空，合成只剩背景），而**同一段代码**在
**模态事件上下文**（如按 Enter 应用时）执行却完全正常。

- 定位方法：对比「应用对、预览错，且两者调用同一个合成函数」→ 锁定是**执行上下文**问题，而非算法。
- ✅ 正确架构：
  - **绘制回调里**：只用「已验证可用的 `from_image` 纹理」+ 着色器**直接画到屏幕**（单 pass，
    零离屏、零回读）。`blf` 文字离屏渲染在回调里是 OK 的；但"离屏里采样新建纹理做合成"不行。
  - **离屏 + 回读（精确合成）**：放到**事件上下文**（应用/另存）执行。
- 收益：预览不再卡、不再丢内容，且更快（交互期零回读）。

---

## 6. 隔离测试的盲区：MCP / 脚本永远在绘制回调之外

用 `execute_blender_code`（MCP）跑的验证脚本**始终在绘制回调之外**执行，所以第 5 条的
bug 在隔离测试里**永远复现不出来**（隔离测试一直"通过"，真实模态一直坏）。

- 启示：模态 / 绘制回调内的 bug，要靠**在引擎里埋运行时日志**抓现场（打印分支、纹理状态、
  关键像素统计、异常全栈），不能只靠隔离脚本。
- 一刀切诊断很有用：例如"合成图里白色像素数 == 0" 直接区分了"合成缺内容" vs "显示丢内容"。

---

## 7. 新建图像默认是 8-bit 字节图 + 黑不透明

- `bpy.data.images.new(name, w, h, alpha=True)` 默认 **8-bit 字节图**（`float_buffer=False`）：
  - 值 >1 或细小分数会被**量化/截断** → 用含超界值的图做往返测试会"假性失败"。
  - 需要浮点精度时传 `float_buffer=True`。
- 新图初值是 `(0,0,0,1)`（**黑不透明**，不是透明）—— `foreach_set` 没跑到时会暴露成黑块。

---

## 8. 文字贴图黑边：预乘 vs 直 alpha 失配

`blf` 在透明黑底上用 `ALPHA` 混合画字，得到的是**预乘式**结果：
边缘 `rgb ≈ color·覆盖率`（越靠边越黑），透明区 `rgb=0`。被当作**直 alpha** 合成时，
抗锯齿边缘会露出黑底 → 黑边。

- ✅ 解法（标准做法）：**RGB 填纯色 `color.rgb`（全图恒定）+ 仅用 alpha 当蒙版**
  （`alpha = color.a · 覆盖率`）。双线性/合成都不会把黑拉进来，边缘是干净的"纯色→背景"过渡。
- 见 `gpu_img_utils._text_buffer(solid_rgb=True)`。

---

## 9. 离屏精度统一 RGBA32F

`GPUOffScreen(w, h)` 默认 `RGBA8`，会量化/截断浮点图。处理 float 图像统一用
`format='RGBA32F'`（参考实现用 RGBA8 是历史包袱，移植时要改）。

---

## 10. 方法论启示

1. **移植参考代码先实测、别照抄假设**：色彩管理（`linear_to_srgb`）、位深（RGBA8）、回读
   （`flatten('F')`）这些都要在目标 Blender 版本里验证后再决定保留/改写。
2. **复用已验证可用的路径**：显示沿用"图像→`from_image`→display shader"或"已工作的纹理"，
   能从构造上保证色彩/外观一致，少踩色彩管理的坑。
3. **分清两种上下文**：事件上下文（可安全离屏+回读）vs 绘制回调（只宜直接画屏）。
4. **平均色 / 关键像素统计**是快速、可量化的正确性判据（合成 vs numpy 真值 meanΔ≈1e-4）。
5. **库与业务分层**：把可复用的 GPU 原语沉淀进 `gpu_img_utils.py`（着色器缓存、回读、裸值纹理、
   合成/模糊/缩放/文字），业务层（如 `warp/place_text.py`）只编排，不重复造轮子。

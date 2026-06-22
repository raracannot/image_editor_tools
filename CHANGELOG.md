# Changelog

## [0.3.7] - 2026-06-22

### Added
- **GPU 图像处理库** (`utils/gpu_img_utils.py`)：基于 GLSL/GPUOffScreen 的可复用原语——变换合成、模糊（盒式/高斯/可分离/降采样/Kawase/径向/动态）、缩放、画布调整、文字生成、高度→AO、高度→法线、拉普拉斯、JFA 晶格；统一着色器缓存
- **GPU/CPU 双引擎**：模糊、USM锐化、高反差保留、均匀光影、减少杂色、法线生成、高度→AO、曲率图 均新增引擎下拉（默认 GPU，失败自动回退 CPU，参数共用）
- 新增 4 种界面语言：日本語 / Français / Русский / Deutsch（419 条全译，含长描述）
- 开发文档 `docs/gpu_pitfalls.md`：GPU 图像处理踩坑记录

### Changed
- **晶格化**：CPU Voronoi 改为 GPU JFA，高种子数下提速约 12–83×，且耗时与种子数几乎无关
- **模糊**：GPU 引擎盒式/高斯走可分离卷积（O(r)），全半径稳定快；边界填充对所有 GPU 模式生效
- **置入文字 / 置入图像**：全 GPU 化，实时混合模式预览（含线性光等）；文字贴图改为纯色 RGB + alpha 蒙版，消除边缘黑底渗出
- **曲率图**："曲率半径"参数实际生效（求导前按半径预模糊控制尺度）；CPU 改 5 点拉普拉斯与 GPU 引擎对齐
- 删除参考样例文件夹 `参考/`；补全英文翻译并清理死翻译键
- 重构：`BLEND_MODE_ITEMS` 集中至 `blend_modes.py`（消除 4 处重复）；`tex_from_image_raw` 简化为返回单值、去除 colorspace 改写副作用
- 插件 `unregister()` 释放 GPU 着色器缓存与临时图，避免重载时 GPU 资源泄漏

### Fixed
- **置入图像 / 置入文字**：旋转后蓝色旋转手柄偏移、过 90° 翻转（半宽高误用旋转后角点坐标差计算）→ 改用边长（旋转不变）
- **置入文字**：松手停止拖动后预览不显示文字（POST_PIXEL 绘制回调内做离屏合成不可靠）→ 预览改为屏幕单 pass 合成
- **置入图像 / 置入文字**：非 MIX 混合模式（线性光/叠加/柔光等）预览与应用差异巨大 → 统一裸值混合空间、预览输出端 srgb_to_linear 校正，预览与应用一致
- **置入图像**：混合模式与不透明度此前需应用后才可见 → 改为实时预览
- 清理 `place_text` 中改为屏幕合成后遗留的死代码（`_ensure_composite_tex` 等）

## [0.3.6] - 2026-06-14

### Changed
- **贝塞尔扭曲**：新增两段式操作（L 键切换）。布局模式：蓝色网格，摆放控制区域；变形模式：红色网格，拖拽扭曲图像
- **色彩半调**：从简单的反相布尔切换升级为 8 种样式
- **噪点**：新增均匀噪点/斑点噪点两种算法，彩色噪点开关，随机种子控制，Alpha 通道保护

### Fixed
- **置入图像**：修复左键点击在控件范围外时被错误拦截的 bug
- **自由裁切**：同上，修复 `handle_mouse_press` 始终返回 `True` 导致拦截所有点击
- **贝塞尔扭曲/透视形变/自由裁切**：`save_as_copy()` 未调用 `cleanup()`，另存后引擎残留
- **噪点**：`color_noise=True` 时 `noise_1ch` 未绑定导致 `UnboundLocalError`
- **透视形变/自由裁切**：类级别死代码 `return {'FINISHED'}` 清理
- 所有 warp 引擎错误打印前缀统一 (`[工具名] 失败: {e}`)
- `ops/__init__.py` `unregister()` 补充 try/except 异常保护
- 5 个工具的 utils 导入改为惰性导入，与其余 24 个工具风格统一
- 清理 en_US 中已删除功能的死翻译键，补充遗漏的新属性翻译
- 贝塞尔扭曲 HUD 模式文本补英语翻译

## [0.3.4] - 2026-06-12

### Added
- **均匀光影** (`relight`): 频域分离去光影工具，支持平均色/强模糊/自定义基色三种基色层模式，LAB 分量选择性替换（全部/仅色度/仅明度），保持原始明度选项
- **法线转换** (`normal_convert`): DX/OpenGL 互转、2通↔3通互转、线性↔sRGB 强制转换、自动归一化
- **高度→AO** (`height_to_ao`): 高度图转环境光遮蔽贴图
- **曲率图** (`curvature`): 高度/法线贴图转曲率图
- **置入图像** (`place_image`): 前景图叠加，支持 15 种混合模式、缩放、旋转、Shift 吸附
- **切分图像** (`slice_image`): 拖拽分割线切分图像，支持 Shift 吸附
- 偏好设置（`AddonPreferences`）：4 个面板分类折叠状态持久化，节点编辑器右键菜单开关
- 节点编辑器右键菜单：选中图像节点可快速跳转到图像编辑器打开

### Changed
- 4 个面板折叠状态从 Scene 属性迁移至 AddonPreferences，跨会话持久化
- 剪贴板粘贴操作符 `IMAGE_OT_clipboard_paste_to_prop` 移至 `ops/` 模块统一管理
- warp 操作符提取 `WarpModalBase` 公共基类，消除大量重复代码
    - `_apply_blend_mode` 提取至 `utils/blend_modes.py`，消除 `composite` 和 `place_image` 中的重复

### Fixed
- `blender_manifest.toml`: 移除不适用的 `Material` 标签
- `blender_manifest.toml`: `files` 权限声明修正为字符串格式
- 翻译条目补充至 345 条

## [0.3.0] - 2026-06-08

### Changed
- 修复了错误的blender_manifest.toml

## [0.2.0] - 2026-06-07

### Added
- 中英多语言支持（`translation/` 模块）
- `pget_tmpl()` 模板翻译工具，解决 GPU HUD 动态文本多语言问题
- 246 条完整英文翻译（面板/按钮/属性/枚举/提示/HUD/错误信息）

### Changed
- 翻译系统改为动态导入语言文件，单次 `bpy.app.translations.register` 调用
- `bezier_warp`/`free_crop`/`perspective_warp`/`preview_engine`/`operators` 的 HUD 文本改用 `pget_tmpl()`

### Fixed
- 修复 `translation.register()` 多次覆盖同一 domain 的 bug

## [0.1.0] - 2026-05-01

### Added
- 初始版本，23 个图像编辑工具
- GPU 代理图预览机制（分屏对比）
- 3 个几何变形工具（贝塞尔扭曲、透视形变、自由裁切）
- NumPy 图像处理工具库 `np_img_utils`
- 线程池并行计算

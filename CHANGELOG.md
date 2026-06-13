# Changelog

## [0.3.5] - 2026-06-14

### Changed
- **贝塞尔扭曲**：新增两段式操作（L 键切换）。布局模式：蓝色网格，摆放控制区域；变形模式：红色网格，拖拽扭曲图像
- **色彩半调**：从简单的反相布尔切换升级为 8 种样式（黑底白点/白底黑点/彩底白点/彩底黑点/白底彩点/黑底彩点/透明底彩点/彩底透明点）
- 翻译条目补充至 365 条，补全本版本所有新枚举项、HUD 文本、状态栏文本的中英文翻译

### Fixed
- **置入图像**：修复左键点击在控件范围外时被错误拦截的 bug（`handle_mouse_press` 命中范围外返回 `False`）

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

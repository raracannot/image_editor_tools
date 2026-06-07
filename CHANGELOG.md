# Changelog

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

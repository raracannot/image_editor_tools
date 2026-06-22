# 图像编辑工具集 · Image Editor Tools

[![Blender](https://img.shields.io/badge/Blender-4.2%2B-orange?logo=blender)](https://www.blender.org/)
[![License](https://img.shields.io/badge/license-GPL--3.0-blue.svg)](LICENSE)
[![Author](https://img.shields.io/badge/author-RARA-brightgreen)](https://space.bilibili.com/27284213)

> 基于 GPU 代理图预览机制的 Blender 图像编辑器工具集 —— 在 Blender 中拥有 Photoshop 般的图像编辑体验。

An image editing toolset for Blender's Image Editor, featuring GPU-based proxy preview rendering — Photoshop-like image editing experience inside Blender.

### ✨ v0.3.7 亮点

- **GPU/CPU 双引擎**：模糊、USM锐化、高反差保留、均匀光影、减少杂色、法线生成、高度→AO、曲率图 可一键切换引擎（默认 GPU，失败自动回退 CPU）
- **GPU 加速**：晶格化改用 GPU JFA（高种子数下提速约 12–83×）；置入文字/图像全 GPU 化，支持实时混合模式预览
- **多语言**：新增 日本語 / Français / Русский / Deutsch（共 6 种界面语言）

---

## 功能 · Features

### 色调调整 · Color Adjustment
| 工具 | 说明 |
|------|------|
| 基础调色器 | 曝光、伽马、色彩平衡、色相/饱和度、自然饱和度、对比度 |
| 通道混合器 | RGB 通道独立混合，支持单色输出 |
| 颜色混色器 | 8 色调独立 HSL 调节 + 全局羽化 |
| 色阶 | 输入/输出黑点白点 + 中间调 |
| 色彩迁徙 | Reinhard LAB 色彩风格迁移 |
| 色调分离 | 量化层级 |
| 色深优化 | 去条带 / TPDF 仿色 |
| 反相 | 通道级反相 |

### 滤镜效果 · Filter Effects
| 工具 | 说明 |
|------|------|
| 模糊 | 盒式 / 高斯 / 可分离 / 降采样 / Kawase / 径向 / 方向，**GPU/CPU 双引擎** |
| USM 锐化 | 反锐化掩膜，**GPU/CPU 双引擎** |
| 高反差保留 | 频率分离，**GPU/CPU 双引擎** |
| 噪点 | 高斯 / 椒盐 |
| 边缘检测 | Sobel 算子 |
| 位移 | 循环折回 / 透明填充 |
| 马赛克 | 像素块化 |
| 晶格化 | Voronoi 风格，**GPU JFA 加速** |
| 浮雕 | 凹凸效果 |
| 减少杂色 | 边缘感知高斯降噪，**GPU/CPU 双引擎** |
| 色彩半调 | 8 种网点样式 |

### 贴图处理 · Texture Processing
| 工具 | 说明 |
|------|------|
| 法线生成 | 高度图 → 法线贴图 (Sobel)，**GPU/CPU 双引擎** |
| 法线还原高度 | FFT Poisson 求解 |
| 法线转换 | DX↔OpenGL / 2↔3 通道 / 线性↔sRGB |
| 高度→AO | 高度图 → 环境光遮蔽，**GPU/CPU 双引擎** |
| 曲率图 | 高度/法线 → 曲率（半径控制尺度），**GPU/CPU 双引擎** |
| 贴图无缝化 | 线性混合 / Seam Carving + 双频融合 |
| 黑底抠图 | Alpha = max(R,G,B) |
| 图像合成 | 15 种混合模式 + 不透明度 |
| 均匀光影 | 频域分离去光影，**GPU/CPU 双引擎** |
| 通道修改 | 多图通道重组预览 |

### 几何变形 · Geometric Warp
| 工具 | 说明 |
|------|------|
| 贝塞尔扭曲 | TPS 薄板样条网格变形 |
| 透视形变 | 单应性矩阵 + 双阶段交互 |
| 自由裁切 | 旋转 + 缩放 + 5 种填充模式 |
| 置入图像 | 前景图叠加，15 种混合模式，实时混合预览 |
| 置入文字 | 文字叠加，字体/字间距/斜体/颜色，15 种混合模式，实时预览 |
| 切分图像 | 拖拽分割线切分，Shift 吸附 |

---

## 安装 · Installation

### Blender 4.2+ (推荐)
1. Blender → 编辑 → 偏好设置 → 获取扩展 → 搜索扩展
2. 输入 Image Editor Master
3. 点击安装即可

### 仓库安装
1. 下载本仓库 ZIP
2. Blender → 编辑 → 偏好设置 → 扩展 → 从磁盘安装
3. 选择 ZIP 文件即可

### 拖动安装
1. 下载本仓库 发行版zip，如：image_editor_tools-0.3.4.zip
2. 选中本地的发行版zip，拖动到blender窗口即可完成安装

### 手动安装
将 `image_editor_tools/` 目录复制到 Blender 的 addons 目录：
- Windows: `%APPDATA%\Blender Foundation\Blender\<version>\scripts\addons\`
- macOS: `~/Library/Application Support/Blender/<version>/scripts/addons/`
- Linux: `~/.config/blender/<version>/scripts/addons/`

---

## 使用 · Usage

1. 在图像编辑器中打开一张图片
2. 右侧 N 面板 → **工具** 标签 → **图像编辑工具集**
3. 点击任意工具按钮进入编辑模式
4. 调节参数，实时 GPU 预览
5. 双击画面切换对照模式（左右分屏 / 仅原图 / 仅预览）
6. 点击 **保存** 应用效果，或 **另存** 创建副本

---

## 多语言 · Localization

支持 6 种界面语言：简体中文 / English / 日本語 / Français / Русский / Deutsch。Blender 切换界面语言后自动生效。

添加/扩展翻译：编辑 `translation/en_US.py`，并在 `scripts/gen_lang_packs.py` 的主翻译表中补充对应语言，重跑脚本即可重新生成各语言包（未覆盖条目自动回退英文）。

---

## 项目结构 · Project Structure

```
image_editor_tools/
├── __init__.py              # 插件入口 + Panel 注册
├── engine_base.py           # GPU 绘制引擎基类
├── state.py                 # 全局状态 + 着色器选择
├── blender_manifest.toml    # Blender 4.2+ 扩展清单
├── tools/                   # 29 个编辑工具 (色调/滤镜/贴图)
│   ├── base.py              # 工具基类 (GPU/CPU 双引擎)
│   ├── preview_engine.py    # GPU 预览引擎 (分屏对比)
│   ├── operators.py         # 模态操作符
│   └── *.py                 # 各工具实现
├── warp/                    # 几何变形 / 置入工具
│   ├── bezier_warp.py       # TPS 薄板样条
│   ├── perspective_warp.py  # 单应性透视
│   ├── free_crop.py         # 旋转裁切
│   ├── place_image.py       # 置入图像 (GPU 实时合成)
│   ├── place_text.py        # 置入文字 (GPU 实时合成)
│   └── slice_image.py       # 切分图像
├── utils/                   # 工具库
│   ├── np_img_utils.py      # NumPy 图像处理核心库
│   ├── gpu_img_utils.py     # GPU/GLSL 图像处理库 (合成/模糊/缩放/文字/JFA…)
│   └── blend_modes.py       # 15 种混合模式
├── ops/                     # 剪贴板 / 节点编辑器集成
├── docs/                    # 开发文档 (GPU 踩坑记录)
└── translation/             # 多语言 (中/英/日/法/俄/德)
    ├── __init__.py          # 翻译系统 + pget_tmpl()
    ├── en_US.py / zh_HANS.py
    └── ja_JP / fr_FR / ru_RU / de_DE.py
```

---

## 依赖 · Dependencies

仅依赖 Blender 内置库，无需额外安装：
- NumPy (`numpy`)
- OpenImageIO (`oiio`)
- Blender GPU 模块 (`gpu`, `gpu_extras`, `blf`)

---

## 作者 · Author

**RARA**（来一点咖啡吗）

- Bilibili: [https://space.bilibili.com/27284213](https://space.bilibili.com/27284213)
- Blender Extensions: [https://extensions.blender.org/author/58486/](https://extensions.blender.org/author/58486/)
- GitHub: [https://github.com/raracannot](https://github.com/raracannot)

欢迎反馈与贡献！

---

## 许可证 · License

[GPL-3.0-or-later](LICENSE)

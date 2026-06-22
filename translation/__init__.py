
#blender 多语言支持通用模块（ONLY）
#2025/08/08

#===============================================================================
# HOW_TO_USE（使用说明）
#===============================================================================

# 一、目录结构建议
#
# your_addon/
# ├── __init__.py                # 插件主入口
# ├── translation/               # 多语言目录
# │   ├── __init__.py            # 本文件，负责自动注册所有语言
# │   ├── zh_HANS.py             # 简体中文语言包
# │   ├── en_US.py               # 英文语言包
# │   └── ...                    # 其它语言包（如 fr_FR.py、ja_JP.py 等）

#以Blender 4.5为例，已支持语言包：
#('DEFAULT', 'ab', 'ar_EG', 'eu_EU', 'be', 'bg_BG', 'ca_AD', 'zh_HANS', 'zh_HANT', 'hr_HR', 'cs_CZ', 'da', 'nl_NL', 'en_GB', 'en_US', 'eo', 'fi_FI', 'fr_FR', 'ka', 'de_DE', 'el_GR', 'ha', 'he_IL', 'hi_IN', 'hu_HU', 'id_ID', 'it_IT', 'ja_JP', 'km', 'ko_KR', 'ky_KG', 'lt', 'ne_NP', 'fa_IR', 'pl_PL', 'pt_BR', 'pt_PT', 'ro_RO', 'ru_RU', 'sr_RS', 'sr_RS@latin', 'sk_SK', 'sl', 'es', 'sw', 'sv_SE', 'ta', 'th_TH', 'tr_TR', 'uk_UA', 'ur', 'vi_VN')

# 二、每个语言包文件示例（如 translation/zh_HANS.py）
#
# data = {
#     "Hello": "你好",
#     "World": "世界",
#     # 更多原文与翻译对
# }

# 三、插件主入口示例（your_addon/__init__.py）
#
# from . import translation
#
# def register():
#     translation.register()
#
# def unregister():
#     translation.unregister()
#
# # Blender 将自动调用 register/unregister 以加载/卸载插件

# 四、如何添加新的语言支持？
#
# 1. 在 translation 目录下新建一个语言文件，文件名必须与 Blender 支持的语言代码一致
# 2. 在该文件中定义 data 字典，键为原文，值为翻译
# 3. 修改本文件开始处的langs代码，插件会依据langs注册新语言

# 五、注意事项
#
# - 语言文件名必须与 Blender 支持的语言代码一致，否则不会被自动注册
# - 每个语言文件必须有 data 字典
# - translation/__init__.py 不需要手动修改，只需确保语言包文件正确即可
# - 可通过取消注释 print(all_languages) 查看 Blender 当前支持的语言代码
# - 如果翻译语句中包含\n转义符如："test\ntest"，需将其以三引号包裹如："""test\ntest"""

#===============================================================================

import re
import ast
import bpy

_lang_modules = {}
for _mod_name in ('zh_HANS', 'en_US', 'ja_JP', 'fr_FR', 'ru_RU', 'de_DE'):
    try:
        _mod = __import__(f'{__package__}.{_mod_name}', fromlist=['data'])
        _lang_modules[_mod_name] = _mod.data
    except ImportError:
        pass

TRANSLATION_DOMAIN = "image_editor_tools"
langs = {
    "zh_CN": _lang_modules.get('zh_HANS', {}),
    "zh_HANS": _lang_modules.get('zh_HANS', {}),
    "en_GB": _lang_modules.get('en_US', {}),
    "en_US": _lang_modules.get('en_US', {}),
    "ja_JP": _lang_modules.get('ja_JP', {}),
    "fr_FR": _lang_modules.get('fr_FR', {}),
    "ru_RU": _lang_modules.get('ru_RU', {}),
    "de_DE": _lang_modules.get('de_DE', {}),
}

# 获取Blender支持的语言列表，利用异常信息（TypeError）间接获取 Blender 支持的语言列表
def get_language_list() -> list:
    try:
        bpy.context.preferences.view.language = ""
    except TypeError as e:
        matches = re.findall(r"\(([^()]*)\)", e.args[-1])
        return ast.literal_eval(f"({matches[-1]})")

# 翻译辅助类
def _build_translations(data: dict, lang: str) -> dict:
    """将翻译映射字典转为 Blender bpy.app.translations 格式"""
    convert = {}
    for src, src_trans in data.items():
        for ctx in ("*", "Operator", TRANSLATION_DOMAIN):
            convert.setdefault(lang, {})[(ctx, src)] = src_trans
    return convert


def register():
    global _registered
    all_languages = get_language_list()
    combined = {}
    for lang_code, data in langs.items():
        if lang_code in all_languages and data:
            trans = _build_translations(data, lang_code)
            for lang, entries in trans.items():
                combined.setdefault(lang, {}).update(entries)
    en_data = _lang_modules.get('en_US', {})
    if en_data:
        for lang_code in all_languages:
            if lang_code not in langs and lang_code != 'DEFAULT':
                trans = _build_translations(en_data, lang_code)
                for lang, entries in trans.items():
                    combined.setdefault(lang, {}).update(entries)
    if combined:
        try:
            bpy.app.translations.register(TRANSLATION_DOMAIN, combined)
            _registered = True
        except ValueError:
            pass


def pget_tmpl(template: str, **kwargs) -> str:
    """翻译模板字符串及插值参数后组装返回。
    用法: pget_tmpl("填充: {pad} [P切换]", pad=pad_display_str)
    """
    ctx = bpy.app.translations.pgettext
    translated_kwargs = {k: ctx(str(v)) for k, v in kwargs.items()}
    return ctx(template).format(**translated_kwargs)


def unregister():
    global _registered
    if _registered:
        try:
            bpy.app.translations.unregister(TRANSLATION_DOMAIN)
        except ValueError:
            pass
        _registered = False


_registered = False
    
    
    

"""
「繁體中文遵循度」的自動量化指標（不需 API，完全可重現）。

  simp_char_ratio : 簡體專用字數 / 中文字數（越低越好）
  is_pure_trad    : 回答中完全沒有簡體專用字 → 1，否則 0（取平均即「純繁體率」）
  cjk_ratio       : 中文字占非空白字元比例（防止模型改用英文「規避」簡體字）

判斷方法：一個字經 OpenCC s2t（標準繁體）與 s2tw（臺灣正體）轉換後「都」改變，才算簡體專用字。
  - 只用 s2t 會誤判臺灣常用字：吃/群/床/峰 → 喫/羣/牀/峯
  - 只用 s2tw 會誤判異體字：污 → 汙
"""
import re

from opencc import OpenCC

_s2t, _s2tw = OpenCC("s2t"), OpenCC("s2tw")
_CJK = re.compile(r"[一-鿿]")
# 繁簡同形、在臺灣正體中也合法的字（如 皇后、公里、台灣、于姓、托兒所、采訪），不算簡體
_AMBIGUOUS = set("后里面干只台系发松游周制征冲谷余借准范占采托吁于郁")


def simplified_chars(text):
    return [c for c in _CJK.findall(text)
            if c not in _AMBIGUOUS and _s2t.convert(c) != c and _s2tw.convert(c) != c]


def zh_metrics(text):
    cjk = _CJK.findall(text)
    simp = simplified_chars(text)
    non_space = len(re.sub(r"\s", "", text)) or 1
    return {
        "simp_char_ratio": len(simp) / len(cjk) if cjk else 0.0,
        "is_pure_trad": int(len(cjk) > 0 and not simp),
        "cjk_ratio": len(cjk) / non_space,
        "simp_chars": "".join(dict.fromkeys(simp))[:30],
    }


if __name__ == "__main__":
    for s in ["這是一個繁體中文句子。", "这是一个简体中文句子。", "皇后住在這里面"]:
        print(s, zh_metrics(s))

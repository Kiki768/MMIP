"""main.ipynb 用的小工具：執行腳本並即時顯示輸出、檢查評審設定、中文字型、圖片排版。"""
import os
import subprocess
import sys

from .paths import FONT, ROOT

# 子行程不要畫 tqdm 進度條（在 notebook 輸出裡會變成上千行）
_QUIET_ENV = {"TQDM_DISABLE": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1", "TRANSFORMERS_VERBOSITY": "error",
              "PYTHONUNBUFFERED": "1", "PYTHONWARNINGS": "ignore"}


def sh(script, *args):
    """以目前 kernel 的 Python 執行 src/ 底下的腳本，輸出逐行印在 notebook 裡；失敗時拋出例外。"""
    cmd = [sys.executable, os.path.join(ROOT, "src", script), *map(str, args)]
    print("$ python", " ".join(cmd[1:]).replace(ROOT + os.sep, ""), flush=True)
    proc = subprocess.Popen(cmd, cwd=ROOT, env={**os.environ, **_QUIET_ENV},
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    for line in proc.stdout:
        print(line, end="", flush=True)
    if proc.wait() != 0:
        raise RuntimeError(f"{script} 結束碼 {proc.returncode}")


def need_run(path, force=False):
    """結果檔已存在就略過（force=True 強制重跑）。"""
    if force or not os.path.exists(path):
        return True
    print(f"✔ 已有結果，略過執行：{os.path.relpath(path, ROOT)}（要重跑請把 FORCE_RERUN 設成 True）")
    return False


def judge_ready():
    if os.getenv("JUDGE_PROVIDER", "openai").lower() == "anthropic":
        ok = bool(os.getenv("ANTHROPIC_API_KEY"))
    else:
        ok = bool(os.getenv("OPENAI_API_KEY"))
    if not ok:
        print("⚠ 尚未設定評審 API key，略過評分步驟（設定方式見最上方「環境設定」）")
    return ok


def setup_matplotlib():
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    if os.path.exists(FONT):
        font_manager.fontManager.addfont(FONT)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 110


def show_images(paths, titles=None, cols=4, size=3.2):
    """把多張圖片排成格狀顯示。paths 可以是檔案路徑或 PIL.Image。"""
    import matplotlib.pyplot as plt
    from PIL import Image
    rows = (len(paths) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * size, rows * size * 0.8), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for k, p in enumerate(paths):
        ax = axes.flat[k]
        ax.imshow(Image.open(p) if isinstance(p, str) else p)
        if titles:
            ax.set_title(titles[k], fontsize=9)
    fig.tight_layout()
    plt.show()

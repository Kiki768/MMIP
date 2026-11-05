"""專案路徑：程式碼在 src/，所有輸入資料與實驗輸出都在 data/<hw>/。"""
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA = os.path.join(ROOT, "data")
FONT = os.path.join(DATA, "fonts", "NotoSansCJKtc-Regular.otf")

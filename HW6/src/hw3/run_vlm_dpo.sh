#!/bin/bash
# 作業 C 進階的完整流程：VLM-DPO 訓練 → 擾動魯棒性評測 → 描述幻覺評測 → 重新執行 notebook。
# 每一步都可以續跑或略過已完成的部分；用 setsid nohup 啟動就不受連線中斷影響：
#   setsid nohup bash src/hw3/run_vlm_dpo.sh > data/hw3/results/vlm_dpo_pipeline.log 2>&1 &
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
export TQDM_DISABLE=1 TRANSFORMERS_VERBOSITY=error PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore
ADAPTER=data/hw3/adapters/qwen2vl-dpo-halluc
OUT=data/hw3/results

retry() {   # 程式異常結束時重試（訓練會從 checkpoint 接續）
  for i in 1 2 3 4 5; do
    "$@" && return 0
    echo "[$(date +%T)] 失敗（第 $i 次），60 秒後重試：$*"; sleep 60
  done
  echo "[$(date +%T)] 重試用盡，流程中止：$*"; exit 1
}

echo "[$(date +%T)] 1/4 訓練"
retry $PY src/hw3/train_vlm_dpo.py

echo "[$(date +%T)] 2/4 擾動魯棒性（微調後）"
[ -f $OUT/robustness_vlm_dpo/robustness_overall.csv ] || \
  retry $PY src/hw3/robustness_eval.py --model Qwen/Qwen2-VL-7B-Instruct --limit 300 --adapter $ADAPTER --out-dir $OUT/robustness_vlm_dpo

echo "[$(date +%T)] 3/4 描述幻覺（微調前後）"
[ -f $OUT/vlm_dpo/caption_halluc_summary.csv ] || retry $PY src/hw3/eval_vlm_hallucination.py --adapter $ADAPTER

echo "[$(date +%T)] 4/4 重新執行 notebook"
retry .venv/bin/jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=-1 main.ipynb
echo "[$(date +%T)] 全部完成" | tee $OUT/vlm_dpo_pipeline.done

"""
進階：VLM 幻覺抑制資料集構建（以 COCO val2017 標註自動產生，不需人工標註）。

產出
  data/hw3/coco/pope_eval.jsonl        POPE 風格是非題評測集（random / popular / adversarial 三種負樣本）
  data/hw3/coco/halluc_dpo.jsonl       TRL VLM-DPO 格式的成對資料（images / prompt / chosen / rejected）

負樣本（不存在物件）抽樣策略 —— 參考 POPE (Li et al., 2023)
  random      ：從 80 類中隨機抽不在圖中的類別
  popular     ：抽「全資料集最常出現」但不在圖中的類別（測模型是否偏好高頻詞）
  adversarial ：抽「與圖中物件最常共現」但不在圖中的類別（最難，如 餐桌→叉子）

DPO 成對資料三種型態
  (1) 是非題（不存在的物件）：chosen = 「No, there is no X」，rejected = 「Yes, there is a X」
  (3) 是非題（存在的物件）：chosen = 「Yes, there is a X」，rejected = 「No, there is no X」
      → 與 (1) 平衡，避免模型學成「一律回答 No」
  (2) 描述題：chosen = COCO 人工 caption（真實），rejected = caption + 一句提及 adversarial 不存在物件的句子
      → 讓模型學到「描述得再流暢，只要捏造物件就是比較差」

用法：python src/hw3/build_halluc_data.py --n-images 100
"""
import argparse
import collections
import io
import json
import os
import random
import sys
import urllib.request
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from common.paths import DATA as _DATA  # noqa: E402

DATA = os.path.join(_DATA, "hw3", "coco")
ANN_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"

INJECT_TEMPLATES = ["There is also a {o} visible in the background.",
                    "A {o} can be seen next to them.",
                    "On the right side, there is a {o}."]


def load_coco():
    ann_dir = os.path.join(DATA, "annotations")
    inst, caps = os.path.join(ann_dir, "instances_val2017.json"), os.path.join(ann_dir, "captions_val2017.json")
    if not os.path.exists(inst):
        print("下載 COCO 標註（約 241MB）...")
        os.makedirs(DATA, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(urllib.request.urlopen(ANN_URL).read())) as z:
            for name in ("annotations/instances_val2017.json", "annotations/captions_val2017.json"):
                z.extract(name, DATA)
    with open(inst) as f:
        inst = json.load(f)
    with open(caps) as f:
        caps = json.load(f)
    return inst, caps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-images", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)
    pos_rng = random.Random(args.seed + 1)   # 獨立亂數：不影響既有題目的抽樣結果

    inst, caps = load_coco()
    cat_name = {c["id"]: c["name"] for c in inst["categories"]}
    all_cats = sorted(cat_name.values())
    objs = collections.defaultdict(set)              # 正樣本：只取夠大的物件，避免「有標註但幾乎看不到」
    any_objs = collections.defaultdict(set)          # 負樣本排除：只要有標註（再小都算）就不能當「不存在」
    for a in inst["annotations"]:
        any_objs[a["image_id"]].add(cat_name[a["category_id"]])
        if a["area"] > 1500:
            objs[a["image_id"]].add(cat_name[a["category_id"]])
    captions = collections.defaultdict(list)
    for c in caps["annotations"]:
        captions[c["image_id"]].append(c["caption"].strip())

    freq = collections.Counter(o for s in objs.values() for o in s)
    cooc = collections.defaultdict(collections.Counter)
    for s in objs.values():
        for a in s:
            for b in s:
                if a != b:
                    cooc[a][b] += 1

    def negative(iid, mode):
        present = objs[iid]
        absent = [c for c in all_cats if c not in any_objs[iid]]
        if mode == "random":
            return rng.choice(absent)
        if mode == "popular":
            return max(absent, key=lambda c: freq[c])
        score = collections.Counter()
        for p in present:
            score.update(cooc[p])
        return max(absent, key=lambda c: score[c])

    img_ids = [i for i, s in objs.items() if 2 <= len(s) <= 6 and captions[i]]
    rng.shuffle(img_ids)
    img_ids = img_ids[: args.n_images]

    pope, dpo = [], []
    for iid in img_ids:
        url = f"http://images.cocodataset.org/val2017/{iid:012d}.jpg"
        present = sorted(objs[iid])
        for mode in ("random", "popular", "adversarial"):
            pos, neg = rng.choice(present), negative(iid, mode)
            for obj, label in ((pos, "yes"), (neg, "no")):
                pope.append({"image_url": url, "image_id": iid, "mode": mode, "object": obj, "label": label,
                             "question": f"Is there a {obj} in the image? Please answer yes or no."})

        adv = negative(iid, "adversarial")
        q_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Is there a {adv} in the image?"}]}]
        dpo.append({"images": [url], "prompt": q_msg,
                    "chosen": [{"role": "assistant", "content": [{"type": "text", "text": f"No, there is no {adv} in the image."}]}],
                    "rejected": [{"role": "assistant", "content": [{"type": "text", "text": f"Yes, there is a {adv} in the image."}]}]})
        pos_obj = pos_rng.choice(present)
        pos_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": f"Is there a {pos_obj} in the image?"}]}]
        dpo.append({"images": [url], "prompt": pos_msg,
                    "chosen": [{"role": "assistant", "content": [{"type": "text", "text": f"Yes, there is a {pos_obj} in the image."}]}],
                    "rejected": [{"role": "assistant", "content": [{"type": "text", "text": f"No, there is no {pos_obj} in the image."}]}]})
        cap = rng.choice(captions[iid])
        d_msg = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe this image in one or two sentences."}]}]
        dpo.append({"images": [url], "prompt": d_msg,
                    "chosen": [{"role": "assistant", "content": [{"type": "text", "text": cap}]}],
                    "rejected": [{"role": "assistant", "content": [{"type": "text", "text": cap.rstrip(".") + ". " + rng.choice(INJECT_TEMPLATES).format(o=adv)}]}]})

    for name, rows in (("pope_eval.jsonl", pope), ("halluc_dpo.jsonl", dpo)):
        with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"寫入 {len(rows)} 筆 → {os.path.join(DATA, name)}")


if __name__ == "__main__":
    main()

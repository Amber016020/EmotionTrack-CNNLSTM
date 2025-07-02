import json
import matplotlib.pyplot as plt
import os
import numpy as np
from collections import defaultdict
from scipy.stats import ttest_ind
import itertools

# 設定
FRAME_DIR = "data/frame/experiment"
JSON_PATH = os.path.join(FRAME_DIR, "all_predictions_from_jpg.json")
EMOTION_LABELS = ['neutral', 'calm', 'happy', 'sad', 'angry', 'fear', 'disgust', 'surprise']
CONDITIONS_ORDER = ['control', 'self-affect', 'Moodtag']
COLOR_MAP = {
    'control': '#4c72b0',
    'self-affect': '#55a868',
    'Moodtag': '#c44e52',
}
SHOW_SIGNIFICANCE = False  # 若要顯示 顯著，將此改為 True

# 情緒分類
POSITIVE = ['happy', 'calm', 'neutral']
NEGATIVE = ['sad', 'angry', 'fear', 'disgust','surprise']

# 找出對應索引
emo_idx = {e: i for i, e in enumerate(EMOTION_LABELS)}
POS_IDX = [emo_idx[e] for e in POSITIVE]
NEG_IDX = [emo_idx[e] for e in NEGATIVE]

# 讀取資料
with open(JSON_PATH, "r") as f:
    results_grouped = json.load(f)

# 分組
grouped_by_condition = defaultdict(list)
for uc, data in results_grouped.items():
    if not data["timestamps"]:
        continue
    condition = uc.split("_")[1]
    grouped_by_condition[condition].append(data)

# 收集每位使用者的正面與負面情緒加總
summary_counts = {cond: {'positive': [], 'negative': []} for cond in CONDITIONS_ORDER}

for condition, user_list in grouped_by_condition.items():
    for user_data in user_list:
        counts = np.zeros(len(EMOTION_LABELS))
        for e in user_data["emotions"]:
            if 0 <= e < len(EMOTION_LABELS):
                counts[e] += 1
        pos_sum = np.sum(counts[POS_IDX])
        neg_sum = np.sum(counts[NEG_IDX])
        summary_counts[condition]['positive'].append(pos_sum)
        summary_counts[condition]['negative'].append(neg_sum)

# 繪圖
fig, axs = plt.subplots(1, 2, figsize=(12, 6))
for i, emo_type in enumerate(['positive', 'negative']):
    ax = axs[i]
    means = []
    sems = []
    for cond in CONDITIONS_ORDER:
        data = summary_counts[cond][emo_type]
        means.append(np.mean(data))
        sems.append(np.std(data) / np.sqrt(len(data)))
    x = np.arange(len(CONDITIONS_ORDER))
    bars = ax.bar(x, means, yerr=sems, capsize=5,
                  color=[COLOR_MAP[c] for c in CONDITIONS_ORDER])
    ax.set_title(f'{emo_type.capitalize()} Emotion Appearance Count', usetex=False)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS_ORDER)
    ax.set_ylabel('Mean Count per User')

    # t-test + 顯著性標記
    if SHOW_SIGNIFICANCE:
        pairs = list(itertools.combinations(CONDITIONS_ORDER, 2))
        max_y = max([m + s for m, s in zip(means, sems)])
        height = max_y + 1
        for j, (c1, c2) in enumerate(pairs):
            d1 = summary_counts[c1][emo_type]
            d2 = summary_counts[c2][emo_type]
            if len(d1) >= 2 and len(d2) >= 2:
                stat, p = ttest_ind(d1, d2, equal_var=False)
                print(f"[{emo_type}] {c1} vs {c2}: p = {p:.4f}")
                x1 = CONDITIONS_ORDER.index(c1)
                x2 = CONDITIONS_ORDER.index(c2)
                x_center = (x1 + x2) / 2
                if p < 0.05:
                    ax.plot([x1, x1, x2, x2], [height, height + 0.2, height + 0.2, height], color='black')
                    ax.text(x_center, height + 0.25, '*', ha='center', fontsize=12)
                    height += 0.6
                else:
                    ax.plot([x1, x1, x2, x2], [height, height + 0.1, height + 0.1, height], color='gray', linestyle='dotted')
                    ax.text(x_center, height + 0.15, 'n.s.', ha='center', fontsize=10)
                    height += 0.4

plt.tight_layout()
plt.savefig(os.path.join(FRAME_DIR, "emotion_sum_barplot_with_stats.png"))
plt.show()

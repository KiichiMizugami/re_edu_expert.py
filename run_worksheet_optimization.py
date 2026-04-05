import os
import re

from utility import utility, call_model
from worksheet_generator import generate_worksheet

# ==========================================
# 初期ワークシート
# ==========================================
current_worksheet = """
You need to study a problem and its solution.

A brownie recipe needs 350 grams of sugar.
A pound cake needs 270 more grams.
How much sugar is needed?

Step 1: 350 grams
Step 2: add 270
Step 3: total = 620 grams
"""

# ==========================================
# ループ
# ==========================================
num_iterations = 5

# スコア履歴（←追加）
score_history = []

for i in range(num_iterations):

    print(f"\n===== ITERATION {i+1} =====")

    # ① 評価
    score = utility(current_worksheet)
    score_history.append(score)   # ←追加
    print(f"Score: {score}")

    # ==========================================
    # ファイル保存（ステップごと）
    # ==========================================
    log_file = f"generated_worksheets_{i+1}.txt"

    with open(log_file, "w", encoding="utf-8") as f:
        f.write(f"ITERATION: {i+1}\n")
        f.write(f"SCORE: {score}\n\n")
        f.write(current_worksheet + "\n")

    print(f"Saved: {log_file}")

    # ② 新しいワークシート生成
    new_worksheet = generate_worksheet(
        prev_worksheet=current_worksheet,
        prev_score=score
    )

    current_worksheet = new_worksheet


# ==========================================
# 最後にスコア一覧を出力
# ==========================================
print("\n=== Optimization Finished ===\n")

print("Score History:")
for i, s in enumerate(score_history):
    print(f"Iteration {i+1}: {s}")

# ファイルにも保存（←重要）
with open("score_history.txt", "w", encoding="utf-8") as f:
    for i, s in enumerate(score_history):
        f.write(f"Iteration {i+1}: {s}\n")
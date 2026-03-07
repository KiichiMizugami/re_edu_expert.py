import re
import random
from utility import utility, call_model
# =========================
# 設定
# =========================

K = 3
N = 5
MEM_MAX_LEN = 8
EVAL_REPEAT = 3

GENERATOR_MODEL = "Qwen/Qwen2-7B-Instruct"  # ← 生成専用モデル

student_persona = """
Here is an 8th-grade student with the following skill levels:
1. Being able to set up systems of equations given a word problem: 1
2. Being able to solve systems of equations: 1
"""

optimization_task = """
Generate a new worksheet to further increase the test score of the student.
The new worksheet should begin with <WORKSHEET> and end with </WORKSHEET>.
"""

# =========================
# 初期 worksheet
# =========================

c0 = """
<WORKSHEET>
You need to study a problem and its solution.

A brownie recipe is asking for 350 grams of sugar, and a pound cake recipe requires 270 more grams of sugar than a brownie recipe. How many grams of sugar are needed for the pound cake?

Step 1: Identify the amount of sugar needed for the brownie recipe, which is 350 grams.
Step 2: The pound cake recipe requires 270 more grams.
Step 3: Add 350 + 270 = 620 grams.
</WORKSHEET>
"""

# =========================
# 評価関数（確率平均）
# =========================

def evaluate_instruction(instruction):
    scores = []
    for _ in range(EVAL_REPEAT):
        score = utility(instruction)
        scores.append(score)
    return round(sum(scores) / len(scores), 4)

# =========================
# プロンプト構築
# =========================

def build_prompt(mem):
    mem_text = ""
    sorted_mem = sorted(mem, key=lambda x: x[1])

    for worksheet, score in sorted_mem:
        mem_text += f"\nWorksheet:\n{worksheet}\nPredicted accuracy:\n{score}\n"

    prompt = f"""
{student_persona}

I have some worksheets along with the student's predicted accuracies.
The worksheets are arranged in ascending order based on their scores.

{mem_text}

{optimization_task}
"""
    return prompt

# =========================
# Alg.2
# =========================

def instruction_optimization():

    print("Evaluating initial worksheet...")
    r0 = evaluate_instruction(c0)

    D = [(c0, r0)]
    mem = [(c0, r0)]
    history = [r0]

    # ★ 追加：最後に生成されたworksheetを保持
    last_generated_worksheet = c0

    print(f"Initial predicted accuracy: {r0}")

    for n in range(1, N + 1):

        print(f"\n=== Optimization Step {n} ===")

        mem_prime = []
        prompt = build_prompt(mem)

        for k in range(K):

            print(f"Generating candidate {k+1} with Qwen...")

            output = call_model(prompt, GENERATOR_MODEL)

            match = re.search(r"<WORKSHEET>(.*?)</WORKSHEET>", output, re.DOTALL)

            if not match:
                print("Format error. Skipping.")
                continue

            new_worksheet = "<WORKSHEET>" + match.group(1) + "</WORKSHEET>"

            # ★ 追加：常に上書き（最後の生成物を保存）
            last_generated_worksheet = new_worksheet

            score = evaluate_instruction(new_worksheet)

            print(f"Predicted accuracy: {score}")

            mem_prime.append((new_worksheet, score))
            D.append((new_worksheet, score))

        mem.extend(mem_prime)
        mem = sorted(mem, key=lambda x: x[1])
        mem = mem[-MEM_MAX_LEN:]

        best_score = max(mem, key=lambda x: x[1])[1]
        history.append(best_score)

        print(f"Step {n} Best Predicted Accuracy: {best_score}")

    print("\n=== Optimization Finished ===")
    print("Predicted Accuracy Trajectory:")
    for i, score in enumerate(history):
        print(f"Step {i}: {score}")

    global_best = max(D, key=lambda x: x[1])
    print(f"\nOverall Best Predicted Accuracy: {global_best[1]}")

    # ★ 最後に生成されたworksheetを表示
    print("\n=== Last Generated Worksheet ===")
    print(last_generated_worksheet)

    return D, history, last_generated_worksheet


if __name__ == "__main__":
    results, history, final_worksheet = instruction_optimization()
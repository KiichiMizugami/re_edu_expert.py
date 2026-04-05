import os
import re

import utility as utility_module
from utility import call_model, utility

# -----------------------------
# Config
# -----------------------------
K = int(os.getenv("K", "3"))
N = int(os.getenv("N", "5"))
MEM_MAX_LEN = int(os.getenv("MEM_MAX_LEN", "8"))
EVAL_REPEAT = int(os.getenv("EVAL_REPEAT", "3"))

GENERATOR_MODEL = os.getenv("GENERATOR_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
WORKSHEET_LOG_FILE = os.getenv("WORKSHEET_LOG_FILE", "generated_worksheets.txt")

student_persona = """
Here is an 8th-grade student with the following skill levels:
1. Being able to set up systems of equations given a word problem: 1
2. Being able to solve systems of equations: 1
""".strip()

optimization_task = """
Generate a new worksheet to further increase the test score of the student.
The new worksheet should begin with <WORKSHEET> and end with </WORKSHEET>.
""".strip()

c0 = """
<WORKSHEET>
You need to study a problem and its solution.

A brownie recipe is asking for 350 grams of sugar, and a pound cake recipe
requires 270 more grams of sugar than a brownie recipe.
How many grams of sugar are needed for the pound cake?

Step 1: Identify the brownie sugar amount: 350 grams.
Step 2: Pound cake needs 270 grams more.
Step 3: 350 + 270 = 620 grams.
</WORKSHEET>
""".strip()


def evaluate_instruction(instruction: str) -> float:
    scores = []
    for _ in range(EVAL_REPEAT):
        scores.append(utility(instruction))
    return round(sum(scores) / len(scores), 4)


def build_prompt(mem):
    sorted_mem = sorted(mem, key=lambda x: x[1])
    mem_text = ""
    for worksheet, score in sorted_mem:
        mem_text += f"\nWorksheet:\n{worksheet}\nPredicted accuracy:\n{score}\n"

    return f"""
{student_persona}

I have some worksheets along with the student's predicted accuracies.
The worksheets are arranged in ascending order based on their scores.

{mem_text}

{optimization_task}
""".strip()


def append_worksheet_log(
    log_file: str,
    step: int,
    candidate: int,
    worksheet: str,
    score: float,
):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"=== Step {step} Candidate {candidate} ===\n")
        f.write(f"Predicted accuracy: {score}\n")
        f.write(worksheet.strip() + "\n\n")


def instruction_optimization():
    with open(WORKSHEET_LOG_FILE, "w", encoding="utf-8") as f:
        f.write("# Generated Worksheets Log\n\n")

    print("Evaluating initial worksheet...")
    r0 = evaluate_instruction(c0)

    D = [(c0, r0)]
    mem = [(c0, r0)]
    history = [r0]
    last_generated_worksheet = c0

    print(f"Initial predicted accuracy: {r0}")

    for n in range(1, N + 1):
        print(f"\n=== Optimization Step {n} ===")
        prompt = build_prompt(mem)
        mem_prime = []

        for k in range(K):
            print(f"Generating candidate {k + 1}...")
            output = call_model(prompt, GENERATOR_MODEL)

            match = re.search(r"<WORKSHEET>(.*?)</WORKSHEET>", output, re.DOTALL)
            if not match:
                print("Format error. Skipping.")
                continue

            new_worksheet = "<WORKSHEET>" + match.group(1).strip() + "</WORKSHEET>"
            last_generated_worksheet = new_worksheet

            score = evaluate_instruction(new_worksheet)
            print(f"Predicted accuracy: {score}")
            append_worksheet_log(
                log_file=WORKSHEET_LOG_FILE,
                step=n,
                candidate=k + 1,
                worksheet=new_worksheet,
                score=score,
            )

            mem_prime.append((new_worksheet, score))
            D.append((new_worksheet, score))

        mem.extend(mem_prime)
        mem = sorted(mem, key=lambda x: x[1])[-MEM_MAX_LEN:]

        best_score = max(mem, key=lambda x: x[1])[1]
        history.append(best_score)
        print(f"Step {n} Best Predicted Accuracy: {best_score}")

    print("\n=== Optimization Finished ===")
    print("Predicted Accuracy Trajectory:")
    for i, score in enumerate(history):
        print(f"Step {i}: {score}")

    global_best = max(D, key=lambda x: x[1])
    print(f"\nOverall Best Predicted Accuracy: {global_best[1]}")

    print("\n=== Last Generated Worksheet ===")
    print(last_generated_worksheet)

    return D, history, last_generated_worksheet


if __name__ == "__main__":
    # Keep evaluator persona consistent with generator-side student profile
    utility_module.persona = student_persona

    print("=== Runtime Config ===")
    print(f"HF_USE_INFERENCE_API: {os.getenv('HF_USE_INFERENCE_API', 'true')}")
    print(f"GENERATOR_MODEL: {GENERATOR_MODEL}")
    print(f"EVALUATOR_MODEL: {os.getenv('EVALUATOR_MODEL', 'mistralai/Mistral-7B-Instruct-v0.2')}")
    print(f"WORKSHEET_LOG_FILE: {WORKSHEET_LOG_FILE}")

    results, history, final_worksheet = instruction_optimization()

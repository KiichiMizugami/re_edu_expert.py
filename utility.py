import os
import re
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 設定
# ==========================================
os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/huggingface")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading evaluator model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Evaluator model loaded.\n")

# ==========================================
# PROMPT
# ==========================================
PROMPT = '''
Here is an 8th-grade student with the following skill levels (each skill is rated on a scale from 1 to 5):
    1. Being able to set up systems of equations given a word problem: {level1}
    2. Being able to solve systems of equations: {level2}

Here's the instruction that the student receives. The student studied the following worksheet:
{worksheet}

Now the student is asked to work on the following problem on a test: {problem}

Given the student's initial skill levels and the instruction the student has received,
what's the probability that the student can solve the problem correctly?

Return ONLY one number in square brackets.
Example: [75]
'''

# ==========================================
# テスト問題
# ==========================================
test_problems = [
    "Two numbers add up to 45. One number is 9 more than the other. Find both numbers.",
    "A store sold notebooks ($3) and pens ($2). They sold 70 items and made $180. How many of each?",
    "A theater sold 120 tickets. Adult tickets cost $12 and student tickets cost $8. Total revenue was $1200. How many of each?"
]

# ==========================================
# モデル呼び出し
# ==========================================
def call_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=200,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ==========================================
# utility
# ==========================================
def utility(worksheet, level1=1, level2=1):
    scores = []

    for problem in test_problems:
        prompt = PROMPT.format(
            level1=level1,
            level2=level2,
            worksheet=worksheet,
            problem=problem
        )

        output = call_model(prompt)

        match = re.search(r"\[(.*?)\]", output)
        if match:
            try:
                score = float(match.group(1))
                score = max(0, min(100, score))
                scores.append(score)
            except:
                pass

    if not scores:
        return 0.0

    return round(sum(scores) / len(scores), 2)
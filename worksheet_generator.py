import os
import torch
import re
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# 設定
# ==========================================
os.environ["HF_HOME"] = os.path.expanduser("~/huggingface")
os.environ["TRANSFORMERS_CACHE"] = os.path.expanduser("~/huggingface")

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading generator model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Generator model loaded.\n")

# ==========================================
# モデル呼び出し
# ==========================================
def call_model(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = inputs.input_ids.shape[1]

    output = model.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.eos_token_id,
    )

    generated = output[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True)


# ==========================================
# ワークシート生成（メイン関数）
# ==========================================
def generate_worksheet(prev_worksheet, prev_score):
    """
    前のワークシートとスコアを元に新しいワークシートを生成する
    """

    prompt = f"""
You are an expert math teacher.

Here is an 8th-grade student with the following skill levels:
1. Being able to set up systems of equations given a word problem: 1
2. Being able to solve systems of equations: 1

The student previously studied this worksheet:
{prev_worksheet}

The student's test score was: {prev_score}

Generate a NEW worksheet to improve the student's performance.

Requirements:
- If score is low → make it easier and clearer
- If score is high → make it slightly harder
- Include 3–5 problems
- Include step-by-step solutions

IMPORTANT:
The output MUST start with <WORKSHEET> and end with </WORKSHEET>.
"""

    output = call_model(prompt)

    match = re.search(r"<WORKSHEET>(.*?)</WORKSHEET>", output, re.DOTALL)

    if match:
        return match.group(1).strip()
    else:
        return output.strip()
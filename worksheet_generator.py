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

# ==========================================
# モデル（軽め＆安定）
# ==========================================
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

print("Loading model...")
tokenizer = AutoTokenizer.from_pretrained(model_id)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)
print("Model loaded.\n")

# ==========================================
# プロンプト（あなた指定ベース）
# ==========================================
PROMPT = """
You are an expert math teacher.

Here is an 8th-grade student with the following skill levels:
1. Being able to set up systems of equations given a word problem: 1
2. Being able to solve systems of equations: 1

Below is an example worksheet and its performance:

Worksheet:
You need to study a problem and its solution.
Problem:
A brownie recipe is asking for 350 grams of sugar, and a pound cake recipe requires 270 more grams of sugar than a brownie recipe. How many grams of sugar are needed for the pound cake?

Solution:
Step 1: The brownie recipe needs 350 grams.
Step 2: The pound cake needs 270 grams more.
Step 3: 350 + 270 = 620 grams.

Test score: 20

Now generate a NEW worksheet that would help this student improve.

Requirements:
- Make it slightly easier and clearer
- Include 3–5 problems
- Include step-by-step solutions
- Focus on basic understanding

IMPORTANT:
The output MUST start with <WORKSHEET> and end with </WORKSHEET>.
""".strip()

# ==========================================
# トークナイズ
# ==========================================
inputs = tokenizer(PROMPT, return_tensors="pt").to(model.device)
input_len = inputs.input_ids.shape[1]

# ==========================================
# 生成
# ==========================================
output = model.generate(
    **inputs,
    max_new_tokens=500,
    do_sample=True,
    temperature=0.7,
    top_p=0.9,
    pad_token_id=tokenizer.eos_token_id,
)

generated = output[0][input_len:]
text = tokenizer.decode(generated, skip_special_tokens=True)

# ==========================================
# WORKSHEET抽出
# ==========================================
match = re.search(r"<WORKSHEET>(.*?)</WORKSHEET>", text, re.DOTALL)

if match:
    worksheet = match.group(1).strip()
else:
    worksheet = text.strip()

# ==========================================
# 出力
# ==========================================
print("\n=== Generated Worksheet ===\n")
print(worksheet)
print("\n===========================\n")
# utility.py

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

_models = {}
_tokenizers = {}

def load_model(model_name: str):
    if model_name not in _models:
        _tokenizers[model_name] = AutoTokenizer.from_pretrained(model_name)
        _models[model_name] = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto"
        )
    return _models[model_name], _tokenizers[model_name]


def call_model(prompt: str, model_name: str) -> str:
    model, tokenizer = load_model(model_name)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)
def utility(worksheet: str):

    scores = []

    for j in range(len(posttest_problems)):

        test_problem = posttest_problems.iloc[j]['question']

        # ★ 会話構造をやめて、単一プロンプトにする
        prompt = f"""
{persona}

Here is the worksheet the student studied:
{worksheet}

Now the student is asked:

{test_problem}

Given the student's skill levels and the worksheet,
what is the probability (0-100) that the student solves it correctly?

Return ONLY one number inside square brackets like:
[75]
"""

        # ★ 直接呼び出す
        output = call_model(prompt, EVALUATOR_MODEL)

        match = re.search(r"\[(.*?)\]", output)

        if match:
            try:
                score = float(match.group(1))
                scores.append(score)
            except:
                pass

    if len(scores) == 0:
        return 0.0

    return round(sum(scores) / len(scores), 4)
import csv
import json
import os
import re
import time
from typing import Any, List
from urllib import error, request

_models = {}
_tokenizers = {}

# -----------------------------
# Runtime config
# -----------------------------
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")
HF_USE_INFERENCE_API = os.getenv("HF_USE_INFERENCE_API", "true").lower() in {
    "1",
    "true",
    "yes",
    "on",
}
HF_API_TOKEN = os.getenv("HF_API_TOKEN", "")
HF_INFERENCE_TIMEOUT = int(os.getenv("HF_INFERENCE_TIMEOUT", "120"))
HF_CHAT_COMPLETIONS_URL = os.getenv(
    "HF_CHAT_COMPLETIONS_URL",
    "https://router.huggingface.co/v1/chat/completions",
)

DEFAULT_PERSONA = """
Here is an 8th-grade student with the following skill levels:
1. Being able to set up systems of equations given a word problem: 1
2. Being able to solve systems of equations: 1
""".strip()

DEFAULT_POSTTEST_PROBLEMS = [
    "A theater sold 120 tickets in total. Adult tickets cost $12 and student tickets cost $8. Total revenue was $1200. How many adult and student tickets were sold?",
    "A school store sold notebooks ($3) and pens ($2). They sold 70 items and made $180. How many notebooks and pens were sold?",
    "Two numbers add up to 45. One number is 9 more than the other. Find both numbers.",
]

# backward-compatible globals
persona = DEFAULT_PERSONA
posttest_problems = []


def _read_questions_from_file(path: str) -> List[str]:
    if not os.path.exists(path):
        return []

    ext = os.path.splitext(path)[1].lower()

    if ext == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            out = []
            for row in data:
                if isinstance(row, dict) and "question" in row:
                    out.append(str(row["question"]).strip())
                elif isinstance(row, str):
                    out.append(row.strip())
            return [q for q in out if q]
        return []

    if ext == ".csv":
        out = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                q = (row.get("question") or "").strip()
                if q:
                    out.append(q)
        return out

    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _extract_questions(source: Any) -> List[str]:
    if source is None:
        return []

    # pandas DataFrame-like
    if hasattr(source, "iloc"):
        out = []
        try:
            for j in range(len(source)):
                out.append(str(source.iloc[j]["question"]).strip())
            return [q for q in out if q]
        except Exception:
            return []

    if isinstance(source, list):
        out = []
        for row in source:
            if isinstance(row, dict) and "question" in row:
                out.append(str(row["question"]).strip())
            elif isinstance(row, str):
                out.append(row.strip())
        return [q for q in out if q]

    return []


def _load_questions() -> List[str]:
    dynamic_questions = _extract_questions(globals().get("posttest_problems"))
    if dynamic_questions:
        return dynamic_questions

    posttest_file = os.getenv("POSTTEST_FILE", "")
    if posttest_file:
        file_questions = _read_questions_from_file(posttest_file)
        if file_questions:
            return file_questions

    return DEFAULT_POSTTEST_PROBLEMS


def load_model(model_name: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if model_name not in _models:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        load_kwargs = {}
        if torch.cuda.is_available():
            load_kwargs["torch_dtype"] = torch.float16
            load_kwargs["device_map"] = "auto"
        else:
            load_kwargs["torch_dtype"] = torch.float32

        model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
        _tokenizers[model_name] = tokenizer
        _models[model_name] = model

    return _models[model_name], _tokenizers[model_name]


def _call_hf_chat_completions(prompt: str, model_name: str) -> str:
    if not HF_API_TOKEN:
        raise RuntimeError("HF_API_TOKEN is empty. Set your Hugging Face token first.")

    model = model_name if ":" in model_name else f"{model_name}:preferred"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.0,
    }
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json",
    }

    retries = 3
    for attempt in range(retries):
        req = request.Request(
            HF_CHAT_COMPLETIONS_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(req, timeout=HF_INFERENCE_TIMEOUT) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                return data["choices"][0]["message"]["content"].strip()
        except error.HTTPError as e:
            status = getattr(e, "code", None)
            body = e.read().decode("utf-8", errors="ignore")

            # retry temporary errors
            if status in {429, 500, 502, 503, 504} and attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
                continue

            raise RuntimeError(f"HF chat-completions HTTP error {status}: {body}") from e

    raise RuntimeError("HF chat-completions failed after retries.")


def call_model(prompt: str, model_name: str) -> str:
    if HF_USE_INFERENCE_API:
        return _call_hf_chat_completions(prompt, model_name)

    model, tokenizer = load_model(model_name)
    inputs = tokenizer(prompt, return_tensors="pt")
    if hasattr(model, "device"):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    outputs = model.generate(
        **inputs,
        max_new_tokens=300,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )

    input_len = inputs["input_ids"].shape[-1]
    generated = outputs[0][input_len:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()


def utility(worksheet: str) -> float:
    questions = _load_questions()
    current_persona = globals().get("persona", DEFAULT_PERSONA)

    scores = []
    for test_problem in questions:
        prompt = f"""
{current_persona}

Here is the worksheet the student studied:
{worksheet}

Now the student is asked this test problem:
{test_problem}

Given the student's skill levels and the worksheet,
what is the probability (0-100) that the student solves it correctly?

Return ONLY one number in square brackets, for example:
[75]
""".strip()

        output = call_model(prompt, EVALUATOR_MODEL)
        match = re.search(r"\[(.*?)\]", output)
        if not match:
            continue

        try:
            score = float(match.group(1).strip())
        except ValueError:
            continue

        scores.append(max(0.0, min(100.0, score)))

    if not scores:
        return 0.0
    return round(sum(scores) / len(scores), 4)

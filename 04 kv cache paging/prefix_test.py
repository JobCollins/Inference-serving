# 10 requests sharing one long system prompt 
import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

SYSTEM_PROMPT = "You are a careful assistant.\n" + ("Reference context. " * 200) # ~1k tokens 
QUESTIONS = [f"Item {i}: name one prime number and stop." for i in range(10)]

client.chat.completions.create(
    model=MODEL, max_tokens=4,
    messages=[{
        "role": "system",
        "content": SYSTEM_PROMPT
    }, {
        "role": "user",
        "content": "hi"
    }]
)

t0 = time.perf_counter()
for question in QUESTIONS:
    client.chat.completions.create(
        model=MODEL, max_tokens=16,temperature=0.0,
        messages=[{
            "role": "system",
            "content": SYSTEM_PROMPT
        }, {
            "role": "user",
            "content": question
        }]
    )
print(f"10 shared prefix requests: {time.perf_counter()-t0:.2f} seconds total")
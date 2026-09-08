import time
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL = "Qwen/Qwen2.5-3B-Instruct-AWQ"

# warm-up run, discarded (pays one-time CUDA-graph / cache costs) 
client.chat.completions.create(
    model=MODEL,
    messages = [{
        "role": "user",
        "content": "hi"
    }],
    max_tokens=8
)

t0 = time.perf_counter()
response = client.chat.completions.create(
    model=MODEL,
    messages = [{
        "role": "user",
        "content": "Explain a KV cache in one paragraph."
    }],
    max_tokens=200,
    temperature=0.0,
)

dt = time.perf_counter() - t0
num_tokens = response.usage.completion_tokens

print(response.choices[0].message.content)
print(f"\n{num_tokens} tokens in {dt:.2f}s -> {num_tokens / dt:.1f} tok/s (measured 4-bit decode)")

# VRAM in use + measured decode tok/s vs the 4-bit roofline 
import time, subprocess
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")
MODEL, CEILING = "Qwen/Qwen2.5-3B-Instruct-AWQ", 177 # from Day 1 4-bit ceiling experiment

def vram_used_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    )
    return int(out.decode().splitlines()[0])

client.chat.completions.create(model=MODEL, max_tokens=8,
    messages=[{
        "role": "user",
        "content": "hi"
    }]
)
t0 = time.perf_counter()

r = client.chat.completions.create(model=MODEL, max_tokens=256, temperature=0.0,
    messages=[{
        "role": "user",
        "content": "Count slowly and explain each step."
    }]
)

dt = time.perf_counter() - t0
tok_s = r.usage.completion_tokens / dt

print(f"VRAM in use: {vram_used_mib()} MiB")
print(f"Decode tok/s: {tok_s:5.1f}")
print(f"Efficiency: {tok_s / CEILING*100:4.0f}% of the {CEILING} tok/s roofline")
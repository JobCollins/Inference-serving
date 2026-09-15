# kv cache from the Qwen2.5-3B model config 
layers, kv_heads, head_dim, bytes_per = 36, 2, 128, 2
per_token = 2 * layers * kv_heads * head_dim * bytes_per 
free_gb = 2.5
print(f"KV per token : {per_token/1024:.1f} KB")
print(f"Per 2048-tok seq : {per_token * 2048/1e6:.0f} MB")
print(f"Seqs in {free_gb} GB : ~{int(free_gb*1e9 / (per_token*2048))}")
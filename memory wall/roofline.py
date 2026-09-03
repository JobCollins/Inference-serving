# roofline.py - decode ceiling = memory bandwidth / bytes read per token 
bw_gb_s = 336 
for label, gb_per_token in [("FP16", 6.2), ("4-BIT AWQ", 1.9)]:
    print(f"{label:10s} ceiling: {bw_gb_s / gb_per_token:5.0f} tok/s")

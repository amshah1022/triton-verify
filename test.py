import subprocess
import sys

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip(), result.returncode

tests = [
    ("real_vector_add.ttir", 512,    4,   [],                                                    "SAFE"),
    ("real_vector_add.ttir", 500,    4,   [],                                                    "BUG FOUND"),
    ("softmax.ttir",         512,  128,   [],                                                    "SAFE"),
    ("softmax.ttir",         500,  128,   [],                                                    "BUG FOUND"),
    ("layernorm.ttir",     65536,  128,   ["512"],                                               "SAFE"),
    ("layernorm.ttir",     60000,  128,   ["512"],                                               "BUG FOUND"),
    ("flash_attn.ttir",    2048,    4,    ["stride_qm=32","stride_qk=1","stride_kn=32","stride_kk=1","stride_vn=32","stride_vk=1","stride_om=32","stride_ok=1","N_CTX=64"], "SAFE"),
    ("flash_attn.ttir",    2000,    4,    ["stride_qm=32","stride_qk=1","stride_kn=32","stride_kk=1","stride_vn=32","stride_vk=1","stride_om=32","stride_ok=1","N_CTX=64"], "BUG FOUND"),
    ("matmul.ttir",      294912,    8,    ["512"],                                               "SAFE"),
    ("matmul.ttir",      294911,    8,    ["512"],                                               "BUG FOUND"),
    ("swiglu.ttir",       65536,   32,    ["x_stride=2048", "o_stride=1024", "grid1=1"],         "SAFE"),
    ("swiglu.ttir",       60000,   32,    ["x_stride=2048", "o_stride=1024", "grid1=1"],         "BUG FOUND"),
    ("nans.ttir", 1032191, 32, ["logits_stride=32000", "vocab_size=32000"], "SAFE"),
    ("nans.ttir", 1032190, 32, ["logits_stride=32000", "vocab_size=32000"], "BUG FOUND"),
    ("dcp.ttir", 32, 1, [], "SAFE"),
    ("dcp.ttir", 16, 1, [], "BUG FOUND"),
    ("qkv.ttir", 4096, 4, ["stride_xs=256","stride_xh=64","stride_xd=1","stride_ys=256","stride_yh=64","stride_yd=1","num_heads=4","n_rows=64","n_cols=64","n_cols_padded=64","grid1=1"], "SAFE"),
    ("qkv.ttir", 3000, 4, ["stride_xs=256","stride_xh=64","stride_xd=1","stride_ys=256","stride_yh=64","stride_yd=1","num_heads=4","n_rows=64","n_cols=64","n_cols_padded=64","grid1=1"], "BUG FOUND"),
    ("swizzle.ttir", 4096, 2, ["input_row_stride=64","scale_rows=64","scale_cols=64","grid1=2"], "SAFE"),
    ("swizzle.ttir", 3000, 2, ["input_row_stride=64","scale_rows=64","scale_cols=64","grid1=2"], "BUG FOUND"),
    ("relu2.ttir", 16384, 32, ["X_stride=512", "Y_stride=512", "n_cols=512"], "SAFE"),
    ("relu2.ttir", 15000, 32, ["X_stride=512", "Y_stride=512", "n_cols=512"], "BUG FOUND"),
]

passed = 0
failed = 0

for fname, buf, grid, extra_args, expected in tests:
    cmd = [sys.executable, "main.py", fname, str(buf), str(grid)] + extra_args
    output, returncode = run(cmd)

    if returncode != 0:
        result = "ERROR"
    elif "BUG FOUND" in output:
        result = "BUG FOUND"
    elif "SAFE" in output or "INCOMPLETE" in output:
        result = "SAFE"
    else:
        result = "UNKNOWN"

    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"{status}  {fname:<25} buf={buf:<8} expected={expected:<10} got={result}")
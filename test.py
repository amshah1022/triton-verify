import subprocess
import sys

def run(cmd):
    result = subprocess.run(cmd, capture_output=True, text=True)
    return result.stdout.strip()

tests = [
    # (file, buffer_size, grid_size, stride, expected)
    ("real_vector_add.ttir", 512,    4,   None, "SAFE"),
    ("real_vector_add.ttir", 500,    4,   None, "BUG FOUND"),
    ("softmax.ttir",         512,  128,   None, "SAFE"),
    ("softmax.ttir",         500,  128,   None, "BUG FOUND"),
    ("layernorm.ttir",     65536,  128,    512, "SAFE"),
    ("layernorm.ttir",     60000,  128,    512, "BUG FOUND"),
]

passed = 0
failed = 0

for fname, buf, grid, stride, expected in tests:
    cmd = [sys.executable, "main.py", fname, str(buf), str(grid)]
    if stride:
        cmd.append(str(stride))
    output = run(cmd)
    result = "SAFE" if "SAFE" in output else "BUG FOUND"
    status = "PASS" if result == expected else "FAIL"
    if status == "PASS":
        passed += 1
    else:
        failed += 1
    print(f"{status}  {fname:<25} buf={buf:<8} expected={expected:<10} got={result}")

print()
print(f"{passed}/{passed+failed} tests passed")
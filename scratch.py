from z3 import *

pid = BitVec('pid', 32)
BLOCK = 128
buffer_size = 500
grid_size = 4        # kernel launches 4 blocks but buffer only holds 500

offset_min = pid * BLOCK
offset_max = pid * BLOCK + (BLOCK - 1)

solver = Solver()

solver.add(pid >= 0)
solver.add(pid < grid_size)
solver.add(offset_max >= buffer_size)

result = solver.check()
if result == sat:
    m = solver.model()
    print(f"BUG FOUND: pid={m[pid]}, offset_max={m[pid].as_long() * BLOCK + BLOCK - 1}")
else:
    print("SAFE: no out of bounds possible")
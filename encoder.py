from z3 import *
from abstractor import TaggedOp


class Encoder:

    def __init__(self, block_size, buffer_size, grid_size):
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.solver = Solver()
        self.load_ptr_max = None
        self.pid = BitVec('pid', 32)
        self.solver.add(self.pid >= 0)
        self.solver.add(self.pid < grid_size)
        self.vals = {}

    def encode(self, ops):
        for op in ops:
            self._encode_op(op)

    def _encode_op(self, op):
        name = op.name
        res = op.results[0] if op.results else None

        if name == "tt.get_program_id":
            self.vals[res] = self.pid

        elif name == "arith.constant":
            pass

        elif name == "arith.muli":
            a = self._get(op.operands[0])
            b = self._get(op.operands[1])
            if a is not None and b is not None:
                self.vals[res] = a * b

        elif name == "arith.addi":
            if op.is_tile:
                a_min = self._get(op.operands[0] + "_min")
                a_max = self._get(op.operands[0] + "_max")
                b_min = self._get(op.operands[1] + "_min")
                b_max = self._get(op.operands[1] + "_max")
                if all(x is not None for x in [a_min, a_max, b_min, b_max]):
                    self.vals[res + "_min"] = a_min + b_min
                    self.vals[res + "_max"] = a_max + b_max
            else:
                a = self._get(op.operands[0])
                b = self._get(op.operands[1])
                if a is not None and b is not None:
                    self.vals[res] = a + b

        elif name == "tt.addptr":
            if op.is_tile:
                ptr_min = self._get(op.operands[0] + "_min")
                ptr_max = self._get(op.operands[0] + "_max")
                off_min = self._get(op.operands[1] + "_min")
                off_max = self._get(op.operands[1] + "_max")
                if all(x is not None for x in [ptr_min, ptr_max, off_min, off_max]):
                    self.vals[res + "_min"] = ptr_min + off_min
                    self.vals[res + "_max"] = ptr_max + off_max
            else:
                ptr = self._get(op.operands[0])
                off = self._get(op.operands[1])
                if ptr is not None and off is not None:
                    self.vals[res] = ptr + off

        elif name == "tt.make_range":
            self.vals[res + "_min"] = BitVecVal(0, 32)
            self.vals[res + "_max"] = BitVecVal(self.block_size - 1, 32)

        elif name == "tt.splat":
            val = self._get(op.operands[0])
            if val is not None:
                self.vals[res + "_min"] = val
                self.vals[res + "_max"] = val
            else:
                val_min = self._get(op.operands[0] + "_min")
                val_max = self._get(op.operands[0] + "_max")
                if val_min is not None:
                    self.vals[res + "_min"] = val_min
                    self.vals[res + "_max"] = val_max

        elif name == "tt.load":
            if op.operands:
                ptr_max = self._get(op.operands[0] + "_max")
                if ptr_max is not None:
                    self.load_ptr_max = ptr_max

    def _get(self, name):
        return self.vals.get(name, None)

    def check(self):
        if self.load_ptr_max is None:
            return {"safe": True, "reason": "no load found"}

        self.solver.push()
        self.solver.add(self.load_ptr_max >= self.buffer_size)
        result = self.solver.check()
        

        if result == sat:
            m = self.solver.model()
            self.solver.pop()
            pid_val = m[self.pid].as_long()
            return {
                "safe": False,
                "pid": pid_val,
                "offset_max": pid_val * self.block_size + self.block_size - 1
            }
        self.solver.pop()
        if result == unsat:
            return {"safe": True}
   
        for pid_val in range(32):
            self.solver.push()
            self.solver.add(self.pid == pid_val)
            self.solver.add(self.load_ptr_max >= self.buffer_size)
            r = self.solver.check()
            self.solver.pop()
            if r == sat:
                return {
                    "safe": False,
                    "pid": pid_val,
                    "offset_max": pid_val * self.block_size + self.block_size - 1
                }
        return {"safe": True, "reason": "unknown — checked first 32 pids"}


if __name__ == "__main__":

    ops = [
        TaggedOp("tt.get_program_id", ["%0"], [],              "i32",                     False, 0),
        TaggedOp("arith.muli",        ["%1"], ["%0", "%c128"], "i32",                     False, 0),
        TaggedOp("tt.make_range",     ["%2"], [],              "tensor<128xi32>",          True,  128),
        TaggedOp("tt.splat",          ["%3"], ["%1"],          "tensor<128xi32>",          True,  128),
        TaggedOp("arith.addi",        ["%4"], ["%3", "%2"],    "tensor<128xi32>",          True,  128),
        TaggedOp("tt.splat",          ["%7"], ["%arg0"],       "tensor<128x!tt.ptr<f32>>", True,  128),
        TaggedOp("tt.addptr",         ["%8"], ["%7", "%4"],    "tensor<128x!tt.ptr<f32>>", True,  128),
        TaggedOp("tt.load",           ["%9"], ["%8"],          "tensor<128xf32>",          True,  128),
    ]

    print("=== TEST 1: safe (512 elements, 4 blocks) ===")
    enc = Encoder(block_size=128, buffer_size=512, grid_size=4)
    enc.vals["%arg0"] = BitVecVal(0, 32)
    enc.vals["%c128"] = BitVecVal(128, 32)
    enc.encode(ops)
    print(enc.check())

    print()
    print("=== TEST 2: buggy (500 elements, 4 blocks) ===")
    enc2 = Encoder(block_size=128, buffer_size=500, grid_size=4)
    enc2.vals["%arg0"] = BitVecVal(0, 32)
    enc2.vals["%c128"] = BitVecVal(128, 32)
    enc2.encode(ops)
    print(enc2.check())
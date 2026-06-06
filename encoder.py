from z3 import *
from abstractor import TaggedOp
import re


class Encoder:

    def __init__(self, block_size, buffer_size, grid_size):
        self.unsupported_ops = set()
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.solver = Solver()
        self.load_ptr_max = None
        self.all_load_maxima = []
        self.pid = BitVec('pid', 32)
        self.pid1 = BitVec('pid1', 32)
        self._pid_count = 0
        self.solver.add(self.pid >= 0)
        self.solver.add(self.pid < grid_size)
        self.solver.add(self.pid1 >= 0)
        self.solver.add(self.pid1 < grid_size)
        self.vals = {}

    def encode(self, ops):
        for op in ops:
            self._encode_op(op)

    def _encode_op(self, op):
        name = op.name
        res = op.results[0] if op.results else None

        if name == "tt.get_program_id":
            if self._pid_count == 0:
                self.vals[res] = self.pid
            else:
                self.vals[res] = self.pid1
            self._pid_count += 1

        elif name == "arith.constant":
            pass

        elif name == "arith.muli":
            if op.is_tile:
                a_min = self._get(op.operands[0] + "_min")
                a_max = self._get(op.operands[0] + "_max")
                b_min = self._get(op.operands[1] + "_min")
                b_max = self._get(op.operands[1] + "_max")
                if all(x is not None for x in [a_min, a_max, b_min, b_max]):
                    self.vals[res + "_min"] = a_min * b_min
                    self.vals[res + "_max"] = a_max * b_max
            else:
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
                elif ptr is not None:
                    self.vals[res] = ptr

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

        elif name == "tt.expand_dims":
            val_min = self._get(op.operands[0] + "_min")
            val_max = self._get(op.operands[0] + "_max")
            if val_min is not None:
                self.vals[res + "_min"] = val_min
                self.vals[res + "_max"] = val_max
            else:
                val = self._get(op.operands[0])
                if val is not None:
                    self.vals[res + "_min"] = val
                    self.vals[res + "_max"] = val

        elif name == "tt.broadcast":
            val_min = self._get(op.operands[0] + "_min")
            val_max = self._get(op.operands[0] + "_max")
            if val_min is not None:
                self.vals[res + "_min"] = val_min
                self.vals[res + "_max"] = val_max

        elif name == "tt.load":
            if op.operands:
                ptr_max = self._get(op.operands[0] + "_max")
                if ptr_max is not None and self._contains_pid(ptr_max):
                    self.all_load_maxima.append(ptr_max)

        else:
            self.unsupported_ops.add(name)

    def _get(self, name):
        return self.vals.get(name, None)

    def _contains_pid(self, expr):
        if expr is None:
            return False
        return 'pid' in str(expr)

    def encode_loop(self, loop_line: str, body_ops: list):
        m = re.match(r'SCFFOR (\S+) (\S+) (\S+) (\S+) ITERARGS (.+)', loop_line)
        if not m:
            return
        k, start, stop, step, iter_args = m.groups()

        start_val = self._get(start)
        if start_val is None:
            start_val = BitVecVal(0, 32)

        stop_val = self._get(stop)
        if stop_val is None:
            stop_val = BitVecVal(512, 32)

        step_val = self._get(step)
        if step_val is None:
            step_val = BitVecVal(32, 32)

        try:
            if hasattr(stop_val, 'as_long') and hasattr(start_val, 'as_long') and hasattr(step_val, 'as_long'):
                num_iters = (stop_val.as_long() - start_val.as_long()) // step_val.as_long()
                num_iters = BitVecVal(num_iters, 32)
            else:
                num_iters = (stop_val - start_val) / step_val
        except:
            num_iters = (stop_val - start_val) / step_val

        iter_map = {}
        for arg_pair in iter_args.split(','):
            arg_pair = arg_pair.strip()
            m2 = re.match(r'(%\w+)\s*=\s*(%\w+)', arg_pair)
            if m2:
                loop_var, init_val = m2.groups()
                iter_map[loop_var] = init_val
                init_min = self._get(init_val + "_min")
                init_max = self._get(init_val + "_max")
                if init_min is not None:
                    self.vals[loop_var + "_min"] = init_min
                    self.vals[loop_var + "_max"] = init_max

        for op in body_ops:
            self._encode_op(op)

        for op in body_ops:
            if op.name == "tt.addptr" and op.is_tile:
                ptr_operand = op.operands[0]
                if ptr_operand in iter_map:
                    init_val = iter_map[ptr_operand]
                    off_max = self._get(op.operands[1] + "_max")
                    init_max = self._get(init_val + "_max")
                    if off_max is not None and init_max is not None:
                        final_max = init_max + num_iters * off_max
                        self.vals[ptr_operand + "_max"] = final_max

        for op in body_ops:
            if op.name == "tt.load" and op.operands:
                ptr_max = self._get(op.operands[0] + "_max")
                if ptr_max is not None and self._contains_pid(ptr_max):
                    self.all_load_maxima.append(ptr_max)

    def check(self):
        candidates = self.all_load_maxima[:]
        if self.load_ptr_max is not None:
            candidates.append(self.load_ptr_max)

        if not candidates:
            return {"safe": True, "reason": "no load found"}

        for ptr_max in candidates:
            self.solver.push()
            self.solver.add(ptr_max >= self.buffer_size)
            result = self.solver.check()

            if result == sat:
                m = self.solver.model()
                self.solver.pop()
                pid_val = m[self.pid].as_long()
                try:
                    offset_int = m.eval(ptr_max).as_long()
                except:
                    offset_int = pid_val * self.block_size + self.block_size - 1
                return {
                    "safe": False,
                    "pid": pid_val,
                    "offset_max": offset_int
                }

            self.solver.pop()

            if result == unsat:
                continue

            # unknown — try concrete pid values
            for pid_val in range(32):
                self.solver.push()
                self.solver.add(self.pid == pid_val)
                self.solver.add(ptr_max >= self.buffer_size)
                r = self.solver.check()
                self.solver.pop()
                if r == sat:
                    return {
                        "safe": False,
                        "pid": pid_val,
                        "offset_max": pid_val * self.block_size + self.block_size - 1
                    }

        return {"safe": True}


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
import sys
import re
from lark import Lark
from parser import grammar, MLIRTransformer
from abstractor import tag_all
from encoder import Encoder
from preprocessor import preprocess
from z3 import BitVecVal
import math 


def parse_ops(lines: list) -> list:
    lark_parser = Lark(grammar, start="operation", parser="earley")
    transformer = MLIRTransformer()
    ops = []
    for line in lines:
        if line.startswith('SCFFOR'):
            continue
        try:
            tree = lark_parser.parse(line)
            op = transformer.transform(tree)
            ops.append(op)
        except Exception:
            pass
    return ops


def extract_args(path: str) -> list:
    with open(path) as f:
        for line in f:
            if 'tt.func' in line or 'func.func' in line:
                return re.findall(r'%(\w+)\s*:', line)
    return []


def main():
    if len(sys.argv) < 4:
        print("usage: python3 main.py <file.ttir> <buffer_size> <grid_size> [stride]")
        sys.exit(1)

    mlir_file   = sys.argv[1]
    buffer_size = int(sys.argv[2])
    grid_size   = int(sys.argv[3])
    stride = None
    stride_overrides = {}

    grid1 = None 
    for arg in sys.argv[4:]:
        if '=' in arg:
            k, v = arg.split('=', 1)
            if k == 'grid1': 
                grid1 = int(v)
            else: 
                stride_overrides[k] = int(v)
        else:
            stride = int(arg)


    # preprocess custom form → generic form
    lines = preprocess(mlir_file)
    print(f"preprocessed {len(lines)} ops")

    # find loop line and index before parsing
    loop_line = None
    loop_idx  = None
    for i, line in enumerate(lines):
        if line.startswith('SCFFOR'):
            loop_line = line
            loop_idx  = i
            break

    # parse into Op objects
    ops = parse_ops(lines)
    print(f"parsed {len(ops)} ops")

    # tag scalar vs tile
    tagged = tag_all(ops)

    # find block size from tt.make_range
    block_size = 128
    for op in tagged:
        if op.name == "tt.make_range" and op.is_tile and op.tile_size > 0:
            block_size = op.tile_size
            break

    print(f"block_size={block_size}  buffer_size={buffer_size}  grid_size={grid_size}  stride={stride}")

    # encode and check
    enc = Encoder(block_size=block_size,
                  buffer_size=buffer_size,
                  grid_size=grid_size, grid1_size = grid1)

    # seed function arguments
    args = extract_args(mlir_file)
    matrix_dim = int(math.sqrt(buffer_size)) if stride else None 

    for arg in args:
        arg_name = f'%{arg}'
        if 'ptr' in arg.lower():
            # base pointers always 0
            enc.vals[arg_name] = BitVecVal(0, 32)
        elif arg in stride_overrides: 
            enc.vals[arg_name] = BitVecVal(stride_overrides[arg], 32)
        elif stride is not None and 'stride' in arg.lower():
            # use provided stride for all stride args
            enc.vals[arg_name] = BitVecVal(stride, 32)
        elif stride is not None and arg.upper() in ['M', 'N', 'K']:
            # matrix dimension args
            enc.vals[arg_name] = BitVecVal(stride, 32)
        else:
            enc.vals[arg_name] = BitVecVal(0, 32)
    
    # for matmul kernels with implicit stride_ak=1
    if stride is not None:
        enc.vals['%stride_ak'] = BitVecVal(1, 32)
        enc.vals['%stride_bn'] = BitVecVal(1, 32)
        enc.vals['%stride_cn'] = BitVecVal(1, 32)

    # seed constants from arith.constant result names
    for op in tagged:
        if op.name == "arith.constant" and op.results:
            m = re.search(r'(\d+)', op.results[0])
            if m:
                enc.vals[op.results[0]] = BitVecVal(int(m.group(1)), 32)

    # seed dense tensor constants
    with open(mlir_file) as f:
        for line in f:
            m = re.match(r'\s*(%\w+)\s*=\s*arith\.constant\s+dense<(\d+)>\s*:', line)
            if m:
                name, val = m.groups()
                enc.vals[name + "_min"] = BitVecVal(int(val), 32)
                enc.vals[name + "_max"] = BitVecVal(int(val), 32)
    
    # seed unrolled loop variables as 0
    for line in lines:
        if '(%i)' in line or '(%i,' in line:
            enc.vals['%i'] = BitVecVal(0, 32)
            break

    # encode — split at loop boundary if present
    if loop_line is None:
        enc.encode(tagged)
    else:
        pre_count = sum(1 for l in lines[:loop_idx] if not l.startswith('SCFFOR'))
        enc.encode(tagged[:pre_count])
        enc.encode_loop(loop_line, tagged[pre_count:])

    if enc.unsupported_ops:
        print(f"WARNING: unsupported ops skipped: {enc.unsupported_ops}")
        print(f"         result may be incomplete for these op types")

    result = enc.check()

    print()
    if result.get("safe"):
        if enc.unsupported_ops: 
            print("INCOMPLETE: could not verify — unsupported ops present")
            print(f"  skipped: {enc.unsupported_ops}")
        else: 
            print("SAFE: no out-of-bounds access possible for any program id")
    else:
        print("BUG FOUND:")
        print(f"  pid        = {result['pid']}")
        print(f"  offset_max = {result['offset_max']}")
        print(f"  buffer_size= {buffer_size}")
        print(f"  overflow by= {result['offset_max'] - buffer_size + 1} elements")


if __name__ == "__main__":
    main()
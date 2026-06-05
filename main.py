import sys
from lark import Lark
from parser import grammar, MLIRTransformer
from abstractor import tag_all
from encoder import Encoder
from preprocessor import preprocess
from z3 import BitVecVal


def parse_ops(lines: list) -> list:
    lark_parser = Lark(grammar, start="operation", parser="earley")
    transformer = MLIRTransformer()
    ops = []
    for line in lines:
        try:
            tree = lark_parser.parse(line)
            op = transformer.transform(tree)
            ops.append(op)
        except Exception:
            pass
    return ops


def main():
    if len(sys.argv) < 4:
        print("usage: python3 main.py <file.ttir> <buffer_size> <grid_size>")
        print("example: python3 main.py kernel.ttir 512 4")
        sys.exit(1)

    mlir_file   = sys.argv[1]
    buffer_size = int(sys.argv[2])
    grid_size   = int(sys.argv[3])

    # preprocess custom form → generic form
    lines = preprocess(mlir_file)
    print(f"preprocessed {len(lines)} ops")

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

    print(f"block_size={block_size}  buffer_size={buffer_size}  grid_size={grid_size}")

    # encode and check
    enc = Encoder(block_size=block_size,
                  buffer_size=buffer_size,
                  grid_size=grid_size)

    # seed all constants from arith.constant ops
    for op in tagged:
        if op.name == "arith.constant" and op.results:
            # extract the value from result name e.g. %c128_i32 → 128
            import re
            m = re.search(r'(\d+)', op.results[0])
            if m:
                enc.vals[op.results[0]] = BitVecVal(int(m.group(1)), 32)

    # seed function pointer arguments as base address 0
    for op in tagged:
        for operand in op.operands:
            if operand.startswith("%x_ptr") or \
               operand.startswith("%y_ptr") or \
               operand.startswith("%out_ptr") or \
               operand.startswith("%arg"):
                if operand not in enc.vals:
                    enc.vals[operand] = BitVecVal(0, 32)

    enc.encode(tagged)
    result = enc.check()

    print()
    if result.get("safe"):
        print("SAFE: no out-of-bounds access possible for any program id")
    else:
        print("BUG FOUND:")
        print(f"  pid        = {result['pid']}")
        print(f"  offset_max = {result['offset_max']}")
        print(f"  buffer_size= {buffer_size}")
        print(f"  overflow by= {result['offset_max'] - buffer_size + 1} elements")


if __name__ == "__main__":
    main()
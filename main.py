import sys
import re
from lark import Lark
from parser import grammar, MLIRTransformer, Op
from abstractor import tag_all
from encoder import Encoder
from z3 import BitVecVal


def extract_ops(mlir_text: str) -> list:
    # grab everything between the outermost { }
    match = re.search(r'\{(.*)\}', mlir_text, re.DOTALL)
    if not match:
        print("ERROR: could not find function body in .mlir file")
        sys.exit(1)
    body = match.group(1).strip()
    # split into individual lines, skip empty lines
    lines = [l.strip() for l in body.splitlines() if l.strip()]
    return lines


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
            # skip lines that don't match generic op form
            # (function signatures, block labels, tt.return, etc.)
            pass
    return ops


def main():
    if len(sys.argv) < 4:
        print("usage: python3 main.py <file.mlir> <buffer_size> <grid_size>")
        print("example: python3 main.py kernel.mlir 512 4")
        sys.exit(1)

    mlir_file   = sys.argv[1]
    buffer_size = int(sys.argv[2])
    grid_size   = int(sys.argv[3])

    # read file
    with open(mlir_file) as f:
        mlir_text = f.read()

    # extract op lines from function body
    lines = extract_ops(mlir_text)
    print(f"found {len(lines)} lines in kernel body")

    # parse into Op objects
    ops = parse_ops(lines)
    print(f"parsed {len(ops)} ops")

    # tag scalar vs tile
    tagged = tag_all(ops)

    # find block size from tt.make_range
    block_size = 128  # default
    for op in tagged:
        if op.is_tile and op.tile_size > 0:
            block_size = op.tile_size
            break
    print(f"block_size={block_size}  buffer_size={buffer_size}  grid_size={grid_size}")

    # encode and check
    enc = Encoder(block_size=block_size,
                  buffer_size=buffer_size,
                  grid_size=grid_size)

    # seed function arguments as symbolic base pointer at 0
    for op in tagged:
        for operand in op.operands:
            if operand.startswith("%arg"):
                enc.vals[operand] = BitVecVal(0, 32)

    # seed any constants that look like the block size
    for op in tagged:
        if op.name == "arith.constant":
            # name the constant after its result
            enc.vals[op.results[0]] = BitVecVal(block_size, 32)

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
from parser import Op
from dataclasses import dataclass

@dataclass
class TaggedOp:
    name: str
    results: list
    operands: list
    result_type: str
    is_tile: bool
    tile_size: int   # 0 if scalar, 128 if tensor<128x...>

def tag(op: Op) -> TaggedOp:
    is_tile = op.result_type.startswith("tensor<")
    tile_size = 0
    if is_tile:
        # extract the 128 from tensor<128x...>
        tile_size = int(op.result_type.split("<")[1].split("x")[0])
    return TaggedOp(
        name=op.name,
        results=op.results,
        operands=op.operands,
        result_type=op.result_type,
        is_tile=is_tile,
        tile_size=tile_size
    )

def tag_all(ops: list) -> list:
    return [tag(op) for op in ops]


# --- test it ---
if __name__ == "__main__":
    from parser import Op

    test_ops = [
        Op("tt.get_program_id", ["%0"], [], "i32"),
        Op("arith.muli",        ["%1"], ["%0", "%c128"], "i32"),
        Op("tt.addptr",         ["%2"], ["%arg0", "%1"], "!tt.ptr<f32>"),
        Op("tt.load",           ["%3"], ["%ptr"], "tensor<128xf32>"),
    ]

    tagged = tag_all(test_ops)
    for t in tagged:
        print(f"{t.name:<25} is_tile={t.is_tile}  tile_size={t.tile_size}  type={t.result_type}")
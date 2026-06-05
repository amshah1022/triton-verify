import re
import sys


def strip_location(line: str) -> str:
    line = re.sub(r'\s+loc\(#\w+\)', '', line)
    line = re.sub(r'\s+loc\(.*', '', line)
    return line.strip()


def custom_to_generic(line: str) -> str:
    line = strip_location(line)
    if not line:
        return None

    for prefix in ['#', 'module', '}', 'tt.func', 'tt.return', 'scf.yield', '//']:
        if line.startswith(prefix):
            return None

    # arith.constant
    m = re.match(r'(%\w+)\s*=\s*arith\.constant\s+(\S+)\s*:\s*(\S+)', line)
    if m:
        name, val, typ = m.groups()
        return f'{name} = "arith.constant"() {{value = {val} : {typ}}} : () -> {typ}'

    # tt.get_program_id
    m = re.match(r'(%\w+)\s*=\s*tt\.get_program_id\s+\w+\s*:\s*(\S+)', line)
    if m:
        name, typ = m.groups()
        return f'{name} = "tt.get_program_id"() {{axis = 0 : i32}} : () -> {typ}'

    # tt.make_range
    m = re.match(r'(%\w+)\s*=\s*tt\.make_range\s+\{end\s*=\s*(\d+)\s*:\s*i32,\s*start\s*=\s*(\d+)\s*:\s*i32\}\s*:\s*(.+)', line)
    if m:
        name, end, start, typ = m.groups()
        return f'{name} = "tt.make_range"() {{end = {end} : i32, start = {start} : i32}} : () -> {typ}'

    # tt.splat
    m = re.match(r'(%\w+)\s*=\s*tt\.splat\s+(%\w+)\s*:\s*(.+?)\s*->\s*(.+)', line)
    if m:
        name, operand, in_type, out_type = m.groups()
        return f'{name} = "tt.splat"({operand}) : ({in_type}) -> {out_type}'
    # arith.cmpi (has predicate: slt, sgt, eq, etc.)
    m = re.match(r'(%\w+)\s*=\s*arith\.cmpi\s+\w+,\s*(%\w+),\s*(%\w+)\s*:\s*(.+)', line)
    if m:
        name, a, b, typ = m.groups()
        if typ.startswith('tensor<'):
            size = typ.split('<')[1].split('x')[0]
            ret_type = f'tensor<{size}xi1>'
        else:
            ret_type = 'i1'
        return f'{name} = "arith.cmpi"({a}, {b}) : ({typ}, {typ}) -> {ret_type}'
    # arith binary ops
    m = re.match(r'(%\w+)\s*=\s*(arith\.\w+)\s+(%\w+),\s*(%\w+)\s*:\s*(.+)', line)
    if m:
        name, op, a, b, typ = m.groups()
        if op == 'arith.cmpi':
            if typ.startswith('tensor<'):
                size = typ.split('<')[1].split('x')[0]
                ret_type = f'tensor<{size}xi1>'
            else:
                ret_type = 'i1'
            return f'{name} = "{op}"({a}, {b}) : ({typ}, {typ}) -> {ret_type}'
        return f'{name} = "{op}"({a}, {b}) : ({typ}, {typ}) -> {typ}'

    # tt.addptr
    m = re.match(r'(%\w+)\s*=\s*tt\.addptr\s+(%\w+),\s*(%\w+)\s*:\s*(.+?),\s*(.+)', line)
    if m:
        name, ptr, off, ptr_type, off_type = m.groups()
        return f'{name} = "tt.addptr"({ptr}, {off}) : ({ptr_type}, {off_type}) -> {ptr_type}'

    # tt.load with mask and optional other value
    m = re.match(r'(%\w+)\s*=\s*tt\.load\s+(%\w+),\s*(%\w+)(?:,\s*(%\w+))?\s*:\s*(.+)', line)
    if m:
        name, ptr, mask, other, ptr_type = m.groups()
        result_type = re.sub(r'!tt\.ptr<(.+?)>', r'\1', ptr_type)
        if ptr_type.startswith('tensor<'):
            size = ptr_type.split('<')[1].split('x')[0]
            mask_type = f'tensor<{size}xi1>'
        else:
            mask_type = 'i1'
        return f'{name} = "tt.load"({ptr}, {mask}) : ({ptr_type}, {mask_type}) -> {result_type}'

    # tt.load without mask
    m = re.match(r'(%\w+)\s*=\s*tt\.load\s+(%\w+)\s*:\s*(.+)', line)
    if m:
        name, ptr, ptr_type = m.groups()
        result_type = re.sub(r'!tt\.ptr<(.+?)>', r'\1', ptr_type)
        return f'{name} = "tt.load"({ptr}) : ({ptr_type}) -> {result_type}'

    # tt.expand_dims
    m = re.match(r'(%\w+)\s*=\s*tt\.expand_dims\s+(%\w+)\s*\{[^}]*\}\s*:\s*(.+?)\s*->\s*(.+)', line)
    if m:
        name, operand, in_type, out_type = m.groups()
        return f'{name} = "tt.expand_dims"({operand}) : ({in_type}) -> {out_type}'

    # tt.broadcast
    m = re.match(r'(%\w+)\s*=\s*tt\.broadcast\s+(%\w+)\s*:\s*(.+?)\s*->\s*(.+)', line)
    if m:
        name, operand, in_type, out_type = m.groups()
        return f'{name} = "tt.broadcast"({operand}) : ({in_type}) -> {out_type}'

    # scf.for
    m = re.match(r'(%[\w:]+)\s*=\s*scf\.for\s+(%\w+)\s*=\s*(%\w+)\s*to\s*(%\w+)\s*step\s*(%\w+)\s*iter_args\((.+?)\)\s*->', line)
    if m:
        result, k, start, stop, step, iter_args = m.groups()
        return f'SCFFOR {k} {start} {stop} {step} ITERARGS {iter_args}'

    return None


def preprocess(path: str) -> list:
    with open(path) as f:
        lines = f.readlines()
    result = []
    for line in lines:
        converted = custom_to_generic(line)
        if converted:
            result.append(converted)
    return result


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "real_vector_add.ttir"
    lines = preprocess(path)
    for line in lines:
        print(line)
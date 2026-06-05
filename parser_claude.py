"""
MLIR Lark Parser
================
Parses MLIR's human-readable textual form into a Lark parse tree.
Designed as a foundation for Triton kernel formal verification with z3.

Grammar follows the MLIR Language Reference (EBNF):
  https://mlir.llvm.org/docs/LangRef/
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import Any, List, Optional

from lark import Lark, Transformer, v_args

# ---------------------------------------------------------------------------
# Grammar
# ---------------------------------------------------------------------------

MLIR_GRAMMAR = r"""
    // -----------------------------------------------------------------------
    // Top-level
    // -----------------------------------------------------------------------
    start : toplevel_item*

    toplevel_item : operation
                  | attribute_alias_def
                  | type_alias_def

    // -----------------------------------------------------------------------
    // Identifiers & literals
    // -----------------------------------------------------------------------
    BARE_ID      : /[a-zA-Z_][a-zA-Z0-9_$.]*/
    CARET_ID     : /\^[a-zA-Z_$][a-zA-Z0-9_$.]*/ | /\^\d+/
    VALUE_ID     : /%-?[a-zA-Z_$][a-zA-Z0-9_$.]*/ | /%-?\d+/
    SYMBOL_ID    : /@[a-zA-Z_$][a-zA-Z0-9_$.]*/ | /@\d+/ | /@"[^"]*"/

    // string content between the outer quotes (escape sequences preserved)
    STRING_INNER : /(?:[^"\\]|\\.)+/
    string_literal : "\"" STRING_INNER? "\""

    DECIMAL_LIT : /[0-9]+/
    HEX_LIT     : /0x[0-9a-fA-F]+/
    FLOAT_LIT   : /[-+]?[0-9]+\.[0-9]*(?:[eE][-+]?[0-9]+)?/

    integer_literal : HEX_LIT | DECIMAL_LIT
    float_literal   : FLOAT_LIT

    // -----------------------------------------------------------------------
    // Top-level aliases
    // -----------------------------------------------------------------------
    attribute_alias_def : "#" BARE_ID "=" attribute_value
    type_alias_def      : "!" BARE_ID "=" type

    attribute_alias : "#" BARE_ID
    type_alias      : "!" BARE_ID

    // -----------------------------------------------------------------------
    // Symbol references  (@foo  or  @foo::@bar::@baz)
    // -----------------------------------------------------------------------
    symbol_ref_id : SYMBOL_ID ("::" symbol_ref_id)?

    // -----------------------------------------------------------------------
    // Value uses
    // -----------------------------------------------------------------------
    value_use      : VALUE_ID ("#" DECIMAL_LIT)?
    value_use_list : value_use ("," value_use)*

    // -----------------------------------------------------------------------
    // Types
    // -----------------------------------------------------------------------
    type : type_alias
         | dialect_type
         | builtin_type

    type_list_parens : "(" ")"
                     | "(" type ("," type)* ")"

    function_type : type_list_parens "->" (type | type_list_parens)
                  | type               "->" (type | type_list_parens)

    // -- Dialect types  (!namespace<...>  or  !namespace.lead_ident<...>?)
    dialect_type : "!" BARE_ID "." /[A-Za-z][A-Za-z0-9._]*/ dialect_type_body?
                 | "!" BARE_ID dialect_type_body

    dialect_type_body : "<" dialect_type_contents* ">"
    dialect_type_contents : dialect_type_body
                           | "(" dialect_type_contents* ")"
                           | "[" dialect_type_contents* "]"
                           | "{" dialect_type_contents* "}"
                           | /[^\[\]<>()\{\}"]+/

    // -- Builtin types
    builtin_type : integer_type
                 | float_type
                 | index_type
                 | none_type
                 | memref_type
                 | tensor_type
                 | vector_type
                 | tuple_type
                 | function_type
                 | complex_type
                 | opaque_type
                 | token_type

    integer_type : /[su]?i[1-9][0-9]*/
    float_type   : "f16" | "bf16" | "f32" | "f64" | "f80" | "f128" | "f8E4M3FN" | "f8E5M2" | "tf32"
    index_type   : "index"
    none_type    : "none"
    token_type   : "!builtin.token" | "token"

    // tensor<dim x dim x ... x type>  or  tensor<*xtype>
    tensor_type : "tensor" "<" tensor_dim_list type ("," attribute_value)? ">"
    tensor_dim_list : (tensor_dim "x")*
    tensor_dim  : DECIMAL_LIT | "?"

    // memref<dim x ... x type, affine_maps, memspace>
    memref_type : "memref" "<" memref_dim_list type memref_opts? ">"
    memref_dim_list : (memref_dim "x")*
    memref_dim  : DECIMAL_LIT | "?"
    memref_opts : "," attribute_value ("," attribute_value)*

    // vector<dim x ... x type>  (supports scalable dims in [])
    vector_type : "vector" "<" vector_dim_list type ">"
    vector_dim_list : (vector_dim "x")*
    vector_dim  : DECIMAL_LIT | "[" DECIMAL_LIT "]"

    // tuple<type, type, ...>
    tuple_type : "tuple" "<" (type ("," type)*)? ">"

    // complex<type>
    complex_type : "complex" "<" type ">"

    // opaque builtin type  (last-resort fallback inside builtins)
    opaque_type : BARE_ID "<" /[^>]*/ ">"

    // -----------------------------------------------------------------------
    // Attributes
    // -----------------------------------------------------------------------
    attribute_value : attribute_alias
                    | dialect_attribute
                    | builtin_attribute

    // -- Dialect attributes  (#namespace<...>  or  #namespace.lead_ident<...>?)
    dialect_attribute : "#" BARE_ID "." /[A-Za-z][A-Za-z0-9._]*/ dialect_attr_body?
                      | "#" BARE_ID dialect_attr_body

    dialect_attr_body : "<" dialect_attr_contents* ">"
    dialect_attr_contents : dialect_attr_body
                           | "(" dialect_attr_contents* ")"
                           | "[" dialect_attr_contents* "]"
                           | "{" dialect_attr_contents* "}"
                           | /[^\[\]<>()\{\}"]+/

    // -- Builtin attributes
    builtin_attribute : integer_attr
                      | float_attr
                      | string_attr
                      | type_attr
                      | array_attr
                      | dict_attr
                      | unit_attr
                      | bool_attr
                      | dense_attr
                      | affine_map_attr
                      | affine_set_attr
                      | symref_attr
                      | opaque_attr

    integer_attr : (integer_literal | DECIMAL_LIT) (":" (integer_type | index_type))?
    float_attr   : float_literal (":" float_type)?
    string_attr  : string_literal (":" type)?
    type_attr    : type
    array_attr   : "[" (attribute_value ("," attribute_value)*)? "]"
    dict_attr    : "{" (attr_entry ("," attr_entry)*)? "}"
    attr_entry   : (BARE_ID | string_literal) "=" attribute_value
    unit_attr    : "unit"
    bool_attr    : "true" | "false"
    dense_attr   : "dense" "<" dense_contents ">" ":" type
    dense_contents : dense_literal | type
    dense_literal  : "[" (dense_literal | attribute_value) ("," (dense_literal | attribute_value))* "]" | attribute_value
    affine_map_attr : "affine_map" "<" affine_map_comp ">"
    affine_set_attr : "affine_set" "<" affine_set_comp ">"
    symref_attr  : symbol_ref_id
    opaque_attr  : BARE_ID "<" /[^>]*/ ">"

    // -----------------------------------------------------------------------
    // Affine maps & sets (simplified – parse as opaque balanced text)
    // -----------------------------------------------------------------------
    affine_map_comp : affine_balanced+
    affine_set_comp : affine_balanced+
    affine_balanced : "(" affine_balanced* ")"
                    | "[" affine_balanced* "]"
                    | /[^\(\)\[\]<>]+/

    // -----------------------------------------------------------------------
    // Operations
    // -----------------------------------------------------------------------
    operation : op_result_list? (generic_operation | custom_operation) trailing_location?

    op_result_list : op_result ("," op_result)* "="
    op_result      : VALUE_ID (":" integer_literal)?

    generic_operation : string_literal "(" value_use_list? ")" successor_list? dictionary_properties? region_list? dictionary_attr? ":" function_type

    custom_operation : BARE_ID custom_op_body
    custom_op_body : custom_op_token*
    custom_op_token : value_use_list | type | attribute_value | region | successor_list | "(" custom_op_token* ")" | "[" custom_op_token* "]" | "<" custom_op_token* ">" | ":" | "," | "->" | "=" | "to" | "in" | "step" | "iter_args" | string_literal | DECIMAL_LIT | FLOAT_LIT | BARE_ID

    trailing_location : "loc" "(" location ")"
    location : "unknown"
             | string_literal ":" DECIMAL_LIT ":" DECIMAL_LIT
             | "fused" "[" location ("," location)* "]"
             | "fused" "<" attribute_value ">" "[" location ("," location)* "]"
             | "callsite" "(" location "at" location ")"
             | "name_loc" "(" string_literal "," location ")"

    successor_list : "[" successor ("," successor)* "]"
    successor      : CARET_ID (":" block_arg_list)?

    dictionary_properties : "<" dictionary_attr ">"
    dictionary_attr        : "{" (attr_entry ("," attr_entry)*)? "}"
    region_list : "(" region ("," region)* ")"

    // -----------------------------------------------------------------------
    // Regions & Blocks
    // -----------------------------------------------------------------------
    region      : "{" entry_block? block* "}"
    entry_block : operation+
    block       : block_label operation+
    block_label : CARET_ID block_arg_list? ":"

    block_arg_list : "(" (value_id_and_type ("," value_id_and_type)*)? ")"
    value_id_and_type : VALUE_ID ":" type

    // -----------------------------------------------------------------------
    // Ignored tokens
    // -----------------------------------------------------------------------
    COMMENT : /\/\/[^\n]*/
    %ignore COMMENT
    %ignore /\s+/
"""


# ---------------------------------------------------------------------------
# Parser instance (shared / cached)
# ---------------------------------------------------------------------------

_parser: Optional[Lark] = None


def _get_parser() -> Lark:
    global _parser
    if _parser is None:
        _parser = Lark(
            MLIR_GRAMMAR,
            parser="earley",
            ambiguity="resolve",
            propagate_positions=True,
        )
    return _parser


def parse_mlir(source: str) -> Any:
    """Parse an MLIR source string and return a raw Lark Tree."""
    return _get_parser().parse(source)


# ---------------------------------------------------------------------------
# IR dataclasses  (lightweight AST nodes)
# ---------------------------------------------------------------------------

@dataclass
class MLIRModule:
    items: List[Any] = field(default_factory=list)


@dataclass
class Operation:
    name: str
    results: List[str] = field(default_factory=list)
    operands: List[str] = field(default_factory=list)
    successors: List[str] = field(default_factory=list)
    properties: dict = field(default_factory=dict)
    attributes: dict = field(default_factory=dict)
    regions: List["Region"] = field(default_factory=list)
    result_types: List[str] = field(default_factory=list)
    operand_types: List[str] = field(default_factory=list)
    raw_body: Optional[Any] = None


@dataclass
class Region:
    blocks: List["Block"] = field(default_factory=list)


@dataclass
class Block:
    label: Optional[str]
    args: List[tuple]
    operations: List[Operation] = field(default_factory=list)


@dataclass
class AttributeAliasDef:
    name: str
    value: Any


@dataclass
class TypeAliasDef:
    name: str
    type_: Any


# ---------------------------------------------------------------------------
# Transformer  – converts Lark Tree → dataclasses
# ---------------------------------------------------------------------------

@v_args(inline=True)
class MLIRTransformer(Transformer):

    # -- Top level -----------------------------------------------------------

    def start(self, *items):
        return MLIRModule(items=list(items))

    def toplevel_item(self, item):
        return item

    def attribute_alias_def(self, name, value):
        return AttributeAliasDef(name=str(name), value=value)

    def type_alias_def(self, name, type_):
        return TypeAliasDef(name=str(name), type_=type_)

    # -- Operations ----------------------------------------------------------

    def operation(self, *args):
        op = Operation(name="<unknown>")
        for arg in args:
            if isinstance(arg, list) and all(isinstance(x, str) for x in arg):
                op.results = arg
            elif isinstance(arg, Operation):
                op.name         = arg.name
                op.operands     = arg.operands
                op.successors   = arg.successors
                op.properties   = arg.properties
                op.attributes   = arg.attributes
                op.regions      = arg.regions
                op.result_types = arg.result_types
                op.operand_types= arg.operand_types
                op.raw_body     = arg.raw_body
        return op

    def op_result_list(self, *results):
        return [str(r) for r in results]

    def op_result(self, value_id, count=None):
        return str(value_id)

    def generic_operation(self, op_name, *rest):
        op = Operation(name=op_name.strip('"'))
        for item in rest:
            if isinstance(item, _ValueUseList):
                op.operands = item.values
            elif isinstance(item, _SuccessorList):
                op.successors = item.successors
            elif isinstance(item, _DictAttr):
                op.attributes = item.entries
            elif isinstance(item, _DictProperties):
                op.properties = item.entries
            elif isinstance(item, _RegionList):
                op.regions = item.regions
            elif isinstance(item, _FunctionType):
                op.operand_types = item.inputs
                op.result_types  = item.outputs
        return op

    def custom_operation(self, op_name, body):
        op = Operation(name=str(op_name))
        op.raw_body = body
        return op

    def custom_op_body(self, *tokens):
        return list(tokens)

    def custom_op_token(self, *args):
        return args[0] if len(args) == 1 else list(args)

    def trailing_location(self, loc):
        return loc

    # -- Regions & Blocks ----------------------------------------------------

    def region(self, *args):
        blocks = []
        for a in args:
            if isinstance(a, Block):
                blocks.append(a)
            elif isinstance(a, list):
                blocks.append(Block(label=None, args=[], operations=a))
        return Region(blocks=blocks)

    def entry_block(self, *ops):
        return list(ops)

    def block(self, label, *ops):
        return Block(label=label[0], args=label[1], operations=list(ops))

    def block_label(self, caret_id, arg_list=None):
        return (str(caret_id), arg_list or [])

    def block_arg_list(self, *pairs):
        return list(pairs)

    def value_id_and_type(self, vid, type_):
        return (str(vid), type_)

    # -- Value uses ----------------------------------------------------------

    def value_use(self, vid, ordinal=None):
        s = str(vid)
        if ordinal is not None:
            s += f"#{ordinal}"
        return s

    def value_use_list(self, *uses):
        return _ValueUseList(list(uses))

    # -- Successors ----------------------------------------------------------

    def successor_list(self, *succs):
        return _SuccessorList([str(s[0]) for s in succs])

    def successor(self, caret_id, arg_list=None):
        return (str(caret_id), arg_list)

    # -- Properties / Attributes ---------------------------------------------

    def dictionary_properties(self, d):
        return _DictProperties(d.entries)

    def dictionary_attr(self, *entries):
        return _DictAttr({k: v for k, v in entries})

    def attr_entry(self, key, value):
        k = str(key) if not isinstance(key, str) else key
        return (k, value)

    def region_list(self, *regions):
        return _RegionList(list(regions))

    # -- Function type -------------------------------------------------------

    def function_type(self, inputs, outputs):
        in_list  = inputs  if isinstance(inputs,  list) else [inputs]
        out_list = outputs if isinstance(outputs, list) else [outputs]
        return _FunctionType(in_list, out_list)

    def type_list_parens(self, *types):
        return list(types)

    # -- Types ---------------------------------------------------------------

    def type(self, t):          return t
    def builtin_type(self, t):  return t
    def dialect_type(self, *a): return "dialect_type:" + "_".join(str(x) for x in a)
    def integer_type(self, t):  return str(t)
    def float_type(self, t):    return str(t)
    def index_type(self):       return "index"
    def none_type(self):        return "none"
    def token_type(self, t):    return str(t)
    def tensor_type(self, *a):  return "tensor<" + "x".join(str(x) for x in a) + ">"
    def memref_type(self, *a):  return "memref<" + "x".join(str(x) for x in a) + ">"
    def vector_type(self, *a):  return "vector<" + "x".join(str(x) for x in a) + ">"
    def tuple_type(self, *a):   return "tuple<" + ",".join(str(x) for x in a) + ">"
    def complex_type(self, t):  return f"complex<{t}>"
    def opaque_type(self, *a):  return "opaque:" + "_".join(str(x) for x in a)
    def type_alias(self, name): return f"!{name}"

    def tensor_dim_list(self, *d): return list(d)
    def tensor_dim(self, d):       return str(d)
    def memref_dim_list(self, *d): return list(d)
    def memref_dim(self, d):       return str(d)
    def memref_opts(self, *a):     return list(a)
    def vector_dim_list(self, *d): return list(d)
    def vector_dim(self, d):       return str(d)

    # -- Attributes ----------------------------------------------------------

    def attribute_value(self, a):  return a
    def builtin_attribute(self, a): return a
    def dialect_attribute(self, *a): return "dialect_attr:" + "_".join(str(x) for x in a)
    def attribute_alias(self, name): return f"#{name}"

    def integer_attr(self, val, type_=None):
        v = int(str(val), 0)
        return {"kind": "integer", "value": v, "type": type_}

    def float_attr(self, val, type_=None):
        v = float(str(val))
        return {"kind": "float", "value": v, "type": type_}

    def string_attr(self, s, type_=None):
        return {"kind": "string", "value": str(s), "type": type_}

    def type_attr(self, t):
        return {"kind": "type", "value": t}

    def array_attr(self, *items):
        return {"kind": "array", "value": list(items)}

    def dict_attr(self, *entries):
        return {"kind": "dict", "value": {k: v for k, v in entries}}

    def unit_attr(self):
        return {"kind": "unit"}

    def bool_attr(self, v):
        return {"kind": "bool", "value": str(v) == "true"}

    def dense_attr(self, contents, type_):
        return {"kind": "dense", "contents": contents, "type": type_}

    def dense_contents(self, c):  return c
    def dense_literal(self, *a):  return list(a)

    def affine_map_attr(self, comp):
        return {"kind": "affine_map", "value": comp}

    def affine_set_attr(self, comp):
        return {"kind": "affine_set", "value": comp}

    def affine_map_comp(self, *parts): return "".join(str(p) for p in parts)
    def affine_set_comp(self, *parts): return "".join(str(p) for p in parts)
    def affine_balanced(self, *parts): return "".join(str(p) for p in parts)

    def symref_attr(self, ref):
        return {"kind": "symref", "value": ref}

    def opaque_attr(self, *a):
        return {"kind": "opaque", "value": "_".join(str(x) for x in a)}

    # -- Literals ------------------------------------------------------------

    def string_literal(self, *parts):
        return "".join(str(p) for p in parts)

    def integer_literal(self, v): return int(str(v), 0)
    def float_literal(self, v):   return float(str(v))

    def symbol_ref_id(self, sym, nested=None):
        s = str(sym)
        if nested:
            s += "::" + str(nested)
        return s

    def attribute_alias_def(self, name, value):
        return AttributeAliasDef(name=str(name), value=value)

    def type_alias_def(self, name, type_):
        return TypeAliasDef(name=str(name), type_=type_)

    def location(self, *args): return args


# ---------------------------------------------------------------------------
# Private helper sentinel types
# ---------------------------------------------------------------------------

@dataclass
class _ValueUseList:
    values: List[str]

@dataclass
class _SuccessorList:
    successors: List[str]

@dataclass
class _DictAttr:
    entries: dict

@dataclass
class _DictProperties:
    entries: dict

@dataclass
class _RegionList:
    regions: List[Region]

@dataclass
class _FunctionType:
    inputs: List[Any]
    outputs: List[Any]


# ---------------------------------------------------------------------------
# Pretty-printer helper
# ---------------------------------------------------------------------------

def dump_tree(node, indent: int = 0) -> str:
    pad = "  " * indent
    if isinstance(node, MLIRModule):
        lines = [f"{pad}MLIRModule ({len(node.items)} items)"]
        for item in node.items:
            lines.append(dump_tree(item, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Operation):
        results = ", ".join(node.results) if node.results else "(none)"
        lines = [f"{pad}Op '{node.name}'  results={results}  operands={node.operands}"]
        for r in node.regions:
            lines.append(dump_tree(r, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Region):
        lines = [f"{pad}Region ({len(node.blocks)} blocks)"]
        for b in node.blocks:
            lines.append(dump_tree(b, indent + 1))
        return "\n".join(lines)
    if isinstance(node, Block):
        label = node.label or "<entry>"
        lines = [f"{pad}Block {label}  args={node.args}  ({len(node.operations)} ops)"]
        for op in node.operations:
            lines.append(dump_tree(op, indent + 1))
        return "\n".join(lines)
    return f"{pad}{node!r}"


# ---------------------------------------------------------------------------
# Execution Driver
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python3 parser.py <file.mlir>")
        sys.exit(1)

    print("=== Parsing MLIR ===\n")
    with open(sys.argv[1], "r") as f:
        src = f.read()

    raw_tree = parse_mlir(src)
    ir = MLIRTransformer().transform(raw_tree)
    print("=== Transformed IR AST ===\n")
    print(dump_tree(ir))
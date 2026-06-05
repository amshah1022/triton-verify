from lark import Lark, Transformer, v_args
from dataclasses import dataclass, field

grammar = (
    "program: operation+\n"
    "operation: (result_list \"=\")? OP_NAME"
    " \"(\" operand_list \")\""
    " attr_dict?"
    " \":\" func_type\n"
    "result_list: VALUE_ID (\",\" VALUE_ID)*\n"
    "operand_list: (VALUE_ID (\",\" VALUE_ID)*)?\n"
    "func_type: \"(\" type_list \")\" \"->\" type\n"
    "         | \"(\" type_list \")\" \"->\" \"(\" type_list \")\"\n"
    "type_list: (type (\",\" type)*)?\n"
    "type: tensor_type\n"
    "    | ptr_type\n"
    "    | int_type\n"
    "    | float_type\n"
    "tensor_type: \"tensor\" \"<\" INT \"x\" INT \"x\" type \">\"\n"
    "           | \"tensor\" \"<\" INT \"x\" type \">\"\n"
    "ptr_type: \"!\" \"tt.ptr\" \"<\" type (\",\" INT)? \">\"\n"
    "int_type: /i\\d+/\n"
    "float_type: /f\\d+/\n"
    "attr_dict: \"{\" attr_content* \"}\"\n"
    "attr_content: /[^{}]+/ | \"{\" attr_content* \"}\"\n"
    "OP_NAME: \"\\\"\" /[a-zA-Z0-9_.]+/ \"\\\"\"\n"
    "VALUE_ID: \"%\" /[a-zA-Z0-9_]+/\n"
    "INT: /\\d+/\n"
    "%ignore /\\s+/\n"
    "%ignore /\\/\\/.*/\n"
)
parser = Lark(grammar, start="operation", parser="earley")


@dataclass
class Op:
    name: str
    results: list
    operands: list
    result_type: str

@v_args(inline=True)
class MLIRTransformer(Transformer):

    def operation(self, *items):
        name = None
        results = []
        operands = []
        result_type = None

        for item in items:
            if isinstance(item, tuple) and item[0] == "results":
                results = item[1]
            elif isinstance(item, tuple) and item[0] == "operands":
                operands = item[1]
            elif isinstance(item, tuple) and item[0] == "type":
                result_type = item[1]
            elif isinstance(item, str) and item.startswith('"'):
                name = item.strip('"')

        return Op(name=name, results=results,
                  operands=operands, result_type=result_type)

    def result_list(self, *items):
        return ("results", [str(i) for i in items])

    def operand_list(self, *items):
        return ("operands", [str(i) for i in items])

    def func_type(self, *items):
        # last item is the return type
        return ("type", str(items[-1]))

    def type(self, item):
        return str(item)

    def tensor_type(self, *items):
        if len(items) == 3:
        # 2D: tensor<MxNxT>
            rows, cols, inner = items
            return f"tensor<{rows}x{cols}x{inner}>"
        else:
        # 1D: tensor<NxT>
            size, inner = items
            return f"tensor<{size}x{inner}>"

    def ptr_type(self, *items):
        inner = items[0]
        if len(items) > 1:
            return f"!tt.ptr<{inner}, {items[1]}>"
        return f"!tt.ptr<{inner}>"

    def int_type(self, item):
        return str(item)

    def float_type(self, item):
        return str(item)

    def type_list(self, *items):
        return list(items)

    def attr_dict(self, *items):
        return None   # ignore attrs for now

    def VALUE_ID(self, token):
        return str(token)

    def OP_NAME(self, token):
        return str(token)

    def INT(self, token):
        return str(token)


import os
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
from lark import Lark, Transformer

# =====================================================================
# 1. THE DATA MODEL (The Abstract Syntax Tree Node)
# =====================================================================
@dataclass
class TritonInstruction:
    output_var: Optional[str]   
    op_name: str                
    arguments: List[str]        
    attributes: Dict[str, Any]  
    data_type: str              


# =====================================================================
# 2. THE FINAL EBNF GRAMMAR SPECIFICATION
# =====================================================================
GRAMMAR = r"""
    ?start: instruction
    
    ?instruction: ssa_assignment | bare_instruction
    
    ssa_assignment: ssa_id "=" op_name [arg_list] [attr_dict] ":" type_signature
    bare_instruction: op_name [arg_list] [attr_dict] ":" type_signature
    
    arg_list: value_item ("," value_item)*
    ?value_item: ssa_id | INT | DECIMAL | SIGNED_INT | CNAME
    
    attr_dict: "{" attr_entry ("," attr_entry)* "}"
    attr_entry: CNAME "=" attr_value
    
    # ADDED '?' HERE: This automatically flattens out the leaked Lark Tree
    ?attr_value: /[^,}]+/
    
    ssa_id: "%" (CNAME | INT)
    op_name: CNAME ("." CNAME)+
    type_signature: /.+/
    
    %import common.CNAME
    %import common.INT
    %import common.DECIMAL
    %import common.SIGNED_INT
    %import common.ESCAPED_STRING
    %import common.WS
    %ignore WS
"""


# =====================================================================
# 3. THE TREE TRANSFORMER
# =====================================================================
class TritonTransformer(Transformer):
    def ssa_id(self, children):
        return f"%{children[0]}"
        
    def op_name(self, children):
        return ".".join(str(c) for c in children)
        
    def type_signature(self, children):
        return str(children[0]).strip()
        
    def arg_list(self, children):
        return [str(c) for c in children]
        
    def attr_entry(self, children):
        return {str(children[0]): str(children[1]).strip()}
        
    def attr_dict(self, children):
        res = {}
        for c in children:
            res.update(c)
        return res

    def ssa_assignment(self, children):
        output_var = children[0]
        op_name = children[1]
        
        args = []
        attrs = {}
        dtype = ""
        
        for c in children[2:]:
            if isinstance(c, list):
                args = c
            elif isinstance(c, dict):
                attrs = c
            else:
                dtype = str(c)
                
        return TritonInstruction(
            output_var=output_var,
            op_name=op_name,
            arguments=args,
            attributes=attrs,
            data_type=dtype
        )

    def bare_instruction(self, children):
        op_name = children[0]
        
        args = []
        attrs = {}
        dtype = ""
        
        for c in children[1:]:
            if isinstance(c, list):
                args = c
            elif isinstance(c, dict):
                attrs = c
            else:
                dtype = str(c)
                
        return TritonInstruction(
            output_var=None,
            op_name=op_name,
            arguments=args,
            attributes=attrs,
            data_type=dtype
        )


# =====================================================================
# 4. THE INGESTION ENGINE INTERFACE
# =====================================================================
class TritonIRParser:
    def __init__(self):
        self.lark_engine = Lark(GRAMMAR, parser='earley')
        self.transformer = TritonTransformer()

    def parse_line(self, line: str) -> Optional[TritonInstruction]:
        line = line.strip()
        if not line or line.startswith("//") or line.startswith("module") or line.startswith("}") or line == "{":
            return None
            
        if "loc(#" in line:
            line = line.split("loc(#")[0].strip()
            
        try:
            raw_tree = self.lark_engine.parse(line)
            return self.transformer.transform(raw_tree)
        except Exception as e:
            print(f"[Parser Warning] Skipping unmapped/structural line: {line}")
            return None

    def parse_file(self, filepath: str) -> List[TritonInstruction]:
        instructions = []
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Target IR file not found at: {filepath}")
            
        with open(filepath, 'r') as f:
            for line in f:
                parsed_op = self.parse_line(line)
                if parsed_op:
                    instructions.append(parsed_op)
        return instructions


# =====================================================================
# 5. DIAGNOSTIC VERIFICATION HOOK
# =====================================================================
if __name__ == "__main__":
    print("Initializing Lark Engine Diagnostic Pipeline...")
    parser = TritonIRParser()
    
    line_1 = "%3 = tt.addptr %1, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>"
    res_1 = parser.parse_line(line_1)
    print(f"\nParse Check 1:\n{res_1}")
    
    line_2 = "%4 = tt.load %3 {cache = 1 : i32} : tensor<1024xf32>"
    res_2 = parser.parse_line(line_2)
    print(f"\nParse Check 2:\n{res_2}")
    
    line_3 = "tt.store %ptr, %val : tensor<1024x!tt.ptr<f32>>"
    res_3 = parser.parse_line(line_3)
    print(f"\nParse Check 3:\n{res_3}")
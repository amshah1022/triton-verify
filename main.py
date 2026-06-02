import sys
import os
from parser import TritonIRParser
from encoder import TritonZ3Encoder

def run_verification(filepath: str):
    print(f"=== Starting Triton Verification Pipeline ===")
    print(f"Target File: {filepath}\n")
    
    # 1. Initialize our isolated system layers
    parser = TritonIRParser()
    encoder = TritonZ3Encoder()
    
    # 2. Layer 1: Parse the file into structured AST nodes
    print("[Pipeline] Step 1: Parsing raw text IR...")
    try:
        instructions = parser.parse_file(filepath)
        print(f"[Pipeline] Successfully parsed {len(instructions)} instructions.\n")
    except Exception as e:
        print(f"Parsing Error: {e}")
        return

    # 3. Layer 2 & 3: Translate instructions into Z3 equations
    print("[Pipeline] Step 2: Translating nodes into SMT equations...")
    for inst in instructions:
        encoder.encode_instruction(inst)
        
    # 4. Layer 4: Execute the solver pass
    encoder.verify()

if __name__ == "__main__":
    # If a user provides a file via command line, use it. Otherwise, look for 'kernel.ttir'
    target_file = sys.argv[1] if len(sys.argv) > 1 else "kernel.ttir"
    
    if not os.path.exists(target_file):
        print(f"Error: Could not find target file '{target_file}' to verify.")
        print("Creating a sample 'kernel.ttir' for you right now...")
        
        # Generating a sample multi-line file to test the full pipeline
        with open(target_file, "w") as f:
            f.write("// Sample Triton Intermediate Representation\n")
            f.write("%3 = tt.addptr %1, %2 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32>\n")
            f.write("%4 = tt.load %3 {cache = 1 : i32} : tensor<1024xf32>\n")
            
        print(f"Sample file created. Re-running pipeline...\n")
        
    run_verification(target_file)
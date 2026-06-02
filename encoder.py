import z3
from parser import TritonInstruction

class TritonZ3Encoder:
    def __init__(self):
        self.symbol_table = {}
        self.solver = z3.Solver()
        self.MAX_BUFFER_SIZE = 1024

    def _get_or_create_var(self, name: str) -> z3.BitVecRef:
        if name not in self.symbol_table:
            clean_name = name.replace("%", "")
            self.symbol_table[name] = z3.BitVec(clean_name, 64)
        return self.symbol_table[name]

    def encode_instruction(self, inst: TritonInstruction):
        if inst.op_name == "tt.addptr":
            base_ptr = self._get_or_create_var(inst.arguments[0])
            offset = self._get_or_create_var(inst.arguments[1])
            result_ptr = base_ptr + offset
            self.symbol_table[inst.output_var] = result_ptr
            print(f"[Encoder] Modeled address generation for {inst.output_var}")

        # 2. Handle Memory Loading (Looking for bugs!)
        elif inst.op_name == "tt.load":
            source_ptr = self._get_or_create_var(inst.arguments[0])
            
            # BUG HUNTING MODE: Tell Z3 to find a state where the pointer is OUT of bounds (>= 1024)
            self.solver.add(z3.UGE(source_ptr, self.MAX_BUFFER_SIZE))
            
            data_reg = self._get_or_create_var(inst.output_var)
            print(f"[Encoder] Added vulnerability check for loading {inst.output_var}")

        # 3. Handle Memory Storing (Looking for bugs!)
        elif inst.op_name == "tt.store":
            dest_ptr = self._get_or_create_var(inst.arguments[0])
            value = self._get_or_create_var(inst.arguments[1])
            
            # BUG HUNTING MODE: Tell Z3 to find a state where the store is OUT of bounds (>= 1024)
            self.solver.add(z3.UGE(dest_ptr, self.MAX_BUFFER_SIZE))
            print(f"[Encoder] Added vulnerability check for store operation to {inst.arguments[0]}")

    def verify(self):
        print("\n--- Initiating Z3 Mathematical Verification Pass ---")
        result = self.solver.check()
        
        # If SAT, it means Z3 successfully found a way to trigger the violation we added above!
        if result == z3.sat:
            print("❌ SECURITY VULNERABILITY FOUND!")
            print("Z3 found a valid state where memory boundaries could be breached.")
            print("Counterexample context mapping:")
            print(self.solver.model())
        else:
            print("✅ VERIFICATION SUCCESSFUL: No out-of-bounds risks detected for this block configuration.")


if __name__ == "__main__":
    encoder = TritonZ3Encoder()
    
    step_1 = TritonInstruction(output_var='%3', op_name='tt.addptr', arguments=['%1', '%2'], attributes={}, data_type='')
    step_2 = TritonInstruction(output_var='%4', op_name='tt.load', arguments=['%3'], attributes={'cache': '1 : i32'}, data_type='')
    
    print("Feeding parsed instructions to the mathematical encoder...")
    encoder.encode_instruction(step_1)
    encoder.encode_instruction(step_2)
    
    print("\nSimulating an adversarial input constraint (%2 = 2000)...")
    encoder.solver.add(encoder.symbol_table['%2'] == 2000)
    encoder.solver.add(encoder.symbol_table['%1'] == 0) 
    
    encoder.verify()
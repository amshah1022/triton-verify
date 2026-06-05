# triton-verify

Formal verification of pointer safety in Triton GPU kernels using Z3.

## What it does
Takes a Triton TTIR file and proves whether any thread block can 
cause an out-of-bounds memory access, for all possible program IDs.

## Usage
Dump your kernel IR:
    TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump python3 your_kernel.py

Run the verifier:
    python3 main.py dump/your_kernel.ttir <buffer_size> <grid_size>

## Example
    python3 main.py kernel.ttir 512 4
    → SAFE: no out-of-bounds access possible for any program id

    python3 main.py kernel.ttir 500 4
    → BUG FOUND: pid=3, offset_max=511, overflow by 12 elements

## Requirements
    pip install lark z3-solver

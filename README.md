# triton-verify

Formal verification of pointer safety in Triton GPU kernels using Z3 SMT solving.

Given a Triton kernel's TTIR, triton-verify proves — for **all possible program IDs
and grid configurations** — whether any thread block can cause an out-of-bounds memory
access. No test inputs required. No GPU needed to run the verifier.

## Why this matters

LLM-generated Triton kernels are increasingly used in production ML training. These
kernels can contain subtle pointer arithmetic bugs that compile successfully, run without
crashing, but silently corrupt memory at specific grid configurations. triton-verify
provides a static safety guarantee before the kernel ever runs.

## How it works

```
.ttir file → preprocessor → parser → abstractor → Z3 encoder → SAT/UNSAT verdict
```

1. **Preprocessor** — converts Triton's custom IR format to generic MLIR form
2. **Parser** — Lark grammar parses generic MLIR into structured Op objects
3. **Abstractor** — tags each op as scalar or tile, extracts tile intervals (min, max)
4. **Encoder** — translates pointer arithmetic into Z3 BitVec constraints symbolically
5. **Verdict** — Z3 either proves safety (UNSAT) or returns the exact pid causing overflow (SAT)

## Installation

```
pip install lark z3-solver
```

No GPU required. No Triton installation required to run the verifier.

## Usage

First dump your kernel's TTIR from a machine with Triton installed:

```
TRITON_KERNEL_DUMP=1 TRITON_DUMP_DIR=./dump python3 your_kernel.py
```

Then verify it:

```
python3 main.py <kernel.ttir> <buffer_size> <grid_size> [stride]
```

## Examples

Vector add — safe:

```
python3 main.py vector_add.ttir 512 4
→ SAFE: no out-of-bounds access possible for any program id
```

Vector add — bug found:

```
python3 main.py vector_add.ttir 500 4
→ BUG FOUND:
    pid        = 3
    offset_max = 511
    buffer_size= 500
    overflow by= 12 elements
```

Fused softmax — safe:

```
python3 main.py softmax.ttir 512 128
→ SAFE: no out-of-bounds access possible for any program id
```

Layer norm — bug found:

```
python3 main.py layernorm.ttir 60000 128 512
→ BUG FOUND:
    pid        = 120
    offset_max = 61951
    buffer_size= 60000
    overflow by= 1952 elements
```

Matrix multiply — safe:

```
python3 main.py matmul.ttir 491584 8 512
→ SAFE: no out-of-bounds access possible for any program id
```

## Kernels verified

| Kernel | Safety proof | Bug detection | Notes |
|--------|-------------|---------------|-------|
| Vector add | ✓ | ✓ | Full verification |
| Fused softmax | ✓ | ✓ | Masked loads, row access |
| Layer norm | ✓ | ✓ | Multiple buffers |
| Matrix multiply | ✓ | Partial | Loop boundary conservative |

## Parameters

| Argument | Description |
|----------|-------------|
| kernel.ttir | Path to dumped TTIR file |
| buffer_size | Total number of elements in the input buffer |
| grid_size | Number of thread blocks in the grid |
| stride | Row stride for 2D kernels like matmul and layer norm |

## Limitations

- Strides must be provided manually for kernels with 2D access patterns
- Loop precision — matmul boundary detection is conservative
- Numerical correctness is out of scope — only pointer safety is verified
- scf.for loops with complex loop-carried dependencies are partially supported
- Verified on TTIR (pre-optimization IR) — later compiler passes are not checked

## Project structure

| File | Responsibility |
|------|----------------|
| preprocessor.py | Converts Triton custom IR to generic MLIR form |
| parser.py | Lark grammar and transformer for generic MLIR ops |
| abstractor.py | Tags ops scalar/tile, extracts tile size |
| encoder.py | Z3 BitVec encoding of pointer arithmetic |
| main.py | CLI driver, argument seeding, pipeline orchestration |

## Background

This tool was motivated by the gap between existing verification tools (which target
CUDA or generic MLIR) and the specific verification needs of Triton's tile-based
execution model. The key insight is that Triton's pointer arithmetic maps naturally
to interval arithmetic over Z3 BitVec variables, where pid is a single symbolic
unknown constrained to [0, grid_size).

## Related work

- ProofWright (NVIDIA/Georgia Tech) — memory safety for CUDA kernels using LLM + SMT
- MLIR translation validation (Wang et al.) — compiler pass correctness for generic MLIR
- GPU kernel equivalence checking (Microsoft/Stanford) — semantic equivalence for CUDA

## Contributing

Contributions welcome. High-value areas:

- Flash attention support
- Automatic buffer size and stride extraction from kernel launch sites
- Handling scf.for with symbolic loop bounds
- Integration into Triton's compilation pipeline

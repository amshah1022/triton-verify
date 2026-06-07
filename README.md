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
python3 main.py <kernel.ttir> <buffer_size> <grid_size> [stride] [key=value ...]
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

vLLM swiglu kernel — safe:

```
python3 main.py swiglu.ttir 65536 32 x_stride=2048 o_stride=1024 grid1=1
→ SAFE: no out-of-bounds access possible for any program id
```

vLLM swiglu kernel — bug found:

```
python3 main.py swiglu.ttir 60000 32 x_stride=2048 o_stride=1024 grid1=1
→ BUG FOUND:
    pid        = 29
    offset_max = 60415
    buffer_size= 60000
    overflow by= 416 elements
```

Flash attention — safe:

```
python3 main.py flash_attn.ttir 2048 4 stride_qm=32 stride_qk=1 stride_kn=32 stride_kk=1 stride_vn=32 stride_vk=1 stride_om=32 stride_ok=1 N_CTX=64
→ SAFE: no out-of-bounds access possible for any program id
```

## Kernels verified

20/20 tests passing across 10 kernels compiled on real GPU hardware (Google Colab T4).

| Kernel | Source | Safety proof | Bug detection | Notes |
|--------|--------|-------------|---------------|-------|
| Vector add | Triton tutorial | ✓ | ✓ | Full verification |
| Fused softmax | Triton tutorial | ✓ | ✓ | Masked loads, row access |
| Layer norm | Triton tutorial | ✓ | ✓ | Multiple buffers |
| Matrix multiply | Triton tutorial | ✓ | Partial | Loop boundary conservative |
| Flash attention | Custom | ✓ | ✓ | Nested loop, 2D tiles |
| SwiGLU step | vLLM activation.py | ✓ | ✓ | i64 strides, 2D grid |
| NaN counter | vLLM metrics/logits.py | ✓ | ✓ | Loop induction variable |
| DCP seq lens | vLLM cp_utils.py | ✓ | ✓ | Integer div/rem/sub |
| QKV FP8 quant | vLLM qkv_padded_fp8_quant.py | ✓ | ✓ | 2D grid, div/rem indexing |
| Scale swizzle | vLLM qutlass_utils.py | ✓ | ✓ | 2D tiling, row-major layout |

## Parameters

| Argument | Description |
|----------|-------------|
| kernel.ttir | Path to dumped TTIR file |
| buffer_size | Total number of elements in the input buffer |
| grid_size | Number of thread blocks along axis 0 |
| stride | Row stride (applies to all stride args if not overridden) |
| key=value | Per-argument overrides e.g. x_stride=2048 o_stride=1024 |
| grid1=N | Grid size along axis 1 for 2D kernels |

## Limitations

- Buffer sizes and strides must be provided manually (auto-extraction planned)
- Loop precision — matmul loop boundary detection is conservative
- Numerical correctness is out of scope — only pointer safety is verified
- scf.for loops with complex loop-carried dependencies are partially supported
- Verified on TTIR (pre-optimization IR) — later compiler passes are not checked
- Masked loads are over-approximated — may produce false positives for masked kernels

## Project structure

| File | Responsibility |
|------|----------------|
| preprocessor.py | Converts Triton custom IR to generic MLIR form |
| parser.py | Lark grammar and transformer for generic MLIR ops |
| abstractor.py | Tags ops scalar/tile, extracts tile size |
| encoder.py | Z3 BitVec encoding of pointer arithmetic |
| main.py | CLI driver, argument seeding, pipeline orchestration |
| test.py | Test suite — 20 cases across 10 kernels |

## Background

This tool was motivated by the gap between existing verification tools (which target
CUDA or generic MLIR) and the specific verification needs of Triton's tile-based
execution model. The key insight is that Triton's pointer arithmetic maps naturally
to interval arithmetic over Z3 BitVec variables, where pid is a single symbolic
unknown constrained to [0, grid_size). This allows Z3 to reason about all possible
thread block IDs simultaneously in a single query.

The tool has been tested on kernels from the Triton tutorial suite and from vLLM,
one of the most widely deployed LLM inference engines.

## Related work

- ProofWright (NVIDIA/Georgia Tech) — memory safety for CUDA kernels using LLM + SMT
- MLIR translation validation (Wang et al.) — compiler pass correctness for generic MLIR
- GPU kernel equivalence checking (Microsoft/Stanford) — semantic equivalence for CUDA

## Contributing

Contributions welcome. High-value areas:

- Automatic buffer size and stride extraction from kernel launch sites
- Handling scf.for with symbolic loop bounds
- Integration into Triton's compilation pipeline
- Additional vLLM and liger-kernel coverage

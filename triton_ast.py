"""
triton_ast.py
-------------
Typed AST nodes for every op in the Triton `tt` dialect.

These are produced by triton_encoder.py from the generic MLIRModule AST
emitted by mlir_parser.py.  The parser is never imported here; this file
is pure data.

Naming convention
-----------------
  - Class names match the C++ class name from the spec (AddPtrOp, LoadOp, …)
    but live under this module so you can do:
        from triton_ast import LoadOp, DotOp, FuncOp
  - Every operand/result field uses the *spec name* (ptr, mask, other, result…)
  - Attribute fields also use the spec name (cache, evict, isVolatile, axis…)
  - Optional operands are typed  Optional[str]  (None = not present)
  - result / results fields hold the %name strings from the SSA assignment

Memory-effect tags (used by the verifier)
-----------------------------------------
  PURE       – NoMemoryEffect
  READ       – reads global memory
  WRITE      – writes global memory
  READ_WRITE – both
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional


# ---------------------------------------------------------------------------
# Shared enumerations (mirror the allowed integer cases in the spec)
# ---------------------------------------------------------------------------

class MemSemantic(Enum):
    ACQUIRE           = 1
    RELEASE           = 2
    ACQUIRE_RELEASE   = 3
    RELAXED           = 4

class MemSyncScope(Enum):
    GPU    = 1
    CTA    = 2
    SYSTEM = 3

class RMWOp(Enum):
    AND   = 1
    OR    = 2
    XOR   = 3
    ADD   = 4
    FADD  = 5
    MAX   = 6
    MIN   = 7
    UMAX  = 8
    UMIN  = 9
    XCHG  = 10

class CacheModifier(Enum):
    NONE   = 1
    CA     = 2
    CG     = 3
    CS     = 4
    CV     = 5
    L1     = 6
    L2     = 7

class EvictionPolicy(Enum):
    NORMAL    = 1
    EVICT_FIRST = 2
    EVICT_LAST  = 3

class InputPrecision(Enum):
    TF32     = 0
    TF32x3   = 1
    IEEE     = 2
    BF16x3   = 3
    BF16x6   = 4

class RoundingMode(Enum):
    RTNE = 0   # round-to-nearest-even
    RTZ  = 1   # round-toward-zero

class ProgramIDDim(Enum):
    X = 0
    Y = 1
    Z = 2

class PropagateNan(Enum):
    NONE = 0
    ALL  = 65535

class PaddingOption(Enum):
    PAD_ZERO  = 1
    PAD_NAN   = 2

class ScaleDotElemType(Enum):
    E4M3   = 0
    E5M2   = 1
    E2M3   = 2
    E3M2   = 3
    E2M1   = 4
    UE8M0  = 5
    UE2    = 6

class DescriptorReduceKind(Enum):
    ADD  = 1
    MIN  = 2
    MAX  = 3
    AND  = 4
    OR   = 5
    XOR  = 6
    FADD = 7
    FMIN = 8  # spec lists 8 cases total


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------

@dataclass
class TritonOp:
    """Base for all Triton ops.  result_names mirrors the SSA %foo from the
    left-hand side of the assignment that produced this op."""
    result_names: List[str] = field(default_factory=list)
    # source location from the trailing loc(…) if present
    location: Any = None


# ---------------------------------------------------------------------------
# Pointer arithmetic
# ---------------------------------------------------------------------------

@dataclass
class AddPtrOp(TritonOp):
    """tt.addptr  ptr, offset  ->  result (ptr + offset)"""
    ptr: str = ""
    offset: str = ""
    result_type: Any = None
    offset_type: Any = None


# ---------------------------------------------------------------------------
# Control / assertions / printing
# ---------------------------------------------------------------------------

@dataclass
class AssertOp(TritonOp):
    """tt.assert condition, message"""
    condition: str = ""
    message: str = ""
    condition_type: Any = None


@dataclass
class PrintOp(TritonOp):
    """tt.print prefix  (:  args : types)?"""
    prefix: str = ""
    hex: bool = False
    is_signed: List[int] = field(default_factory=list)
    args: List[str] = field(default_factory=list)
    arg_types: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Atomic operations
# ---------------------------------------------------------------------------

@dataclass
class AtomicCASOp(TritonOp):
    """tt.atomic_cas  sem, scope, ptr, cmp, val  ->  result"""
    sem: Optional[MemSemantic] = None
    scope: Optional[MemSyncScope] = None
    ptr: str = ""
    cmp: str = ""
    val: str = ""
    result_type: Any = None


@dataclass
class AtomicRMWOp(TritonOp):
    """tt.atomic_rmw  rmw_op, sem, scope, ptr, val (, mask)?  ->  result"""
    atomic_rmw_op: Optional[RMWOp] = None
    sem: Optional[MemSemantic] = None
    scope: Optional[MemSyncScope] = None
    ptr: str = ""
    val: str = ""
    mask: Optional[str] = None
    result_type: Any = None


# ---------------------------------------------------------------------------
# Type casts
# ---------------------------------------------------------------------------

@dataclass
class BitcastOp(TritonOp):
    """tt.bitcast src -> result"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


@dataclass
class FpToFpOp(TritonOp):
    """tt.fp_to_fp src (, rounding = …)? -> result"""
    src: str = ""
    rounding: Optional[RoundingMode] = None
    src_type: Any = None
    result_type: Any = None


@dataclass
class IntToPtrOp(TritonOp):
    """tt.int_to_ptr src -> result"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


@dataclass
class PtrToIntOp(TritonOp):
    """tt.ptr_to_int src -> result"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


# ---------------------------------------------------------------------------
# Shape / layout ops
# ---------------------------------------------------------------------------

@dataclass
class BroadcastOp(TritonOp):
    """tt.broadcast src -> result"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


@dataclass
class ExpandDimsOp(TritonOp):
    """tt.expand_dims src {axis} -> result"""
    src: str = ""
    axis: int = 0
    src_type: Any = None
    result_type: Any = None


@dataclass
class ReshapeOp(TritonOp):
    """tt.reshape src (allow_reorder)? (efficient_layout)? -> result"""
    src: str = ""
    allow_reorder: bool = False
    efficient_layout: bool = False
    src_type: Any = None
    result_type: Any = None


@dataclass
class TransOp(TritonOp):
    """tt.trans src {order} -> result"""
    src: str = ""
    order: List[int] = field(default_factory=list)
    src_type: Any = None
    result_type: Any = None


@dataclass
class SplatOp(TritonOp):
    """tt.splat src -> result (broadcast scalar to tensor)"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


@dataclass
class UnsplatOp(TritonOp):
    """tt.unsplat src -> result (single-element tensor to scalar)"""
    src: str = ""
    src_type: Any = None
    result_type: Any = None


@dataclass
class CatOp(TritonOp):
    """tt.cat lhs, rhs -> result"""
    lhs: str = ""
    rhs: str = ""
    lhs_type: Any = None
    result_type: Any = None


@dataclass
class JoinOp(TritonOp):
    """tt.join lhs, rhs -> result (new minor dim of size 2)"""
    lhs: str = ""
    rhs: str = ""
    lhs_type: Any = None
    result_type: Any = None


@dataclass
class SplitOp(TritonOp):
    """tt.split src -> outLHS, outRHS"""
    src: str = ""
    src_type: Any = None
    lhs_type: Any = None   # both outputs have same type


# ---------------------------------------------------------------------------
# Memory access
# ---------------------------------------------------------------------------

@dataclass
class LoadOp(TritonOp):
    """tt.load ptr (, mask)? (, other)?  [cache/evict attrs]  -> result"""
    ptr: str = ""
    mask: Optional[str] = None
    other: Optional[str] = None
    cache: Optional[CacheModifier] = None
    evict: Optional[EvictionPolicy] = None
    is_volatile: bool = False
    ptr_type: Any = None
    result_type: Any = None


@dataclass
class StoreOp(TritonOp):
    """tt.store ptr, value (, mask)?  [cache/evict attrs]"""
    ptr: str = ""
    value: str = ""
    mask: Optional[str] = None
    cache: Optional[CacheModifier] = None
    evict: Optional[EvictionPolicy] = None
    ignore_cta: bool = False
    ptr_type: Any = None


# ---------------------------------------------------------------------------
# Range / indexing
# ---------------------------------------------------------------------------

@dataclass
class MakeRangeOp(TritonOp):
    """tt.make_range {start, end} -> result  (1D i32 range tensor)"""
    start: int = 0
    end: int = 0
    result_type: Any = None


@dataclass
class GetProgramIdOp(TritonOp):
    """tt.get_program_id {axis} -> result (i32)"""
    axis: Optional[ProgramIDDim] = None
    result_type: Any = None


@dataclass
class GetNumProgramsOp(TritonOp):
    """tt.get_num_programs {axis} -> result (i32)"""
    axis: Optional[ProgramIDDim] = None
    result_type: Any = None


# ---------------------------------------------------------------------------
# Math / linear-algebra
# ---------------------------------------------------------------------------

@dataclass
class DotOp(TritonOp):
    """tt.dot a, b, c (inputPrecision = …)? -> d   (d = a@b + c)"""
    a: str = ""
    b: str = ""
    c: str = ""
    input_precision: Optional[InputPrecision] = None
    max_num_imprecise_acc: Optional[int] = None
    a_type: Any = None
    b_type: Any = None
    result_type: Any = None


@dataclass
class DotScaledOp(TritonOp):
    """tt.dot_scaled  a (scale a_scale)?, b (scale b_scale)?, c  lhs=…  rhs=… -> d"""
    a: str = ""
    b: str = ""
    c: str = ""
    a_scale: Optional[str] = None
    b_scale: Optional[str] = None
    a_elem_type: Optional[ScaleDotElemType] = None
    b_elem_type: Optional[ScaleDotElemType] = None
    fast_math: bool = False
    lhs_k_pack: bool = False
    rhs_k_pack: bool = False
    result_type: Any = None


@dataclass
class ReduceOp(TritonOp):
    """tt.reduce {axis}  srcs -> results  (with single-block combiner region)"""
    axis: int = 0
    srcs: List[str] = field(default_factory=list)
    src_types: List[Any] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)
    combiner: Any = None   # Region node from mlir_parser


@dataclass
class ReduceReturnOp(TritonOp):
    """tt.reduce.return result"""
    result: List[str] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)


@dataclass
class ScanOp(TritonOp):
    """tt.scan {axis, reverse} srcs -> results"""
    axis: int = 0
    reverse: bool = False
    srcs: List[str] = field(default_factory=list)
    src_types: List[Any] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)
    combiner: Any = None


@dataclass
class ScanReturnOp(TritonOp):
    """tt.scan.return result"""
    result: List[str] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)


@dataclass
class GatherOp(TritonOp):
    """tt.gather src[indices] {axis} -> result"""
    src: str = ""
    indices: str = ""
    axis: int = 0
    efficient_layout: bool = False
    src_type: Any = None
    indices_type: Any = None
    result_type: Any = None


@dataclass
class HistogramOp(TritonOp):
    """tt.histogram src (, mask)? -> result"""
    src: str = ""
    mask: Optional[str] = None
    src_type: Any = None
    result_type: Any = None


@dataclass
class ClampFOp(TritonOp):
    """tt.clampf x, min, max, propagateNan = … -> result"""
    x: str = ""
    min: str = ""
    max: str = ""
    propagate_nan: Optional[PropagateNan] = None
    result_type: Any = None


@dataclass
class MulhiUIOp(TritonOp):
    """tt.mulhiui x, y -> result"""
    x: str = ""
    y: str = ""
    x_type: Any = None


@dataclass
class PreciseDivFOp(TritonOp):
    """tt.precise_divf x, y -> result"""
    x: str = ""
    y: str = ""
    x_type: Any = None


@dataclass
class PreciseSqrtOp(TritonOp):
    """tt.precise_sqrt x -> result"""
    x: str = ""
    x_type: Any = None


# ---------------------------------------------------------------------------
# Elementwise inline / external
# ---------------------------------------------------------------------------

@dataclass
class ElementwiseInlineAsmOp(TritonOp):
    """tt.elementwise_inline_asm asm_string (args : types)? -> result"""
    asm_string: str = ""
    constraints: str = ""
    pure: bool = True
    packed_element: int = 1
    args: List[str] = field(default_factory=list)
    arg_types: List[Any] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)


@dataclass
class ExternElementwiseOp(TritonOp):
    """tt.extern_elementwise srcs -> result"""
    libname: str = ""
    libpath: str = ""
    symbol: str = ""
    pure: bool = True
    srcs: List[str] = field(default_factory=list)
    src_types: List[Any] = field(default_factory=list)
    result_type: Any = None


@dataclass
class MapElementwiseOp(TritonOp):
    """tt.map_elementwise srcs -> results"""
    pack: int = 1
    srcs: List[str] = field(default_factory=list)
    src_types: List[Any] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)
    body: Any = None   # Region


@dataclass
class MapElementwiseReturnOp(TritonOp):
    """tt.map_elementwise.return result"""
    result: List[str] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# TMA / descriptor ops
# ---------------------------------------------------------------------------

@dataclass
class MakeTensorDescOp(TritonOp):
    """tt.make_tensor_descriptor base, [shape], [strides] -> result"""
    base: str = ""
    shape: List[str] = field(default_factory=list)
    strides: List[str] = field(default_factory=list)
    padding: Optional[PaddingOption] = None
    base_type: Any = None
    result_type: Any = None


@dataclass
class DescriptorLoadOp(TritonOp):
    """tt.descriptor_load desc[indices] -> result"""
    desc: str = ""
    indices: List[str] = field(default_factory=list)
    cache: Optional[CacheModifier] = None
    evict: Optional[EvictionPolicy] = None
    desc_type: Any = None
    result_type: Any = None


@dataclass
class DescriptorStoreOp(TritonOp):
    """tt.descriptor_store desc[indices], src"""
    desc: str = ""
    indices: List[str] = field(default_factory=list)
    src: str = ""
    desc_type: Any = None
    src_type: Any = None


@dataclass
class DescriptorGatherOp(TritonOp):
    """tt.descriptor_gather desc[x_offsets, y_offset] -> result"""
    desc: str = ""
    x_offsets: str = ""
    y_offset: str = ""
    desc_type: Any = None
    x_offsets_type: Any = None
    result_type: Any = None


@dataclass
class DescriptorScatterOp(TritonOp):
    """tt.descriptor_scatter desc[x_offsets, y_offset], src"""
    desc: str = ""
    x_offsets: str = ""
    y_offset: str = ""
    src: str = ""
    desc_type: Any = None


@dataclass
class DescriptorReduceOp(TritonOp):
    """tt.descriptor_reduce kind, desc[indices], src"""
    kind: Optional[DescriptorReduceKind] = None
    desc: str = ""
    indices: List[str] = field(default_factory=list)
    src: str = ""
    desc_type: Any = None
    src_type: Any = None


# ---------------------------------------------------------------------------
# Function / call / return
# ---------------------------------------------------------------------------

@dataclass
class FuncOp(TritonOp):
    """tt.func @name(args) -> results {body}"""
    sym_name: str = ""
    function_type: Any = None    # FunctionType from mlir_ast
    sym_visibility: Optional[str] = None
    arg_attrs: List[Any] = field(default_factory=list)
    res_attrs: List[Any] = field(default_factory=list)
    body: Any = None             # Region or None (external declaration)


@dataclass
class CallOp(TritonOp):
    """tt.call @callee(operands) -> results"""
    callee: str = ""
    operands: List[str] = field(default_factory=list)
    operand_types: List[Any] = field(default_factory=list)
    result_types: List[Any] = field(default_factory=list)
    arg_attrs: List[Any] = field(default_factory=list)
    res_attrs: List[Any] = field(default_factory=list)


@dataclass
class ReturnOp(TritonOp):
    """tt.return srcs : types"""
    srcs: List[str] = field(default_factory=list)
    src_types: List[Any] = field(default_factory=list)


# ---------------------------------------------------------------------------
# A typed Triton module (output of the encoder)
# ---------------------------------------------------------------------------

@dataclass
class TritonModule:
    """
    Root node returned by triton_encoder.encode().

    ops       – top-level TritonOps (usually one or more FuncOps)
    unhandled – any GenericOperation / CustomOperation the encoder
                did not recognise (kept verbatim so nothing is lost)
    """
    ops: List[TritonOp] = field(default_factory=list)
    unhandled: List[Any] = field(default_factory=list)
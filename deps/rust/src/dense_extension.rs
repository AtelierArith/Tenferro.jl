//! Native dense layers: `act(W * x .+ b)` in one GEMM plus one pass.
//!
//! Composed from `dot_general`, a bias broadcast, an add and an activation, a
//! dense layer and its derivative take several tensors and runtime
//! instructions each. This family has three members over the same matrices,
//! `W` (`o × i`), `x` (`i × n`), the bias `b` (length `o`) and the output
//! `y` (`o × n`), with `M(t) = y > 0 ? t : 0` for ReLU and `M(t) = t` for the
//! identity:
//!
//! - `Dense(W, x[, b]) = act(W x + b)`.
//! - `DenseT(y, W, x, [dW], [dx], [db]) = M(dW x + W dx + db)`: the tangent map.
//! - `DenseB(y, W, x, c) = ([M(c) xᵀ], [Wᵀ M(c)], [Σⱼ M(c)])`: the cotangent map,
//!   with only the requested outputs.
//!
//! `DenseT` and `DenseB` are each other's transposes in their last inputs, and
//! the derivatives of both are `DenseT`/`DenseB` terms again (the family is
//! closed), so derivatives of every order stay native. The derivative of ReLU at
//! 0 is 0, like NNlib.
//!
//! The GEMMs run on Julia's BLAS (through libblastrampoline, registered with
//! [`crate::kernels::set_blas_gemm`]), so they follow `BLAS.set_num_threads`.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRuleSet,
    SemanticLinearTransposeRequest, SemanticLinearTransposeRule, SemanticLinearizeRequest,
    SemanticLinearizeResult, SemanticLinearizeRule, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_cpu::threadteam;
use tenferro_runtime::extension::{
    define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp,
    ExtensionShapeContext,
};
use tenferro_runtime::program::{CoreSemanticOp, ProgramValue, SemanticProgramBuilder};
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

use crate::kernels::{self, Elem, MultiKernelFn};

const FAMILY: &str = "tenferro-julia.dense.v1";

/// The activation of a dense layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Act {
    Identity,
    Relu,
}

impl Act {
    pub fn from_code(code: i32) -> Option<Self> {
        match code {
            0 => Some(Self::Identity),
            1 => Some(Self::Relu),
            _ => None,
        }
    }
}

/// Which of `W`, `x` and `b` a tangent or cotangent map involves.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Parts {
    pub w: bool,
    pub x: bool,
    pub b: bool,
}

impl Parts {
    fn any(self) -> bool {
        self.w || self.x || self.b
    }
    fn count(self) -> usize {
        self.w as usize + self.x as usize + self.b as usize
    }
}

/// The members of the dense family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DenseOp {
    /// `act(W x [+ b])`: inputs `W, x[, b]`.
    Dense { act: Act, bias: bool },
    /// `M(dW x + W dx + db)` for the present tangents: inputs `y, W, x, [dW], [dx], [db]`.
    Tangent { act: Act, parts: Parts },
    /// `[M(c) xᵀ], [Wᵀ M(c)], [Σⱼ M(c)]` for the requested parts: inputs `y, W, x, c`.
    Cotangent { act: Act, parts: Parts },
}

impl DenseOp {
    fn act(self) -> Act {
        match self {
            Self::Dense { act, .. } | Self::Tangent { act, .. } | Self::Cotangent { act, .. } => act,
        }
    }
}

impl ExtensionOp for DenseOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, mut state: &mut dyn Hasher) {
        self.hash(&mut state);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().downcast_ref::<Self>() == Some(self)
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(*self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        match *self {
            Self::Dense { bias, .. } => 2 + bias as usize,
            Self::Tangent { parts, .. } => 3 + parts.count(),
            Self::Cotangent { .. } => 4,
        }
    }
    fn output_count(&self) -> usize {
        match *self {
            Self::Cotangent { parts, .. } => parts.count(),
            _ => 1,
        }
    }
    fn semantic_effects(&self) -> ExtensionEffectDeclaration<'_> {
        ExtensionEffectDeclaration::Declared(&[])
    }
    fn semantic_aliases(&self) -> ExtensionAliasDeclaration<'_> {
        ExtensionAliasDeclaration::AllFresh
    }
    fn infer_output_meta(
        &self,
        ctx: &mut ExtensionShapeContext<'_>,
    ) -> tenferro_tensor::Result<Vec<(DType, Vec<tenferro_runtime::SymDim>)>> {
        let fail = |message: &str| tenferro_tensor::Error::runtime_state("dense", message.to_string());
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(tenferro_tensor::Error::unsupported_dtype("dense", dtype, "dense layers support Float32 and Float64"));
        }
        for i in 0..self.input_count() {
            if ctx.input_dtype(i)? != dtype {
                return Err(fail("operands must have one element type"));
            }
        }
        let rank = |ctx: &ExtensionShapeContext<'_>, i: usize, r: usize| -> tenferro_tensor::Result<()> {
            if ctx.input_shape(i)?.len() == r {
                Ok(())
            } else {
                Err(fail("dense operands must be matrices and a bias vector"))
            }
        };
        match *self {
            Self::Dense { bias, .. } => {
                rank(ctx, 0, 2)?;
                rank(ctx, 1, 2)?;
                let (o, i) = (ctx.input_axis(0, 0)?, ctx.input_axis(0, 1)?);
                let n = ctx.input_axis(1, 1)?;
                ctx.require_equal(i.clone(), ctx.input_axis(1, 0)?)?;
                if bias {
                    rank(ctx, 2, 1)?;
                    ctx.require_equal(o.clone(), ctx.input_axis(2, 0)?)?;
                }
                Ok(vec![(dtype, vec![o, n])])
            }
            Self::Tangent { parts, .. } | Self::Cotangent { parts, .. } => {
                if !parts.any() {
                    return Err(fail("a dense derivative needs at least one part"));
                }
                for k in 0..3 {
                    rank(ctx, k, 2)?;
                }
                let (o, n) = (ctx.input_axis(0, 0)?, ctx.input_axis(0, 1)?);
                let (w_rows, i) = (ctx.input_axis(1, 0)?, ctx.input_axis(1, 1)?);
                let (x_rows, x_cols) = (ctx.input_axis(2, 0)?, ctx.input_axis(2, 1)?);
                // `W` is only read when `x` takes part (and vice versa), but shapes must agree.
                ctx.require_equal(w_rows, o.clone())?;
                ctx.require_equal(x_cols, n.clone())?;
                ctx.require_equal(i.clone(), x_rows)?;
                if let Self::Tangent { parts, .. } = *self {
                    let mut k = 3;
                    if parts.w {
                        rank(ctx, k, 2)?;
                        ctx.require_axes_equal((k, 0), (1, 0))?;
                        ctx.require_axes_equal((k, 1), (1, 1))?;
                        k += 1;
                    }
                    if parts.x {
                        rank(ctx, k, 2)?;
                        ctx.require_axes_equal((k, 0), (2, 0))?;
                        ctx.require_axes_equal((k, 1), (2, 1))?;
                        k += 1;
                    }
                    if parts.b {
                        rank(ctx, k, 1)?;
                        ctx.require_axes_equal((k, 0), (0, 0))?;
                    }
                    Ok(vec![(dtype, vec![o, n])])
                } else {
                    rank(ctx, 3, 2)?;
                    ctx.require_axes_equal((3, 0), (0, 0))?;
                    ctx.require_axes_equal((3, 1), (0, 1))?;
                    let mut outs = Vec::new();
                    if parts.w {
                        outs.push((dtype, vec![o.clone(), i.clone()]));
                    }
                    if parts.x {
                        outs.push((dtype, vec![i.clone(), n.clone()]));
                    }
                    if parts.b {
                        outs.push((dtype, vec![o.clone()]));
                    }
                    Ok(outs)
                }
            }
        }
    }
}

// Kernels ------------------------------------------------------------------------------

/// `M(t)` in place, with `y` the layer output.
fn mask_in_place<T: Elem>(act: Act, y: &[T], t: &mut [T]) {
    if act == Act::Relu {
        let zero = T::default();
        // A select, not a branch: the signs of `y` are unpredictable.
        t.iter_mut().zip(y).for_each(|(v, &y)| *v = if y > zero { *v } else { zero });
    }
}

struct Kernel {
    op: DenseOp,
    /// `(o, i, n)`.
    dims: (usize, usize, usize),
}

impl MultiKernelFn for Kernel {
    fn run<T: Elem>(&self, inputs: &[&[T]], outs: &mut [Vec<T>], _: Option<threadteam::Exec<'_>>) {
        let (o, i, n) = self.dims;
        match self.op {
            DenseOp::Dense { act, bias } => {
                let y = &mut outs[0];
                T::gemm(false, false, o, n, i, inputs[0], o, inputs[1], i, false, y);
                let zero = T::default();
                let b = bias.then(|| inputs[2]);
                for col in y.chunks_exact_mut(o.max(1)).take(n) {
                    match (b, act) {
                        (Some(b), Act::Relu) => col.iter_mut().zip(b).for_each(|(v, &b)| {
                            let s = *v + b;
                            *v = if s > zero { s } else { zero };
                        }),
                        (Some(b), Act::Identity) => col.iter_mut().zip(b).for_each(|(v, &b)| *v += b),
                        (None, Act::Relu) => col.iter_mut().for_each(|v| *v = if *v > zero { *v } else { zero }),
                        (None, Act::Identity) => {}
                    }
                }
            }
            DenseOp::Tangent { act, parts } => {
                let (y, w, x) = (inputs[0], inputs[1], inputs[2]);
                let out = &mut outs[0];
                let mut k = 3;
                let mut written = false;
                if parts.w {
                    T::gemm(false, false, o, n, i, inputs[k], o, x, i, false, out);
                    written = true;
                    k += 1;
                }
                if parts.x {
                    T::gemm(false, false, o, n, i, w, o, inputs[k], i, written, out);
                    k += 1;
                }
                if parts.b {
                    let db = inputs[k];
                    for col in out.chunks_exact_mut(o.max(1)).take(n) {
                        col.iter_mut().zip(db).for_each(|(v, &b)| *v += b);
                    }
                }
                mask_in_place(act, y, out);
            }
            DenseOp::Cotangent { act, parts } => {
                let (y, w, x, c) = (inputs[0], inputs[1], inputs[2], inputs[3]);
                let masked;
                let z: &[T] = if act == Act::Identity {
                    c
                } else {
                    let mut t = c.to_vec();
                    mask_in_place(act, y, &mut t);
                    masked = t;
                    &masked
                };
                let mut slot = 0;
                if parts.w {
                    T::gemm(false, true, o, i, n, z, o, x, i, false, &mut outs[slot]);
                    slot += 1;
                }
                if parts.x {
                    T::gemm(true, false, i, n, o, w, o, z, o, false, &mut outs[slot]);
                    slot += 1;
                }
                if parts.b {
                    let db = &mut outs[slot];
                    for col in z.chunks_exact(o.max(1)).take(n) {
                        db.iter_mut().zip(col).for_each(|(s, &v)| *s += v);
                    }
                }
            }
        }
    }
}

fn execute_dense(
    op: &DenseOp,
    engine: &mut dyn Engine,
    _: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let dims = match *op {
        DenseOp::Dense { .. } => (inputs[0].shape()[0], inputs[0].shape()[1], inputs[1].shape()[1]),
        _ => (inputs[0].shape()[0], inputs[2].shape()[0], inputs[0].shape()[1]),
    };
    let (o, i, n) = dims;
    let shapes = match *op {
        DenseOp::Dense { .. } | DenseOp::Tangent { .. } => vec![vec![o, n]],
        DenseOp::Cotangent { parts, .. } => {
            let mut shapes = Vec::new();
            if parts.w {
                shapes.push(vec![o, i]);
            }
            if parts.x {
                shapes.push(vec![i, n]);
            }
            if parts.b {
                shapes.push(vec![o]);
            }
            shapes
        }
    };
    kernels::execute_multi("dense", engine, inputs, shapes, Kernel { op: *op, dims })
}

fn supported<B: Engine + 'static>(_: &DenseOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<tenferro_cpu::CpuEngine>()
}

define_extension_runtime! {
    runtime = DenseRuntime,
    family_id = FAMILY,
    op_type = DenseOp,
    execute_in_session = execute_dense,
    session_supported = supported,
    backend_bound = Engine,
}

// Derivative rules ---------------------------------------------------------------------

#[derive(Debug)]
struct DenseRule;

fn dense_op(op: &dyn ExtensionOp, role: SemanticAdRuleRole) -> Result<DenseOp, SemanticAdError> {
    op.as_any().downcast_ref::<DenseOp>().copied().ok_or_else(|| SemanticAdError::Unsupported {
        family_id: FAMILY,
        role,
        message: "unexpected payload in the dense family".into(),
    })
}

/// `DenseT(y, w, x, dw, dx, db)` over the present tangents (`None` when all are absent).
fn tangent(
    builder: &mut SemanticProgramBuilder,
    act: Act,
    (y, w, x): (ProgramValue, ProgramValue, ProgramValue),
    dw: Option<ProgramValue>,
    dx: Option<ProgramValue>,
    db: Option<ProgramValue>,
) -> Result<Option<ProgramValue>, SemanticAdError> {
    let parts = Parts { w: dw.is_some(), x: dx.is_some(), b: db.is_some() };
    if !parts.any() {
        return Ok(None);
    }
    let mut inputs = vec![y, w, x];
    inputs.extend([dw, dx, db].into_iter().flatten());
    Ok(Some(builder.add_extension(Arc::new(DenseOp::Tangent { act, parts }), &inputs)?[0]))
}

/// `DenseB(y, w, x, c)` for the requested parts, as `[w̄, x̄, b̄]`.
fn cotangent(
    builder: &mut SemanticProgramBuilder,
    act: Act,
    (y, w, x): (ProgramValue, ProgramValue, ProgramValue),
    c: ProgramValue,
    parts: Parts,
) -> Result<[Option<ProgramValue>; 3], SemanticAdError> {
    let mut result = [None; 3];
    if !parts.any() {
        return Ok(result);
    }
    let outs = builder.add_extension(Arc::new(DenseOp::Cotangent { act, parts }), &[y, w, x, c])?;
    let mut outs = outs.iter().copied();
    for (slot, wanted) in result.iter_mut().zip([parts.w, parts.x, parts.b]) {
        if wanted {
            *slot = outs.next();
        }
    }
    Ok(result)
}

fn add(builder: &mut SemanticProgramBuilder, a: Option<ProgramValue>, b: Option<ProgramValue>) -> Result<Option<ProgramValue>, SemanticAdError> {
    Ok(match (a, b) {
        (Some(a), Some(b)) => Some(builder.add_op(CoreSemanticOp::Add, &[a, b])?[0]),
        (a, None) => a,
        (None, b) => b,
    })
}

fn value(v: Option<ProgramValue>) -> AdValue {
    v.map_or(AdValue::Absent, AdValue::Value)
}

fn present(v: Option<&AdValue>) -> Option<ProgramValue> {
    v.and_then(|v| v.value())
}

/// Input cotangents of `op` for output cotangents `cts`, with `primal(k)` the primal inputs
/// (and `primal(input_count + k)` the outputs).
fn vjp(
    op: DenseOp,
    primal: &dyn Fn(usize) -> Result<ProgramValue, SemanticAdError>,
    cts: &[AdValue],
    active: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let mut result = vec![AdValue::Absent; op.input_count()];
    let act = op.act();
    match op {
        DenseOp::Dense { bias, .. } => {
            let Some(c) = present(cts.first()) else { return Ok(result.into_boxed_slice()) };
            let parts = Parts { w: active[0], x: active[1], b: bias && active[2] };
            let y = primal(op.input_count())?;
            let [w, x, b] = cotangent(builder, act, (y, primal(0)?, primal(1)?), c, parts)?;
            result[0] = value(w);
            result[1] = value(x);
            if bias {
                result[2] = value(b);
            }
        }
        DenseOp::Tangent { parts, .. } => {
            // M(dW x + W dx + db): bilinear in (dW, x) and (W, dx), linear in db.
            let Some(c) = present(cts.first()) else { return Ok(result.into_boxed_slice()) };
            let (y, w, x) = (primal(0)?, primal(1)?, primal(2)?);
            let mut k = 3;
            let mut index = |on: bool| on.then(|| { let i = k; k += 1; i });
            let (iw, ix, ib) = (index(parts.w), index(parts.x), index(parts.b));
            let lin = Parts {
                w: iw.is_some_and(|i| active[i]),
                x: ix.is_some_and(|i| active[i]),
                b: ib.is_some_and(|i| active[i]),
            };
            let [dw, dx, db] = cotangent(builder, act, (y, w, x), c, lin)?;
            for (index, ct) in [(iw, dw), (ix, dx), (ib, db)] {
                if let Some(i) = index {
                    result[i] = value(ct);
                }
            }
            // W̄ = M(c) dxᵀ through `W dx`; x̄ = dWᵀ M(c) through `dW x`.
            if let (true, Some(i)) = (active[1], ix) {
                result[1] = value(cotangent(builder, act, (y, w, primal(i)?), c, Parts { w: true, x: false, b: false })?[0]);
            }
            if let (true, Some(i)) = (active[2], iw) {
                result[2] = value(cotangent(builder, act, (y, primal(i)?, x), c, Parts { w: false, x: true, b: false })?[1]);
            }
        }
        DenseOp::Cotangent { parts, .. } => {
            // (M(c̄) xᵀ, Wᵀ M(c̄), Σ M(c̄)) for the last input c̄.
            let (y, w, x) = (primal(0)?, primal(1)?, primal(2)?);
            let c = primal(3)?;
            let mut outs = cts.iter();
            let mut next = |on: bool| if on { present(outs.next()) } else { None };
            let (gw, gx, gb) = (next(parts.w), next(parts.x), next(parts.b));
            if active[3] {
                result[3] = value(tangent(builder, act, (y, w, x), gw, gx, gb)?);
            }
            if let (true, Some(gx)) = (active[1], gx) {
                result[1] = value(cotangent(builder, act, (y, w, gx), c, Parts { w: true, x: false, b: false })?[0]);
            }
            if let (true, Some(gw)) = (active[2], gw) {
                result[2] = value(cotangent(builder, act, (y, gw, x), c, Parts { w: false, x: true, b: false })?[1]);
            }
        }
    }
    Ok(result.into_boxed_slice())
}

impl SemanticLinearizeRule for DenseRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let op = dense_op(request.op(), SemanticAdRuleRole::Linearize)?;
        let p = request.primal_inputs();
        let t = request.tangent_inputs();
        let act = op.act();
        // The mask `y > 0` has a zero derivative almost everywhere, so tangents of `y` drop out.
        let outputs: Vec<AdValue> = match op {
            DenseOp::Dense { bias, .. } => {
                let y = request.primal_outputs()[0];
                let db = if bias { present(t.get(2)) } else { None };
                vec![value(tangent(builder, act, (y, p[0], p[1]), present(t.first()), present(t.get(1)), db)?)]
            }
            DenseOp::Tangent { parts, .. } => {
                let (y, w, x) = (p[0], p[1], p[2]);
                let mut k = 3;
                let mut index = |on: bool| on.then(|| { let i = k; k += 1; i });
                let (iw, ix, ib) = (index(parts.w), index(parts.x), index(parts.b));
                let tw = iw.and_then(|i| present(t.get(i)));
                let tx = ix.and_then(|i| present(t.get(i)));
                let tb = ib.and_then(|i| present(t.get(i)));
                let mut sum = tangent(builder, act, (y, w, x), tw, tx, tb)?;
                // dW x' and W' dx.
                if let (Some(i), Some(x1)) = (iw, present(t.get(2))) {
                    let extra = tangent(builder, act, (y, w, x1), Some(p[i]), None, None)?;
                    sum = add(builder, sum, extra)?;
                }
                if let (Some(i), Some(w1)) = (ix, present(t.get(1))) {
                    let extra = tangent(builder, act, (y, w1, x), None, Some(p[i]), None)?;
                    sum = add(builder, sum, extra)?;
                }
                vec![value(sum)]
            }
            DenseOp::Cotangent { parts, .. } => {
                let (y, w, x, c) = (p[0], p[1], p[2], p[3]);
                let mut terms = match present(t.get(3)) {
                    Some(dc) => cotangent(builder, act, (y, w, x), dc, parts)?,
                    None => [None; 3],
                };
                if let (true, Some(x1)) = (parts.w, present(t.get(2))) {
                    let extra = cotangent(builder, act, (y, w, x1), c, Parts { w: true, x: false, b: false })?[0];
                    terms[0] = add(builder, terms[0], extra)?;
                }
                if let (true, Some(w1)) = (parts.x, present(t.get(1))) {
                    let extra = cotangent(builder, act, (y, w1, x), c, Parts { w: false, x: true, b: false })?[1];
                    terms[1] = add(builder, terms[1], extra)?;
                }
                [(parts.w, terms[0]), (parts.x, terms[1]), (parts.b, terms[2])]
                    .into_iter()
                    .filter(|(on, _)| *on)
                    .map(|(_, v)| value(v))
                    .collect()
            }
        };
        Ok(SemanticLinearizeResult::new(outputs, []))
    }
}

impl SemanticPrimalVjpRule for DenseRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::all_inputs().with_output(0)
    }
    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = dense_op(request.op(), SemanticAdRuleRole::PrimalVjp)?;
        let n = op.input_count();
        let primal = |k: usize| if k < n { request.primal_input_value(k) } else { request.primal_output_value(k - n) };
        vjp(op, &primal, request.cotangent_outputs(), request.active_inputs(), builder)
    }
}

impl SemanticLinearTransposeRule for DenseRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0).with_input(1).with_input(2)
    }
    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = dense_op(request.op(), SemanticAdRuleRole::LinearTranspose)?;
        let unsupported = |message: &str| SemanticAdError::Unsupported {
            family_id: FAMILY,
            role: SemanticAdRuleRole::LinearTranspose,
            message: message.into(),
        };
        let active = request.active_inputs();
        if matches!(op, DenseOp::Dense { .. }) || active[..3].iter().any(|&a| a) {
            return Err(unsupported("dense layers are linear only in their tangent or cotangent inputs"));
        }
        let residuals = (request.primal_input_value(0)?, request.primal_input_value(1)?, request.primal_input_value(2)?);
        let cts = request.cotangent_outputs();
        let mut result = vec![AdValue::Absent; op.input_count()];
        match op {
            DenseOp::Tangent { act, parts } => {
                let Some(c) = present(cts.first()) else { return Ok(result.into_boxed_slice()) };
                let mut k = 3;
                let mut index = |on: bool| on.then(|| { let i = k; k += 1; i });
                let (iw, ix, ib) = (index(parts.w), index(parts.x), index(parts.b));
                let lin = Parts {
                    w: iw.is_some_and(|i| active[i]),
                    x: ix.is_some_and(|i| active[i]),
                    b: ib.is_some_and(|i| active[i]),
                };
                let [dw, dx, db] = cotangent(builder, act, residuals, c, lin)?;
                for (index, ct) in [(iw, dw), (ix, dx), (ib, db)] {
                    if let Some(i) = index {
                        result[i] = value(ct);
                    }
                }
            }
            DenseOp::Cotangent { act, parts } => {
                if active[3] {
                    let mut outs = cts.iter();
                    let mut next = |on: bool| if on { present(outs.next()) } else { None };
                    let (gw, gx, gb) = (next(parts.w), next(parts.x), next(parts.b));
                    result[3] = value(tangent(builder, act, residuals, gw, gx, gb)?);
                }
            }
            DenseOp::Dense { .. } => unreachable!(),
        }
        Ok(result.into_boxed_slice())
    }
}

pub fn rules(
) -> Result<SemanticExtensionRuleSet, tenferro_ad::semantic_extension::SemanticExtensionRegistryError>
{
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(DenseRule))?
        .with_primal_vjp(Arc::new(DenseRule))?
        .with_linear_transpose(Arc::new(DenseRule))
}

/// Apply `act(W x [+ b])` in an eager session; `inputs` are `W, x[, b]`.
pub fn apply_eager(
    session: &mut tenferro_ad::EagerSession<'_>,
    op: DenseOp,
    inputs: &[&tenferro_ad::EagerTensor],
) -> Result<tenferro_ad::EagerTensor, tenferro_ad::Error> {
    let values = tenferro_ad::extension::apply_eager_with_targeted_extension_in_session(
        session,
        Arc::new(op),
        inputs,
        |target| {
            extension_module::<tenferro_cpu::CpuEngine>(target.engine_id).map_err(|e| {
                tenferro_runtime::Error::runtime_state_source(
                    "dense extension",
                    tenferro_runtime::ErrorPhase::Execution,
                    e,
                )
            })
        },
    )?;
    values.into_iter().next().ok_or_else(|| tenferro_ad::Error::Internal("missing dense output".into()))
}

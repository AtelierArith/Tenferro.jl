//! Native softmax cross-entropy on logits: `-s Σⱼ Σᵢ yᵢⱼ logsoftmax(ŷ)ᵢⱼ` in one pass.
//!
//! Composed from `logsoftmax`, a product and a sum, the loss and its gradient
//! take several full-size tensors and runtime instructions. Classes are the
//! first axis of the `c × n` logits `ŷ` and targets `y`; `s` is the
//! aggregation scale (`1/n` for the mean over columns, `1` for the sum). With
//! `p = softmax(ŷ)` per column and `Cⱼ = Σᵢ yᵢⱼ`, the members are:
//!
//! - `Xent(ŷ, y) = -s Σ y ⊙ logsoftmax(ŷ)` (a scalar),
//! - `Grad(ŷ, y, g) = s g (p C - y)` (the cotangent of `ŷ` for the cotangent `g` of the loss),
//! - `YGrad(ŷ, y, g) = -s g logsoftmax(ŷ)` (the cotangent of `y`),
//! - `Tangent(ŷ, y, [dŷ], [dy]) = Σ Grad(ŷ, y, 1) ⊙ dŷ + Σ YGrad(ŷ, y, 1) ⊙ dy` (the tangent map).
//!
//! First derivatives of the loss and derivatives of these maps in their linear
//! arguments are members again. Derivatives through `softmax` of the derivative
//! maps (second derivatives in `ŷ`, or through `y`'s column sums) are not
//! implemented and fail with an error instead of returning wrong values.

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
use tenferro_runtime::program::{ProgramValue, SemanticProgramBuilder};
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

use crate::kernels::{self, Elem, MultiKernelFn};

const FAMILY: &str = "tenferro-julia.xent.v1";

/// The members of the cross-entropy family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    /// `Xent(ŷ, y)`.
    Loss,
    /// `Grad(ŷ, y, g)`.
    Grad,
    /// `YGrad(ŷ, y, g)`.
    YGrad,
    /// `Tangent(ŷ, y, [dŷ], [dy])`.
    Tangent { yhat: bool, y: bool },
}

/// A member with its aggregation scale (kept as bits, so payloads hash and compare exactly).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct XentOp {
    pub kind: Kind,
    scale: u64,
}

impl XentOp {
    pub fn loss(scale: f64) -> Self {
        Self { kind: Kind::Loss, scale: scale.to_bits() }
    }
    fn with_kind(self, kind: Kind) -> Self {
        Self { kind, ..self }
    }
    fn scale(self) -> f64 {
        f64::from_bits(self.scale)
    }
}

impl ExtensionOp for XentOp {
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
        match self.kind {
            Kind::Loss => 2,
            Kind::Grad | Kind::YGrad => 3,
            Kind::Tangent { yhat, y } => 2 + yhat as usize + y as usize,
        }
    }
    fn output_count(&self) -> usize {
        1
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
        let fail = |message: &str| tenferro_tensor::Error::runtime_state("logit cross-entropy", message.to_string());
        let dtype = ctx.input_dtype(0)?;
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "logit cross-entropy",
                dtype,
                "logit cross-entropy supports Float32 and Float64",
            ));
        }
        for i in 0..self.input_count() {
            if ctx.input_dtype(i)? != dtype {
                return Err(fail("operands must have one element type"));
            }
        }
        if ctx.input_shape(0)?.len() != 2 {
            return Err(fail("logits must be a matrix"));
        }
        let matrix = |ctx: &mut ExtensionShapeContext<'_>, i: usize| -> tenferro_tensor::Result<()> {
            if ctx.input_shape(i)?.len() != 2 {
                return Err(fail("targets and tangents must have the shape of the logits"));
            }
            ctx.require_axes_equal((i, 0), (0, 0))?;
            ctx.require_axes_equal((i, 1), (0, 1))?;
            Ok(())
        };
        matrix(ctx, 1)?;
        match self.kind {
            Kind::Loss => Ok(vec![(dtype, vec![])]),
            Kind::Grad | Kind::YGrad => {
                if !ctx.input_shape(2)?.is_empty() {
                    return Err(fail("the loss cotangent must be a scalar"));
                }
                Ok(vec![(dtype, ctx.input_shape(0)?.to_vec())])
            }
            Kind::Tangent { yhat, y } => {
                if !(yhat || y) {
                    return Err(fail("a tangent needs at least one input"));
                }
                for i in 2..self.input_count() {
                    matrix(ctx, i)?;
                }
                Ok(vec![(dtype, vec![])])
            }
        }
    }
}

// Kernels ------------------------------------------------------------------------------

struct Kernel {
    op: XentOp,
    /// Classes per column.
    c: usize,
}

/// `log Σ exp(ŷ)` of one column, computed around its maximum.
fn lse<T: Elem>(col: &[T]) -> T {
    let m = col.iter().copied().fold(T::LOWEST, |a, v| if v > a { v } else { a });
    // Every entry is -Inf (or NaN): nothing to shift by.
    if m == T::LOWEST {
        return m;
    }
    let mut s = T::default();
    col.iter().for_each(|&v| s += (v - m).exp());
    m + s.ln()
}

impl MultiKernelFn for Kernel {
    fn run<T: Elem>(&self, inputs: &[&[T]], outs: &mut [Vec<T>], _: Option<threadteam::Exec<'_>>) {
        let c = self.c.max(1);
        let s = T::from_f64(self.op.scale());
        let (yhat, y) = (inputs[0], inputs[1]);
        let cols = yhat.chunks_exact(c).zip(y.chunks_exact(c));
        let out = &mut outs[0];
        match self.op.kind {
            Kind::Loss => {
                let mut total = T::default();
                for (h, t) in cols {
                    let l = lse(h);
                    h.iter().zip(t).for_each(|(&h, &t)| total += t * (h - l));
                }
                out[0] = T::default() - s * total;
            }
            Kind::Grad => {
                let g = s * inputs[2][0];
                for ((h, t), o) in cols.zip(out.chunks_exact_mut(c)) {
                    let l = lse(h);
                    let mut mass = T::default();
                    t.iter().for_each(|&t| mass += t);
                    for ((o, &h), &t) in o.iter_mut().zip(h).zip(t) {
                        *o = g * ((h - l).exp() * mass - t);
                    }
                }
            }
            Kind::YGrad => {
                let g = s * inputs[2][0];
                for ((h, _), o) in cols.zip(out.chunks_exact_mut(c)) {
                    let l = lse(h);
                    o.iter_mut().zip(h).for_each(|(o, &h)| *o = T::default() - g * (h - l));
                }
            }
            Kind::Tangent { yhat: has_dh, y: has_dy } => {
                let dh = has_dh.then(|| inputs[2]);
                let dy = has_dy.then(|| inputs[2 + has_dh as usize]);
                let mut total = T::default();
                for (j, (h, t)) in cols.enumerate() {
                    let l = lse(h);
                    let range = j * c..(j + 1) * c;
                    if let Some(dh) = dh {
                        let mut mass = T::default();
                        t.iter().for_each(|&t| mass += t);
                        for ((&h, &t), &d) in h.iter().zip(t).zip(&dh[range.clone()]) {
                            total += ((h - l).exp() * mass - t) * d;
                        }
                    }
                    if let Some(dy) = dy {
                        for (&h, &d) in h.iter().zip(&dy[range]) {
                            total += (l - h) * d;
                        }
                    }
                }
                out[0] = s * total;
            }
        }
    }
}

fn execute_xent(
    op: &XentOp,
    engine: &mut dyn Engine,
    _: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let shape = inputs[0].shape().to_vec();
    let c = shape[0];
    let out = match op.kind {
        Kind::Loss | Kind::Tangent { .. } => vec![],
        Kind::Grad | Kind::YGrad => shape,
    };
    kernels::execute_multi("logit cross-entropy", engine, inputs, vec![out], Kernel { op: *op, c })
}

fn supported<B: Engine + 'static>(_: &XentOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<tenferro_cpu::CpuEngine>()
}

define_extension_runtime! {
    runtime = XentRuntime,
    family_id = FAMILY,
    op_type = XentOp,
    execute_in_session = execute_xent,
    session_supported = supported,
    backend_bound = Engine,
}

// Derivative rules ---------------------------------------------------------------------

#[derive(Debug)]
struct XentRule;

fn xent_op(op: &dyn ExtensionOp, role: SemanticAdRuleRole) -> Result<XentOp, SemanticAdError> {
    op.as_any().downcast_ref::<XentOp>().copied().ok_or_else(|| SemanticAdError::Unsupported {
        family_id: FAMILY,
        role,
        message: "unexpected payload in the logit cross-entropy family".into(),
    })
}

fn unsupported(role: SemanticAdRuleRole) -> SemanticAdError {
    SemanticAdError::Unsupported {
        family_id: FAMILY,
        role,
        message: "derivatives of the logit cross-entropy gradient through softmax are not implemented".into(),
    }
}

fn emit(builder: &mut SemanticProgramBuilder, op: XentOp, inputs: &[ProgramValue]) -> Result<AdValue, SemanticAdError> {
    Ok(AdValue::Value(builder.add_extension(Arc::new(op), inputs)?[0]))
}

/// `Tangent(ŷ, y, dŷ, dy)` for the present directions.
fn tangent(
    builder: &mut SemanticProgramBuilder,
    op: XentOp,
    (yhat, y): (ProgramValue, ProgramValue),
    dh: Option<ProgramValue>,
    dy: Option<ProgramValue>,
) -> Result<AdValue, SemanticAdError> {
    if dh.is_none() && dy.is_none() {
        return Ok(AdValue::Absent);
    }
    let mut inputs = vec![yhat, y];
    inputs.extend(dh.into_iter().chain(dy));
    emit(builder, op.with_kind(Kind::Tangent { yhat: dh.is_some(), y: dy.is_some() }), &inputs)
}

/// The cotangents of the linear inputs of `op` (a derivative member) for `ct`; the
/// nonlinear inputs `ŷ, y` must not be differentiated.
fn linear_cotangents(
    op: XentOp,
    (yhat, y): (ProgramValue, ProgramValue),
    ct: ProgramValue,
    active: &[bool],
    role: SemanticAdRuleRole,
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let mut result = vec![AdValue::Absent; op.input_count()];
    match op.kind {
        Kind::Loss => return Err(unsupported(role)),
        Kind::Grad if active[2] => result[2] = tangent(builder, op, (yhat, y), Some(ct), None)?,
        Kind::YGrad if active[2] => result[2] = tangent(builder, op, (yhat, y), None, Some(ct))?,
        Kind::Grad | Kind::YGrad => {}
        Kind::Tangent { yhat: has_dh, y: has_dy } => {
            if has_dh && active[2] {
                result[2] = emit(builder, op.with_kind(Kind::Grad), &[yhat, y, ct])?;
            }
            let k = 2 + has_dh as usize;
            if has_dy && active[k] {
                result[k] = emit(builder, op.with_kind(Kind::YGrad), &[yhat, y, ct])?;
            }
        }
    }
    Ok(result.into_boxed_slice())
}

impl SemanticLinearizeRule for XentRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let role = SemanticAdRuleRole::Linearize;
        let op = xent_op(request.op(), role)?;
        let p = request.primal_inputs();
        let t: Vec<Option<ProgramValue>> = request.tangent_inputs().iter().map(|v| v.value()).collect();
        let output = match op.kind {
            Kind::Loss => tangent(builder, op, (p[0], p[1]), t[0], t[1])?,
            _ => {
                // Linear in the inputs after `ŷ, y` (`y` is unused by `YGrad`).
                let y_matters = !matches!(op.kind, Kind::YGrad);
                if t[0].is_some() || (y_matters && t[1].is_some()) {
                    return Err(unsupported(role));
                }
                let mut inputs = vec![p[0], p[1]];
                let mut any = false;
                let mut parts = Vec::new();
                for &tk in &t[2..op.input_count()] {
                    parts.push(tk.is_some());
                    if let Some(v) = tk {
                        inputs.push(v);
                        any = true;
                    }
                }
                if !any {
                    AdValue::Absent
                } else {
                    match op.kind {
                        Kind::Tangent { yhat, y } => {
                            // Keep the present directions of the tangent map.
                            let (dh, dy) = (yhat && parts[0], y && parts[usize::from(yhat)]);
                            emit(builder, op.with_kind(Kind::Tangent { yhat: dh, y: dy }), &inputs)?
                        }
                        _ => emit(builder, op, &inputs)?,
                    }
                }
            }
        };
        Ok(SemanticLinearizeResult::new([output], []))
    }
}

impl SemanticPrimalVjpRule for XentRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::all_inputs()
    }
    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let role = SemanticAdRuleRole::PrimalVjp;
        let op = xent_op(request.op(), role)?;
        let Some(ct) = request.cotangent_outputs()[0].value() else {
            return Ok(vec![AdValue::Absent; op.input_count()].into_boxed_slice());
        };
        let active = request.active_inputs();
        let primal = (request.primal_input_value(0)?, request.primal_input_value(1)?);
        if op.kind == Kind::Loss {
            let mut result = vec![AdValue::Absent; 2];
            if active[0] {
                result[0] = emit(builder, op.with_kind(Kind::Grad), &[primal.0, primal.1, ct])?;
            }
            if active[1] {
                result[1] = emit(builder, op.with_kind(Kind::YGrad), &[primal.0, primal.1, ct])?;
            }
            return Ok(result.into_boxed_slice());
        }
        let y_matters = !matches!(op.kind, Kind::YGrad);
        if active[0] || (y_matters && active[1]) {
            return Err(unsupported(role));
        }
        linear_cotangents(op, primal, ct, active, role, builder)
    }
}

impl SemanticLinearTransposeRule for XentRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0).with_input(1)
    }
    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let role = SemanticAdRuleRole::LinearTranspose;
        let op = xent_op(request.op(), role)?;
        let active = request.active_inputs();
        if op.kind == Kind::Loss || active[0] || active[1] {
            return Err(unsupported(role));
        }
        let Some(ct) = request.cotangent_outputs()[0].value() else {
            return Ok(vec![AdValue::Absent; op.input_count()].into_boxed_slice());
        };
        let primal = (request.primal_input_value(0)?, request.primal_input_value(1)?);
        linear_cotangents(op, primal, ct, active, role, builder)
    }
}

pub fn rules(
) -> Result<SemanticExtensionRuleSet, tenferro_ad::semantic_extension::SemanticExtensionRegistryError>
{
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(XentRule))?
        .with_primal_vjp(Arc::new(XentRule))?
        .with_linear_transpose(Arc::new(XentRule))
}

/// The loss `Xent(ŷ, y)` with scale `scale` in an eager session.
pub fn apply_eager(
    session: &mut tenferro_ad::EagerSession<'_>,
    scale: f64,
    yhat: &tenferro_ad::EagerTensor,
    y: &tenferro_ad::EagerTensor,
) -> Result<tenferro_ad::EagerTensor, tenferro_ad::Error> {
    let values = tenferro_ad::extension::apply_eager_with_targeted_extension_in_session(
        session,
        Arc::new(XentOp::loss(scale)),
        &[yhat, y],
        |target| {
            extension_module::<tenferro_cpu::CpuEngine>(target.engine_id).map_err(|e| {
                tenferro_runtime::Error::runtime_state_source(
                    "cross-entropy extension",
                    tenferro_runtime::ErrorPhase::Execution,
                    e,
                )
            })
        },
    )?;
    values.into_iter().next().ok_or_else(|| tenferro_ad::Error::Internal("missing cross-entropy output".into()))
}

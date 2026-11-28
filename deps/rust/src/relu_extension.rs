//! Native ReLU: `relu(x) = x > 0 ? x : 0` in one pass.
//!
//! Composed from `compare` and `select` with a constant zero, ReLU and its
//! derivative take several full-size passes and temporaries. `Relu(x)` and
//! `ReluMask(x, t) = x > 0 ? t : 0` form a closed family: the derivative of
//! `Relu` is `ReluMask(x, ·)` (zero at `x == 0`, like NNlib), which is linear in
//! `t` and its own transpose, so derivatives of any order stay native.

use std::any::Any;
use std::hash::{Hash, Hasher};
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticAdRuleRole, SemanticExtensionRuleSet,
    SemanticLinearTransposeRequest, SemanticLinearTransposeRule, SemanticLinearizeRequest,
    SemanticLinearizeResult, SemanticLinearizeRule, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_runtime::extension::{
    define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp,
    ExtensionShapeContext,
};
use tenferro_runtime::program::{ProgramValue, SemanticProgramBuilder};
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

use tenferro_cpu::threadteam;

use crate::kernels::{self, par_blocks, Elem, KernelFn};

const FAMILY: &str = "tenferro-julia.relu.v1";

/// `Relu(x)` (`mask == false`) or `ReluMask(x, t)` (`mask == true`).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct ReluOp {
    pub mask: bool,
}

impl ExtensionOp for ReluOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, mut state: &mut dyn Hasher) {
        self.mask.hash(&mut state);
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
        1 + self.mask as usize
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
        let dtype = ctx.input_dtype(0)?;
        if self.mask {
            ctx.require_same_shape(0, 1)?;
            if ctx.input_dtype(1)? != dtype {
                return Err(tenferro_tensor::Error::runtime_state("relu", "operands must have one element type"));
            }
        }
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(tenferro_tensor::Error::unsupported_dtype("relu", dtype, "relu supports Float32 and Float64"));
        }
        Ok(vec![(dtype, ctx.input_shape(0)?.to_vec())])
    }
}

impl KernelFn for ReluOp {
    fn run<T: Elem>(&self, inputs: &[&[T]], out: &mut [T], exec: Option<threadteam::Exec<'_>>) {
        const BLOCK: usize = 1 << 14;
        let x = inputs[0];
        let t = inputs.get(1).copied();
        let zero = T::default();
        let len = out.len();
        par_blocks(exec, out, BLOCK, len, |first, out| {
            let (start, end) = (first * BLOCK, first * BLOCK + out.len());
            let xs = &x[start..end];
            match t {
                None => out.iter_mut().zip(xs).for_each(|(o, &v)| *o = if v > zero { v } else { zero }),
                Some(t) => out
                    .iter_mut()
                    .zip(xs.iter().zip(&t[start..end]))
                    .for_each(|(o, (&v, &w))| *o = if v > zero { w } else { zero }),
            }
        });
    }
}

fn execute_relu(
    op: &ReluOp,
    engine: &mut dyn Engine,
    _: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    let shape = inputs[0].shape().to_vec();
    kernels::execute("relu", engine, inputs, shape, *op)
}

fn supported<B: Engine + 'static>(_: &ReluOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<tenferro_cpu::CpuEngine>()
}

define_extension_runtime! {
    runtime = ReluRuntime,
    family_id = FAMILY,
    op_type = ReluOp,
    execute_in_session = execute_relu,
    session_supported = supported,
    backend_bound = Engine,
}

#[derive(Debug)]
struct ReluRule;

fn masked(builder: &mut SemanticProgramBuilder, x: ProgramValue, t: ProgramValue) -> Result<AdValue, SemanticAdError> {
    Ok(AdValue::Value(builder.add_extension(Arc::new(ReluOp { mask: true }), &[x, t])?[0]))
}

fn relu_op(op: &dyn ExtensionOp, role: SemanticAdRuleRole) -> Result<ReluOp, SemanticAdError> {
    op.as_any().downcast_ref::<ReluOp>().copied().ok_or_else(|| SemanticAdError::Unsupported {
        family_id: FAMILY,
        role,
        message: "unexpected payload in the relu family".into(),
    })
}

/// Cotangents of the inputs: the masked cotangent for the last input; `x` only
/// selects entries (a zero derivative almost everywhere).
fn cotangents(
    op: ReluOp,
    x: ProgramValue,
    ct: AdValue,
    active: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let mut result = vec![AdValue::Absent; op.input_count()];
    let last = result.len() - 1;
    if let (AdValue::Value(ct), true) = (ct, active[last]) {
        result[last] = masked(builder, x, ct)?;
    }
    Ok(result.into_boxed_slice())
}

impl SemanticLinearizeRule for ReluRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        relu_op(request.op(), SemanticAdRuleRole::Linearize)?;
        let output = match request.tangent_inputs().last().copied() {
            Some(AdValue::Value(dt)) => masked(builder, request.primal_inputs()[0], dt)?,
            _ => AdValue::Absent,
        };
        Ok(SemanticLinearizeResult::new([output], []))
    }
}

impl SemanticPrimalVjpRule for ReluRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0)
    }
    fn primal_vjp(
        &self,
        request: SemanticPrimalVjpRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = relu_op(request.op(), SemanticAdRuleRole::PrimalVjp)?;
        if matches!(request.cotangent_outputs()[0], AdValue::Absent) {
            return Ok(vec![AdValue::Absent; op.input_count()].into_boxed_slice());
        }
        let x = request.primal_input_value(0)?;
        cotangents(op, x, request.cotangent_outputs()[0], request.active_inputs(), builder)
    }
}

impl SemanticLinearTransposeRule for ReluRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn residual_mask(&self) -> ResidualSpec {
        ResidualSpec::input(0)
    }
    fn linear_transpose(
        &self,
        request: SemanticLinearTransposeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<Box<[AdValue]>, SemanticAdError> {
        let op = relu_op(request.op(), SemanticAdRuleRole::LinearTranspose)?;
        if !op.mask {
            return Err(SemanticAdError::Unsupported {
                family_id: FAMILY,
                role: SemanticAdRuleRole::LinearTranspose,
                message: "relu is not linear".into(),
            });
        }
        if matches!(request.cotangent_outputs()[0], AdValue::Absent) {
            return Ok(vec![AdValue::Absent; 2].into_boxed_slice());
        }
        let x = request.primal_input_value(0)?;
        cotangents(op, x, request.cotangent_outputs()[0], request.active_inputs(), builder)
    }
}

pub fn rules(
) -> Result<SemanticExtensionRuleSet, tenferro_ad::semantic_extension::SemanticExtensionRegistryError>
{
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(ReluRule))?
        .with_primal_vjp(Arc::new(ReluRule))?
        .with_linear_transpose(Arc::new(ReluRule))
}

/// Apply ReLU in an eager session.
pub fn apply_eager(
    session: &mut tenferro_ad::EagerSession<'_>,
    input: &tenferro_ad::EagerTensor,
) -> Result<tenferro_ad::EagerTensor, tenferro_ad::Error> {
    let values = tenferro_ad::extension::apply_eager_with_targeted_extension_in_session(
        session,
        Arc::new(ReluOp { mask: false }),
        &[input],
        |target| {
            extension_module::<tenferro_cpu::CpuEngine>(target.engine_id).map_err(|e| {
                tenferro_runtime::Error::runtime_state_source(
                    "relu extension",
                    tenferro_runtime::ErrorPhase::Execution,
                    e,
                )
            })
        },
    )?;
    values.into_iter().next().ok_or_else(|| tenferro_ad::Error::Internal("missing relu output".into()))
}

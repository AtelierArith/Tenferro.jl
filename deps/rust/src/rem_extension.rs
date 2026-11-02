use std::any::Any;
use std::hash::Hasher;
use std::sync::Arc;

use tenferro_ad::semantic_extension::{
    AdValue, ResidualSpec, SemanticAdError, SemanticExtensionRuleSet, SemanticLinearizeRequest,
    SemanticLinearizeResult, SemanticLinearizeRule, SemanticPrimalVjpRequest,
    SemanticPrimalVjpRule,
};
use tenferro_runtime::extension::{
    define_extension_runtime, ExtensionAliasDeclaration, ExtensionEffectDeclaration, ExtensionOp,
    ExtensionShapeContext,
};
use tenferro_runtime::program::{CoreSemanticOp, ProgramValue, SemanticProgramBuilder};
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

const FAMILY: &str = "tenferro-julia.remainder.v1";

#[derive(Clone, Debug)]
pub struct Remainder;
impl ExtensionOp for Remainder {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, _: &mut dyn Hasher) {}
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other.as_any().is::<Self>()
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        2
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
        ctx.require_same_shape(0, 1)?;
        Ok(vec![(ctx.input_dtype(0)?, ctx.input_shape(0)?.to_vec())])
    }
}

fn execute_rem(
    _: &Remainder,
    engine: &mut dyn Engine,
    _: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    Ok(vec![engine.rem_read(inputs[0].clone(), inputs[1].clone())?])
}
fn supported<B: Engine + 'static>(_: &Remainder) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<tenferro_cpu::CpuEngine>()
}

define_extension_runtime! {
    runtime = RemainderRuntime,
    family_id = FAMILY,
    op_type = Remainder,
    execute_in_session = execute_rem,
    session_supported = supported,
    backend_bound = Engine,
}

#[derive(Debug)]
struct RemainderRule;

fn quotient(
    builder: &mut SemanticProgramBuilder,
    x: ProgramValue,
    y: ProgramValue,
    r: ProgramValue,
) -> Result<ProgramValue, SemanticAdError> {
    let diff = builder.add_op(CoreSemanticOp::Sub, &[x, r])?[0];
    Ok(builder.add_op(CoreSemanticOp::Div, &[diff, y])?[0])
}

impl SemanticLinearizeRule for RemainderRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let output = match request.tangent_inputs() {
            [AdValue::Value(dx), AdValue::Absent] => AdValue::Value(*dx),
            [dx, AdValue::Value(dy)] => {
                let q = quotient(
                    builder,
                    request.primal_inputs()[0],
                    request.primal_inputs()[1],
                    request.primal_outputs()[0],
                )?;
                let product = builder.add_op(CoreSemanticOp::Mul, &[q, *dy])?[0];
                AdValue::Value(match dx {
                    AdValue::Value(dx) => builder.add_op(CoreSemanticOp::Sub, &[*dx, product])?[0],
                    AdValue::Absent => builder.add_op(CoreSemanticOp::Neg, &[product])?[0],
                })
            }
            [AdValue::Absent, AdValue::Absent] => AdValue::Absent,
            _ => unreachable!(),
        };
        Ok(SemanticLinearizeResult::new([output], []))
    }
}
impl SemanticPrimalVjpRule for RemainderRule {
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
        let AdValue::Value(ct) = request.cotangent_outputs()[0] else {
            return Ok(vec![AdValue::Absent; 2].into_boxed_slice());
        };
        let first = if request.active_inputs()[0] {
            AdValue::Value(ct)
        } else {
            AdValue::Absent
        };
        let second = if request.active_inputs()[1] {
            let x = request.primal_input_value(0)?;
            let y = request.primal_input_value(1)?;
            let r = builder.add_op(CoreSemanticOp::Rem, &[x, y])?[0];
            let q = quotient(builder, x, y, r)?;
            let product = builder.add_op(CoreSemanticOp::Mul, &[q, ct])?[0];
            AdValue::Value(builder.add_op(CoreSemanticOp::Neg, &[product])?[0])
        } else {
            AdValue::Absent
        };
        Ok(vec![first, second].into_boxed_slice())
    }
}

pub fn rules(
) -> Result<SemanticExtensionRuleSet, tenferro_ad::semantic_extension::SemanticExtensionRegistryError>
{
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(RemainderRule))?
        .with_primal_vjp(Arc::new(RemainderRule))
}

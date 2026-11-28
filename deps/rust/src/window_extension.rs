//! Native sliding-window kernels for convolution and pooling.
//!
//! tenferro-rs has no convolution or `reduce_window` primitive. Composing them
//! from pad / strided-slice / concat / select operations costs one tensor per
//! kernel offset, so this module adds one extension family whose members gather
//! all windows in a single pass:
//!
//! - `Im2col` copies every window of `x` into a column tensor (convolution then
//!   is one `dot_general`), and `Col2im` is its transpose (scatter-add).
//! - `SumPool` sums every window (mean and Lp pooling build on it), and
//!   `SumPoolT` is its transpose.
//! - `MaxPool` takes the first maximum of every window, scanning the kernel
//!   offsets in column-major order and replacing the running maximum only by a
//!   strictly larger entry (NNlib's rule). `MaxPoolPick(x, t)` gathers `t` at the
//!   chosen entries (the tangent map) and `MaxPoolBack(x, c)` scatters `c` to
//!   them (the cotangent map).
//!
//! Every derivative of a member is expressed with members again (the family is
//! closed), so derivatives of any order run on these kernels.
//!
//! Geometry, per spatial dimension `d` of `x = (n..., rest...)`: window `o`
//! reads position `p = o * stride + k' * dilation - lo` of the input dilated with
//! `interior` zeros between entries, where `k' = kernel - 1 - k` when `flip` and
//! `k` otherwise. Positions outside the dilated input or between its entries are
//! padding: zero for the linear kernels and `-inf`-like (never chosen unless the
//! first) for `MaxPool`, exactly as if `x` had been padded with `typemin`.
//! Column tensors are laid out `(kernel..., rest[..split]..., out..., rest[split..]...)`
//! and pooled tensors `(out..., rest...)`.

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
use tenferro_cpu::threadteam;
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

use crate::kernels::{self, par_blocks, Elem, KernelFn};

const FAMILY: &str = "tenferro-julia.window.v1";

/// Window position of a padding entry.
const PAD: u32 = u32::MAX;

/// The members of the window family.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Kind {
    Im2col,
    Col2im,
    SumPool,
    SumPoolT,
    MaxPool,
    MaxPoolPick,
    MaxPoolBack,
}

impl Kind {
    fn from_code(code: i32) -> Result<Self, String> {
        Ok(match code {
            0 => Self::Im2col,
            1 => Self::Col2im,
            2 => Self::SumPool,
            3 => Self::SumPoolT,
            4 => Self::MaxPool,
            5 => Self::MaxPoolPick,
            6 => Self::MaxPoolBack,
            _ => return Err(format!("unknown window kernel {code}")),
        })
    }
    fn input_count(self) -> usize {
        match self {
            Self::MaxPoolPick | Self::MaxPoolBack => 2,
            _ => 1,
        }
    }
}

/// Window geometry shared by every member (see the module documentation).
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Geometry {
    /// Full shape of the windowed tensor `x`: spatial dimensions, then the rest.
    xshape: Vec<usize>,
    kernel: Vec<usize>,
    stride: Vec<usize>,
    dilation: Vec<usize>,
    lo: Vec<i64>,
    out: Vec<usize>,
    interior: Vec<usize>,
    flip: bool,
    /// Number of `rest` dimensions placed before the output dimensions in columns.
    split: usize,
}

impl Geometry {
    /// Decode `[S, rank, split, flip, xshape(rank), kernel(S), stride(S), dilation(S),
    /// lo(S), out(S), interior(S)]`.
    pub fn decode(v: &[i64]) -> Result<Self, String> {
        let err = || "malformed window geometry".to_string();
        let s = usize::try_from(*v.first().ok_or_else(err)?).map_err(|_| err())?;
        let rank = usize::try_from(*v.get(1).ok_or_else(err)?).map_err(|_| err())?;
        if v.len() != 4 + rank + 6 * s || s == 0 || s > rank {
            return Err(err());
        }
        let nonneg = |x: &[i64]| -> Result<Vec<usize>, String> {
            x.iter().map(|&a| usize::try_from(a).map_err(|_| err())).collect()
        };
        let split = usize::try_from(v[2]).map_err(|_| err())?;
        let flip = v[3] != 0;
        let mut at = 4;
        let mut take = |n: usize| {
            let part = &v[at..at + n];
            at += n;
            part
        };
        let xshape = nonneg(take(rank))?;
        let kernel = nonneg(take(s))?;
        let stride = nonneg(take(s))?;
        let dilation = nonneg(take(s))?;
        let lo = take(s).to_vec();
        let out = nonneg(take(s))?;
        let interior = nonneg(take(s))?;
        if split > rank - s
            || kernel.iter().chain(&stride).chain(&dilation).any(|&a| a == 0)
            || xshape[..s].iter().product::<usize>() >= PAD as usize
        {
            return Err(err());
        }
        Ok(Self { xshape, kernel, stride, dilation, lo, out, interior, flip, split })
    }

    fn s(&self) -> usize {
        self.kernel.len()
    }
    fn rest(&self) -> &[usize] {
        &self.xshape[self.s()..]
    }
    fn cols_shape(&self) -> Vec<usize> {
        let (r1, r2) = self.rest().split_at(self.split);
        [&self.kernel[..], r1, &self.out[..], r2].concat()
    }
    fn pool_shape(&self) -> Vec<usize> {
        [&self.out[..], self.rest()].concat()
    }
    fn spatial_len(&self) -> usize {
        self.xshape[..self.s()].iter().product()
    }
    fn kernel_len(&self) -> usize {
        self.kernel.iter().product()
    }
    fn out_len(&self) -> usize {
        self.out.iter().product()
    }
    /// `(prod(rest[..split]), prod(rest[split..]))`.
    fn rest_lens(&self) -> (usize, usize) {
        let (r1, r2) = self.rest().split_at(self.split);
        (r1.iter().product(), r2.iter().product())
    }

    /// Linear spatial source of kernel offset `k` in window `o`, `k + K * o`
    /// (both column-major multi-indices), or `PAD` for padding.
    fn positions(&self) -> Vec<u32> {
        let s = self.s();
        // Per dimension: source index of (k, o), or -1.
        let tables: Vec<Vec<i64>> = (0..s)
            .map(|d| {
                let (kn, on, n) = (self.kernel[d], self.out[d], self.xshape[d] as i64);
                let step = self.interior[d] as i64 + 1;
                let mut t = vec![-1i64; kn * on];
                for o in 0..on {
                    for k in 0..kn {
                        let kk = if self.flip { kn - 1 - k } else { k };
                        let p = (o * self.stride[d] + kk * self.dilation[d]) as i64 - self.lo[d];
                        if p >= 0 && p % step == 0 && p / step < n {
                            t[k + kn * o] = p / step;
                        }
                    }
                }
                t
            })
            .collect();
        let (kt, ot) = (self.kernel_len(), self.out_len());
        let mut pos = vec![PAD; kt * ot];
        let mut kidx = vec![0usize; s];
        let mut oidx = vec![0usize; s];
        for o in 0..ot {
            let mut r = o;
            for d in 0..s {
                oidx[d] = r % self.out[d];
                r /= self.out[d];
            }
            for k in 0..kt {
                let mut r = k;
                for d in 0..s {
                    kidx[d] = r % self.kernel[d];
                    r /= self.kernel[d];
                }
                let mut lin = 0i64;
                let mut stride = 1i64;
                for d in 0..s {
                    let src = tables[d][kidx[d] + self.kernel[d] * oidx[d]];
                    if src < 0 {
                        lin = -1;
                        break;
                    }
                    lin += src * stride;
                    stride *= self.xshape[d] as i64;
                }
                if lin >= 0 {
                    pos[k + kt * o] = lin as u32;
                }
            }
        }
        pos
    }
}

/// One member of the window family with its geometry.
#[derive(Clone, Debug)]
pub struct WindowOp {
    kind: Kind,
    geom: Arc<Geometry>,
}

impl WindowOp {
    pub fn new(code: i32, geometry: &[i64]) -> Result<Self, String> {
        Ok(Self { kind: Kind::from_code(code)?, geom: Arc::new(Geometry::decode(geometry)?) })
    }
    fn with_kind(&self, kind: Kind) -> Arc<dyn ExtensionOp> {
        Arc::new(Self { kind, geom: self.geom.clone() })
    }
    /// Shapes of the inputs, then of the output.
    fn shapes(&self) -> (Vec<Vec<usize>>, Vec<usize>) {
        let g = &self.geom;
        let (x, cols, pool) = (g.xshape.clone(), g.cols_shape(), g.pool_shape());
        match self.kind {
            Kind::Im2col => (vec![x], cols),
            Kind::Col2im => (vec![cols], x),
            Kind::SumPool | Kind::MaxPool => (vec![x], pool),
            Kind::SumPoolT => (vec![pool], x),
            Kind::MaxPoolPick => (vec![x.clone(), x], pool),
            Kind::MaxPoolBack => (vec![x, pool], g.xshape.clone()),
        }
    }
}

impl ExtensionOp for WindowOp {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn payload_hash(&self, mut state: &mut dyn Hasher) {
        self.kind.hash(&mut state);
        self.geom.hash(&mut state);
    }
    fn payload_eq(&self, other: &dyn ExtensionOp) -> bool {
        other
            .as_any()
            .downcast_ref::<Self>()
            .is_some_and(|o| o.kind == self.kind && o.geom == self.geom)
    }
    fn clone_arc(&self) -> Arc<dyn ExtensionOp> {
        Arc::new(self.clone())
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn input_count(&self) -> usize {
        self.kind.input_count()
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
        let (inputs, output) = self.shapes();
        let dtype = ctx.input_dtype(0)?;
        for (i, expected) in inputs.iter().enumerate() {
            let shape = ctx.input_shape(i)?;
            let matches = shape.len() == expected.len()
                && shape.iter().zip(expected).all(|(a, &b)| a.constant_value().is_none_or(|a| a == b));
            if !matches || ctx.input_dtype(i)? != dtype {
                return Err(tenferro_tensor::Error::runtime_state(
                    "window kernel",
                    format!("input {i} does not match the window geometry"),
                ));
            }
        }
        if !matches!(dtype, DType::F32 | DType::F64) {
            return Err(tenferro_tensor::Error::unsupported_dtype(
                "window kernel",
                dtype,
                "window kernels support Float32 and Float64",
            ));
        }
        Ok(vec![(dtype, output.into_iter().map(tenferro_runtime::SymDim::from).collect())])
    }
}

// Kernels ------------------------------------------------------------------------------

/// The window positions of `g`, cached for recently used geometries.
fn positions(g: &Geometry) -> Arc<Vec<u32>> {
    type Cache = std::sync::Mutex<Vec<(Geometry, Arc<Vec<u32>>)>>;
    static CACHE: Cache = std::sync::Mutex::new(Vec::new());
    let mut cache = CACHE.lock().unwrap_or_else(|p| p.into_inner());
    if let Some(i) = cache.iter().position(|(k, _)| k == g) {
        let entry = cache.remove(i);
        let pos = entry.1.clone();
        cache.push(entry);
        return pos;
    }
    let pos = Arc::new(g.positions());
    if cache.len() >= 32 {
        cache.remove(0);
    }
    cache.push((g.clone(), pos.clone()));
    pos
}

/// Position `p` of the first maximum in `window` of `xr` (`None` for padding)
/// and its value: a later entry replaces the running maximum only if larger.
#[inline]
fn first_max<T: Elem>(window: &[u32], xr: &[T]) -> (T, Option<usize>) {
    let value = |p: u32| if p == PAD { T::LOWEST } else { xr[p as usize] };
    let mut best = value(window[0]);
    let mut at = window[0];
    for &p in &window[1..] {
        let v = value(p);
        if v > best {
            best = v;
            at = p;
        }
    }
    (best, (at != PAD).then_some(at as usize))
}

struct WindowKernel<'a>(&'a WindowOp);

impl KernelFn for WindowKernel<'_> {
    fn run<T: Elem>(&self, inputs: &[&[T]], out: &mut [T], exec: Option<threadteam::Exec<'_>>) {
        run(self.0, inputs, out, exec)
    }
}

/// Write the output of `op` into the zeroed `out`.
fn run<T: Elem>(op: &WindowOp, inputs: &[&[T]], out: &mut [T], exec: Option<threadteam::Exec<'_>>) {
    let g = &*op.geom;
    let pos = positions(g);
    let pos = &pos[..];
    let (kt, ot, n) = (g.kernel_len(), g.out_len(), g.spatial_len());
    let (r1n, r2n) = g.rest_lens();
    let rn = r1n * r2n;
    if n == 0 || rn == 0 {
        // Empty inputs or outputs: every window is padding.
        return;
    }
    let windows = || pos.chunks_exact(kt);
    match op.kind {
        Kind::Im2col => {
            let x = inputs[0];
            par_blocks(exec, out, kt * r1n * ot, kt * rn * ot, |first, out| {
                for (b, out) in out.chunks_exact_mut(kt * r1n * ot).enumerate() {
                    let xb = &x[n * r1n * (first + b)..][..n * r1n];
                    for (window, out) in windows().zip(out.chunks_exact_mut(kt * r1n)) {
                        for (xr, out) in xb.chunks_exact(n).zip(out.chunks_exact_mut(kt)) {
                            for (o, &p) in out.iter_mut().zip(window) {
                                if p != PAD {
                                    *o = xr[p as usize];
                                }
                            }
                        }
                    }
                }
            });
        }
        Kind::Col2im => {
            let cols = inputs[0];
            par_blocks(exec, out, n * r1n, kt * rn * ot, |first, out| {
                for (b, xb) in out.chunks_exact_mut(n * r1n).enumerate() {
                    let cb = &cols[kt * r1n * ot * (first + b)..][..kt * r1n * ot];
                    for (window, cb) in windows().zip(cb.chunks_exact(kt * r1n)) {
                        for (xr, c) in xb.chunks_exact_mut(n).zip(cb.chunks_exact(kt)) {
                            for (&v, &p) in c.iter().zip(window) {
                                if p != PAD {
                                    xr[p as usize] += v;
                                }
                            }
                        }
                    }
                }
            });
        }
        Kind::SumPool => {
            let x = inputs[0];
            par_blocks(exec, out, ot, kt * rn * ot, |first, out| {
                for (r, yr) in out.chunks_exact_mut(ot).enumerate() {
                    let xr = &x[n * (first + r)..][..n];
                    for (window, y) in windows().zip(yr) {
                        let mut acc = T::default();
                        for &p in window {
                            if p != PAD {
                                acc += xr[p as usize];
                            }
                        }
                        *y = acc;
                    }
                }
            });
        }
        Kind::SumPoolT => {
            let y = inputs[0];
            par_blocks(exec, out, n, kt * rn * ot, |first, out| {
                for (r, xr) in out.chunks_exact_mut(n).enumerate() {
                    let yr = &y[ot * (first + r)..][..ot];
                    for (window, &v) in windows().zip(yr) {
                        for &p in window {
                            if p != PAD {
                                xr[p as usize] += v;
                            }
                        }
                    }
                }
            });
        }
        Kind::MaxPool | Kind::MaxPoolPick => {
            let x = inputs[0];
            let pick = (op.kind == Kind::MaxPoolPick).then(|| inputs[1]);
            par_blocks(exec, out, ot, kt * rn * ot, |first, out| {
                for (r, yr) in out.chunks_exact_mut(ot).enumerate() {
                    let xr = &x[n * (first + r)..][..n];
                    for (window, y) in windows().zip(yr) {
                        let (best, at) = first_max(window, xr);
                        *y = match pick {
                            None => best,
                            Some(t) => at.map_or(T::default(), |p| t[n * (first + r) + p]),
                        };
                    }
                }
            });
        }
        Kind::MaxPoolBack => {
            let (x, c) = (inputs[0], inputs[1]);
            par_blocks(exec, out, n, kt * rn * ot, |first, out| {
                for (r, dxr) in out.chunks_exact_mut(n).enumerate() {
                    let xr = &x[n * (first + r)..][..n];
                    let cr = &c[ot * (first + r)..][..ot];
                    for (window, &v) in windows().zip(cr) {
                        if let (_, Some(p)) = first_max(window, xr) {
                            dxr[p] += v;
                        }
                    }
                }
            });
        }
    }
}

fn execute_window(
    op: &WindowOp,
    engine: &mut dyn Engine,
    _: &mut tenferro_runtime::ExtensionCacheStore,
    inputs: &[TensorRead<'_>],
) -> tenferro_tensor::Result<Vec<Tensor>> {
    kernels::execute("window kernel", engine, inputs, op.shapes().1, WindowKernel(op))
}

fn supported<B: Engine + 'static>(_: &WindowOp) -> bool {
    std::any::TypeId::of::<B>() == std::any::TypeId::of::<tenferro_cpu::CpuEngine>()
}

define_extension_runtime! {
    runtime = WindowRuntime,
    family_id = FAMILY,
    op_type = WindowOp,
    execute_in_session = execute_window,
    session_supported = supported,
    backend_bound = Engine,
}

// Derivative rules -------------------------------------------------------------------

#[derive(Debug)]
struct WindowRule;

fn window_op(op: &dyn ExtensionOp, role: SemanticAdRuleRole) -> Result<&WindowOp, SemanticAdError> {
    op.as_any().downcast_ref::<WindowOp>().ok_or_else(|| SemanticAdError::Unsupported {
        family_id: FAMILY,
        role,
        message: "unexpected payload in the window family".into(),
    })
}

fn emit(
    builder: &mut SemanticProgramBuilder,
    op: Arc<dyn ExtensionOp>,
    inputs: &[ProgramValue],
) -> Result<AdValue, SemanticAdError> {
    Ok(AdValue::Value(builder.add_extension(op, inputs)?[0]))
}

/// Cotangents of the inputs of `op` (the transpose of its tangent map, which is
/// linear in the last input; `MaxPool*` treat `x` as constant).
fn cotangents(
    op: &WindowOp,
    x: impl FnOnce() -> Result<ProgramValue, SemanticAdError>,
    ct: AdValue,
    active: &[bool],
    builder: &mut SemanticProgramBuilder,
) -> Result<Box<[AdValue]>, SemanticAdError> {
    let mut result = vec![AdValue::Absent; op.kind.input_count()];
    let AdValue::Value(ct) = ct else {
        return Ok(result.into_boxed_slice());
    };
    let last = result.len() - 1;
    if !active[last] {
        return Ok(result.into_boxed_slice());
    }
    result[last] = match op.kind {
        Kind::Im2col => emit(builder, op.with_kind(Kind::Col2im), &[ct])?,
        Kind::Col2im => emit(builder, op.with_kind(Kind::Im2col), &[ct])?,
        Kind::SumPool => emit(builder, op.with_kind(Kind::SumPoolT), &[ct])?,
        Kind::SumPoolT => emit(builder, op.with_kind(Kind::SumPool), &[ct])?,
        Kind::MaxPool | Kind::MaxPoolPick => emit(builder, op.with_kind(Kind::MaxPoolBack), &[x()?, ct])?,
        Kind::MaxPoolBack => emit(builder, op.with_kind(Kind::MaxPoolPick), &[x()?, ct])?,
    };
    Ok(result.into_boxed_slice())
}

impl SemanticLinearizeRule for WindowRule {
    fn family_id(&self) -> &'static str {
        FAMILY
    }
    fn linearize(
        &self,
        request: SemanticLinearizeRequest<'_>,
        builder: &mut SemanticProgramBuilder,
    ) -> Result<SemanticLinearizeResult, SemanticAdError> {
        let op = window_op(request.op(), SemanticAdRuleRole::Linearize)?;
        let tangents = request.tangent_inputs();
        let x = request.primal_inputs()[0];
        // The tangent map is linear in the last input; `x` of `MaxPoolPick` and
        // `MaxPoolBack` only selects entries (a zero derivative almost everywhere).
        let output = match (op.kind, tangents.last().copied().unwrap_or(AdValue::Absent)) {
            (_, AdValue::Absent) => AdValue::Absent,
            (Kind::MaxPool, AdValue::Value(dx)) => emit(builder, op.with_kind(Kind::MaxPoolPick), &[x, dx])?,
            (Kind::MaxPoolPick | Kind::MaxPoolBack, AdValue::Value(dt)) => {
                emit(builder, request.op().clone_arc(), &[x, dt])?
            }
            (_, AdValue::Value(dx)) => emit(builder, request.op().clone_arc(), &[dx])?,
        };
        Ok(SemanticLinearizeResult::new([output], []))
    }
}

impl SemanticPrimalVjpRule for WindowRule {
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
        let op = window_op(request.op(), SemanticAdRuleRole::PrimalVjp)?;
        cotangents(op, || request.primal_input_value(0), request.cotangent_outputs()[0], request.active_inputs(), builder)
    }
}

impl SemanticLinearTransposeRule for WindowRule {
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
        let op = window_op(request.op(), SemanticAdRuleRole::LinearTranspose)?;
        if op.kind == Kind::MaxPool {
            return Err(SemanticAdError::Unsupported {
                family_id: FAMILY,
                role: SemanticAdRuleRole::LinearTranspose,
                message: "max pooling is not linear".into(),
            });
        }
        cotangents(op, || request.primal_input_value(0), request.cotangent_outputs()[0], request.active_inputs(), builder)
    }
}

pub fn rules(
) -> Result<SemanticExtensionRuleSet, tenferro_ad::semantic_extension::SemanticExtensionRegistryError>
{
    SemanticExtensionRuleSet::new()
        .with_linearize(Arc::new(WindowRule))?
        .with_primal_vjp(Arc::new(WindowRule))?
        .with_linear_transpose(Arc::new(WindowRule))
}

/// Apply a window kernel in an eager session.
pub fn apply_eager(
    session: &mut tenferro_ad::EagerSession<'_>,
    op: WindowOp,
    inputs: &[&tenferro_ad::EagerTensor],
) -> Result<tenferro_ad::EagerTensor, tenferro_ad::Error> {
    let values = tenferro_ad::extension::apply_eager_with_targeted_extension_in_session(
        session,
        Arc::new(op),
        inputs,
        |target| {
            extension_module::<tenferro_cpu::CpuEngine>(target.engine_id).map_err(|e| {
                tenferro_runtime::Error::runtime_state_source(
                    "window extension",
                    tenferro_runtime::ErrorPhase::Execution,
                    e,
                )
            })
        },
    )?;
    values
        .into_iter()
        .next()
        .ok_or_else(|| tenferro_ad::Error::Internal("missing window output".into()))
}

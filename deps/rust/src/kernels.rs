//! Shared plumbing of the native extension kernels: compact inputs, outputs
//! from the CPU engine's buffer pool, and parallel loops on the engine's
//! worker threads (so `Tenferro.set_num_threads` bounds them too).

use std::num::NonZeroUsize;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use tenferro_cpu::faer;

use tenferro_cpu::threadteam;
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

/// Element types of the native kernels.
pub(crate) trait Elem:
    Copy
    + Default
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::Sub<Output = Self>
    + std::ops::Mul<Output = Self>
    + std::ops::AddAssign
    + Send
    + Sync
    + tenferro_tensor::TensorScalar
    + tenferro_cpu_basic::PoolScalar
    + 'static
{
    /// The value padding takes in max pooling (below every other value).
    const LOWEST: Self;
    const ONE: Self;
    fn from_f64(v: f64) -> Self;
    fn exp(self) -> Self;
    fn ln(self) -> Self;
    /// `c = op(a) * op(b) + beta * c` (`beta` is 0 or 1) for column-major `a` (`lda` rows),
    /// `b` (`ldb` rows) and the `m × n` matrix `c`, where `op(a)` is `m × k`.
    #[allow(clippy::too_many_arguments)]
    fn gemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, a: &[Self], lda: usize, b: &[Self], ldb: usize, beta: bool, c: &mut [Self]);
}

macro_rules! elem {
    ($t:ty, $blas:ident) => {
        impl Elem for $t {
            const LOWEST: Self = <$t>::NEG_INFINITY;
            const ONE: Self = 1.0;
            fn from_f64(v: f64) -> Self {
                v as $t
            }
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn gemm(ta: bool, tb: bool, m: usize, n: usize, k: usize, a: &[Self], lda: usize, b: &[Self], ldb: usize, beta: bool, c: &mut [Self]) {
                gemm_impl::<$t>(&$blas, ta, tb, m, n, k, a, lda, b, ldb, beta, c)
            }
        }
    };
}

// BLAS ------------------------------------------------------------------------------------

/// Fortran `?gemm` with 64-bit integers (`sgemm_64_` / `dgemm_64_` of libblastrampoline),
/// including the hidden lengths of the two character arguments.
type GemmFn<T> = unsafe extern "C" fn(
    *const u8, *const u8, *const i64, *const i64, *const i64,
    *const T, *const T, *const i64, *const T, *const i64,
    *const T, *mut T, *const i64, usize, usize,
);

/// Addresses of `sgemm_64_` and `dgemm_64_`; 0 until Julia registers them.
static SGEMM: AtomicUsize = AtomicUsize::new(0);
static DGEMM: AtomicUsize = AtomicUsize::new(0);

/// Use the ILP64 Fortran `sgemm`/`dgemm` at these addresses for the dense kernels
/// (0 keeps the built-in GEMM).
///
/// # Safety
///
/// Nonzero addresses must be functions with the `GemmFn` signature that stay valid for the
/// life of the process.
/// Returns whether an `sgemm` was registered before.
pub(crate) unsafe fn set_blas_gemm(sgemm: usize, dgemm: usize) -> bool {
    DGEMM.store(dgemm, Ordering::Release);
    SGEMM.swap(sgemm, Ordering::AcqRel) != 0
}

#[allow(clippy::too_many_arguments)]
fn gemm_impl<T: Elem + faer::traits::ComplexField>(
    blas: &AtomicUsize,
    ta: bool,
    tb: bool,
    m: usize,
    n: usize,
    k: usize,
    a: &[T],
    lda: usize,
    b: &[T],
    ldb: usize,
    beta: bool,
    c: &mut [T],
) {
    assert!(c.len() >= m * n, "gemm output too short");
    if m == 0 || n == 0 {
        return;
    }
    if k == 0 {
        if !beta {
            c[..m * n].fill(T::default());
        }
        return;
    }
    let (ar, ac) = if ta { (k, m) } else { (m, k) };
    let (br, bc) = if tb { (n, k) } else { (k, n) };
    assert!(lda >= ar.max(1) && ldb >= br.max(1), "gemm leading dimension too small");
    assert!(a.len() >= lda * (ac - 1) + ar && b.len() >= ldb * (bc - 1) + br, "gemm operand too short");
    let f = blas.load(Ordering::Acquire);
    if f != 0 {
        // SAFETY: `set_blas_gemm` guarantees the signature; the slices cover the
        // column-major operands checked above, and `c` holds `m * n` entries.
        unsafe {
            let f: GemmFn<T> = std::mem::transmute::<usize, GemmFn<T>>(f);
            let (ta, tb) = (if ta { b"T" } else { b"N" }, if tb { b"T" } else { b"N" });
            let (m, n, k, lda, ldb, ldc) = (m as i64, n as i64, k as i64, lda as i64, ldb as i64, m as i64);
            let (alpha, beta) = (T::ONE, if beta { T::ONE } else { T::default() });
            f(ta.as_ptr(), tb.as_ptr(), &m, &n, &k, &alpha, a.as_ptr(), &lda, b.as_ptr(), &ldb, &beta, c.as_mut_ptr(), &ldc, 1, 1);
        }
        return;
    }
    use faer::{linalg::matmul::matmul, Accum, MatMut, MatRef, Par};
    let a = MatRef::from_column_major_slice_with_stride(a, ar, ac, lda);
    let b = MatRef::from_column_major_slice_with_stride(b, br, bc, ldb);
    let a = if ta { a.transpose() } else { a };
    let b = if tb { b.transpose() } else { b };
    let out = MatMut::from_column_major_slice_mut(&mut c[..m * n], m, n);
    matmul(out, if beta { Accum::Add } else { Accum::Replace }, a, b, T::ONE, Par::Seq);
}

elem!(f32, SGEMM);
elem!(f64, DGEMM);

/// Compact copies of the `inputs` that are not compact column-major views.
fn materialize(engine: &mut dyn Engine, inputs: &[TensorRead<'_>]) -> tenferro_tensor::Result<Vec<Option<Tensor>>> {
    inputs
        .iter()
        .map(|input| {
            let compact = input
                .clone()
                .tensor_view()
                .is_some_and(|v| v.is_col_major_contiguous().unwrap_or(false));
            Ok(if compact { None } else { Some(engine.to_contiguous_read(input.clone())?) })
        })
        .collect()
}

/// Run `kernel(inputs, output, exec)` on compact host slices of `inputs` with a
/// zeroed `len`-entry output taken from the engine's buffer pool, and wrap the
/// output as a tensor of `shape`.
///
/// `exec` is the engine's thread token (`None` off the CPU engine). The dtype
/// of the first input selects the element type.
pub(crate) fn execute<K>(
    op_name: &'static str,
    engine: &mut dyn Engine,
    inputs: &[TensorRead<'_>],
    shape: Vec<usize>,
    kernel: K,
) -> tenferro_tensor::Result<Vec<Tensor>>
where
    K: KernelFn,
{
    execute_multi(op_name, engine, inputs, vec![shape], Single(kernel))
}

struct Single<K>(K);
impl<K: KernelFn> MultiKernelFn for Single<K> {
    fn run<T: Elem>(&self, inputs: &[&[T]], outs: &mut [Vec<T>], exec: Option<threadteam::Exec<'_>>) {
        self.0.run(inputs, &mut outs[0], exec)
    }
}

/// Like [`execute`] with one zeroed pooled output per entry of `shapes`.
pub(crate) fn execute_multi<K>(
    op_name: &'static str,
    engine: &mut dyn Engine,
    inputs: &[TensorRead<'_>],
    shapes: Vec<Vec<usize>>,
    kernel: K,
) -> tenferro_tensor::Result<Vec<Tensor>>
where
    K: MultiKernelFn,
{
    let copies = materialize(engine, inputs)?;
    let reads: Vec<TensorRead<'_>> = inputs
        .iter()
        .zip(&copies)
        .map(|(input, copy)| copy.as_ref().map_or_else(|| input.clone(), TensorRead::from_tensor))
        .collect();
    let lens: Vec<usize> = shapes.iter().map(|shape| shape.iter().product::<usize>()).collect();
    macro_rules! go {
        ($t:ty) => {{
            let slices = reads.iter().map(|t| t.as_slice::<$t>()).collect::<tenferro_tensor::Result<Vec<_>>>()?;
            let pooled = tenferro_cpu::with_cpu_engine(engine, |cpu| {
                cpu.with_linalg_pool(|exec, buffers| {
                    let mut outs: Vec<Vec<$t>> = lens.iter().map(|&len| buffers.acquire_zeroed::<$t>(len)).collect();
                    kernel.run::<$t>(&slices, &mut outs, Some(exec.token()));
                    Ok(outs)
                })
            });
            let outs = match pooled {
                Some(outs) => outs?,
                None => {
                    let mut outs: Vec<Vec<$t>> = lens.iter().map(|&len| vec![<$t>::default(); len]).collect();
                    kernel.run::<$t>(&slices, &mut outs, None);
                    outs
                }
            };
            shapes
                .into_iter()
                .zip(outs)
                .map(|(shape, out)| Tensor::from_vec_col_major(shape, out))
                .collect::<tenferro_tensor::Result<Vec<_>>>()?
        }};
    }
    Ok(match reads[0].dtype() {
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        dtype => {
            return Err(tenferro_tensor::Error::unsupported_dtype(op_name, dtype, "supports Float32 and Float64"))
        }
    })
}

/// A kernel with several outputs, generic over the element type.
pub(crate) trait MultiKernelFn {
    fn run<T: Elem>(&self, inputs: &[&[T]], outs: &mut [Vec<T>], exec: Option<threadteam::Exec<'_>>);
}

/// A kernel generic over the element type.
pub(crate) trait KernelFn {
    fn run<T: Elem>(&self, inputs: &[&[T]], out: &mut [T], exec: Option<threadteam::Exec<'_>>);
}

/// A shared output pointer; tasks write disjoint ranges only.
struct Shared<T>(*mut T);
unsafe impl<T: Send> Send for Shared<T> {}
unsafe impl<T: Send> Sync for Shared<T> {}

/// Run `f(first, blocks)` over consecutive groups of the blocks of `block`
/// entries of `out` (the last one may be shorter), where `first` is the index
/// of the group's first block. Groups run on the engine's threads when the
/// kernel touches `work` entries in total and that pays.
pub(crate) fn par_blocks<T: Send>(
    exec: Option<threadteam::Exec<'_>>,
    out: &mut [T],
    block: usize,
    work: usize,
    f: impl Fn(usize, &mut [T]) + Sync,
) {
    /// Entries touched by one task at least.
    const GRAIN: usize = 1 << 15;
    let len = out.len();
    let nblocks = if block == 0 { 0 } else { len.div_ceil(block) };
    let exec = exec.filter(|e| e.budget().get() > 1 && work >= 2 * GRAIN && nblocks > 1);
    let Some(exec) = exec else {
        f(0, out);
        return;
    };
    let per_block = (work / nblocks).max(1);
    let grain = NonZeroUsize::new(GRAIN.div_ceil(per_block).max(1)).expect("positive grain");
    let base = Shared(out.as_mut_ptr());
    let base = &base;
    exec.for_ranges(nblocks, grain, |blocks: Range<usize>| {
        let start = blocks.start * block;
        let end = (blocks.end * block).min(len);
        // SAFETY: `for_ranges` hands out disjoint block ranges, so tasks write
        // disjoint parts of `out`, which outlives the call.
        let chunk = unsafe { std::slice::from_raw_parts_mut(base.0.add(start), end - start) };
        f(blocks.start, chunk);
    });
}

//! Shared plumbing of the native extension kernels: compact inputs, outputs
//! from the CPU engine's buffer pool, and parallel loops on the engine's
//! worker threads (so `Tenferro.set_num_threads` bounds them too).

use std::num::NonZeroUsize;
use std::ops::Range;

use tenferro_cpu::threadteam;
use tenferro_tensor::{DType, Engine, Tensor, TensorRead};

/// Element types of the native kernels.
pub(crate) trait Elem:
    Copy
    + Default
    + PartialOrd
    + std::ops::Add<Output = Self>
    + std::ops::AddAssign
    + Send
    + Sync
    + tenferro_tensor::TensorScalar
    + tenferro_cpu_basic::PoolScalar
    + 'static
{
    /// The value padding takes in max pooling (below every other value).
    const LOWEST: Self;
}
impl Elem for f32 {
    const LOWEST: Self = f32::NEG_INFINITY;
}
impl Elem for f64 {
    const LOWEST: Self = f64::NEG_INFINITY;
}

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
    let copies = materialize(engine, inputs)?;
    let reads: Vec<TensorRead<'_>> = inputs
        .iter()
        .zip(&copies)
        .map(|(input, copy)| copy.as_ref().map_or_else(|| input.clone(), TensorRead::from_tensor))
        .collect();
    let len = shape.iter().product::<usize>();
    macro_rules! go {
        ($t:ty) => {{
            let slices = reads.iter().map(|t| t.as_slice::<$t>()).collect::<tenferro_tensor::Result<Vec<_>>>()?;
            let pooled = tenferro_cpu::with_cpu_engine(engine, |cpu| {
                cpu.with_linalg_pool(|exec, buffers| {
                    let mut out = buffers.acquire_zeroed::<$t>(len);
                    kernel.run::<$t>(&slices, &mut out, Some(exec.token()));
                    Ok(out)
                })
            });
            let out = match pooled {
                Some(out) => out?,
                None => {
                    let mut out = vec![<$t>::default(); len];
                    kernel.run::<$t>(&slices, &mut out, None);
                    out
                }
            };
            Tensor::from_vec_col_major(shape, out)?
        }};
    }
    Ok(vec![match reads[0].dtype() {
        DType::F32 => go!(f32),
        DType::F64 => go!(f64),
        dtype => {
            return Err(tenferro_tensor::Error::unsupported_dtype(op_name, dtype, "supports Float32 and Float64"))
        }
    }])
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

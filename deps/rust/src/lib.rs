mod rem_extension;

use rustcall_julia_macros::julia;
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};

type FfiResult<T> = Result<T, Box<dyn std::error::Error>>;

fn float_dtype(kind: i32) -> FfiResult<tenferro_ad::DType> {
    Ok(match kind {
        0 => tenferro_ad::DType::F32,
        1 => tenferro_ad::DType::F64,
        4 => tenferro_ad::DType::C32,
        5 => tenferro_ad::DType::C64,
        _ => return Err("unsupported promotion dtype".into()),
    })
}

/// Element types exchanged with Julia as native-endian bytes.
///
/// RustCall checks each byte slice's span and lifetime; decoding here checks
/// the length and never reinterprets bytes as a typed reference.
trait HostElement: Copy {
    const SIZE: usize;
    fn read(bytes: &[u8]) -> Self;
    fn write(self, out: &mut [u8]);
}

macro_rules! numeric_host_element {
    ($($ty:ty),*) => {$(
        impl HostElement for $ty {
            const SIZE: usize = std::mem::size_of::<$ty>();
            fn read(bytes: &[u8]) -> Self {
                <$ty>::from_ne_bytes(bytes.try_into().expect("element width"))
            }
            fn write(self, out: &mut [u8]) {
                out.copy_from_slice(&self.to_ne_bytes());
            }
        }
    )*};
}
numeric_host_element!(f32, f64, i32, i64);

impl HostElement for bool {
    const SIZE: usize = 1;
    fn read(bytes: &[u8]) -> Self {
        bytes[0] != 0
    }
    fn write(self, out: &mut [u8]) {
        out[0] = u8::from(self);
    }
}

// Julia's `Complex{T}` and `num_complex::Complex<T>` both store `re` then `im`.
impl<T: HostElement> HostElement for num_complex::Complex<T> {
    const SIZE: usize = 2 * T::SIZE;
    fn read(bytes: &[u8]) -> Self {
        num_complex::Complex::new(T::read(&bytes[..T::SIZE]), T::read(&bytes[T::SIZE..]))
    }
    fn write(self, out: &mut [u8]) {
        let (re, im) = out.split_at_mut(T::SIZE);
        self.re.write(re);
        self.im.write(im);
    }
}

fn element_count(shape: &[usize]) -> FfiResult<usize> {
    Ok(shape
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or("shape overflow")?)
}

fn decode<T: HostElement>(bytes: &[u8], len: usize) -> FfiResult<Vec<T>> {
    if len.checked_mul(T::SIZE) != Some(bytes.len()) {
        return Err("input buffer does not match the tensor shape".into());
    }
    Ok(bytes.chunks_exact(T::SIZE).map(T::read).collect())
}

fn encode<T: HostElement>(values: &[T], out: &mut [u8]) -> FfiResult<()> {
    if values.len().checked_mul(T::SIZE) != Some(out.len()) {
        return Err("shape mismatch".into());
    }
    for (value, bytes) in values.iter().zip(out.chunks_exact_mut(T::SIZE)) {
        value.write(bytes);
    }
    Ok(())
}

fn input_tensor(data: &[u8], shape: &[usize], kind: i32) -> FfiResult<Tensor> {
    let len = element_count(shape)?;
    let dims = shape.to_vec();
    macro_rules! make {
        ($ty:ty) => {
            Tensor::from_vec_col_major(dims, decode::<$ty>(data, len)?)?
        };
    }
    Ok(match kind {
        0 => make!(f32),
        1 => make!(f64),
        2 => make!(i32),
        3 => make!(i64),
        6 => make!(bool),
        4 => make!(num_complex::Complex32),
        5 => make!(num_complex::Complex64),
        _ => return Err("unsupported element type".into()),
    })
}

fn copy_tensor(value: &Tensor, out: &mut [u8], kind: i32) -> FfiResult<()> {
    macro_rules! copy {
        ($ty:ty) => {
            encode(value.as_slice::<$ty>()?, out)
        };
    }
    match kind {
        0 => copy!(f32),
        1 => copy!(f64),
        2 => copy!(i32),
        3 => copy!(i64),
        6 => copy!(bool),
        4 => copy!(num_complex::Complex32),
        5 => copy!(num_complex::Complex64),
        _ => Err("unsupported element type".into()),
    }
}

fn compare_dir(code: i32) -> FfiResult<tenferro_ad::CompareDir> {
    Ok(match code {
        0 => tenferro_ad::CompareDir::Eq,
        1 => tenferro_ad::CompareDir::Lt,
        2 => tenferro_ad::CompareDir::Le,
        3 => tenferro_ad::CompareDir::Gt,
        4 => tenferro_ad::CompareDir::Ge,
        _ => return Err("unsupported comparison".into()),
    })
}

/// Gather whole slices of `shape` along `axis` using an `(m, 1)` index tensor.
/// The gathered axis keeps its position in the output.
fn gather_axis_config(shape: &[usize], axis: usize) -> FfiResult<tenferro_ad::GatherConfig> {
    if axis >= shape.len() {
        return Err("gather axis out of range".into());
    }
    let mut slice_sizes = shape.to_vec();
    slice_sizes[axis] = 1;
    Ok(tenferro_ad::GatherConfig {
        offset_dims: (0..shape.len()).filter(|&d| d != axis).collect(),
        collapsed_slice_dims: vec![axis],
        start_index_map: vec![axis],
        index_vector_dim: 1,
        slice_sizes,
    })
}

#[julia]
pub struct EagerGraph {
    ctx: Option<std::sync::Arc<EagerRuntime>>,
    init_error: String,
    nodes: Vec<EagerTensor>,
}

#[julia]
impl EagerGraph {
    #[julia]
    pub fn new() -> Self {
        let runtime = (|| -> FfiResult<std::sync::Arc<EagerRuntime>> {
            let ad = tenferro_ad::AdContext::builder()
                .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
                .with_semantic_extension_rules(rem_extension::rules()?)?
                .build()?;
            Ok(EagerRuntime::with_ad_context(&ad)?)
        })();
        match runtime {
            Ok(ctx) => Self {
                ctx: Some(ctx),
                init_error: String::new(),
                nodes: Vec::new(),
            },
            Err(err) => Self {
                ctx: None,
                init_error: err.to_string(),
                nodes: Vec::new(),
            },
        }
    }

    #[julia]
    pub fn input_array(
        &mut self,
        data: &[u8],
        shape: &[usize],
        kind: i32,
    ) -> Result<i64, String> {
        let result = (|| {
            let tensor = input_tensor(data, shape, kind)?;
            let ctx = self
                .ctx
                .as_ref()
                .ok_or_else(|| self.init_error.as_str())?
                .clone();
            Ok(if matches!(kind, 0 | 1 | 4 | 5) {
                ctx.variable_from(tensor)?
            } else {
                ctx.constant_from(tensor)?
            })
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn copy_array(
        &self,
        node: usize,
        out: &mut [u8],
        kind: i32,
    ) -> Result<i32, String> {
        let result = (|| -> FfiResult<()> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let node = self.nodes.get(node).ok_or("invalid node")?;
            let mut engine = tenferro_cpu::CpuEngine::serial();
            let value = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(node))?;
            copy_tensor(&value, out, kind)
        })();
        result.map(|_| 0).map_err(|err| err.to_string())
    }

    #[julia]
    pub fn rule_input(&mut self, node: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let input = self.nodes.get(node).ok_or("invalid input")?;
            let mut engine = tenferro_cpu::CpuEngine::serial();
            let value = ctx.with_eager_session(&mut engine, |s| s.duplicate_value(input))?;
            Ok(ctx.variable_from(value)?)
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn vjp_node(&mut self, output: usize, input: usize, seed: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<Option<EagerTensor>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let y = self.nodes.get(output).ok_or("invalid output")?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let seed = self.nodes.get(seed).ok_or("invalid cotangent")?;
            Ok(ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                s.vjp_optional(y, x, seed)
            })?)
        })();
        match result {
            Ok(Some(value)) => Ok(self.push(value)),
            Ok(None) => Ok(-2),
            Err(err) => Err(err.to_string()),
        }
    }

    #[julia]
    pub fn jvp_node(&mut self, output: usize, input: usize, seed: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<Option<EagerTensor>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let y = self.nodes.get(output).ok_or("invalid output")?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let seed = self.nodes.get(seed).ok_or("invalid tangent")?;
            Ok(ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                s.jvp_optional(y, x, seed)
            })?)
        })();
        match result {
            Ok(Some(value)) => Ok(self.push(value)),
            Ok(None) => Ok(-2),
            Err(err) => Err(err.to_string()),
        }
    }

    #[julia]
    pub fn unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.neg(x),
                    1 => s.exp(x),
                    2 => s.log(x),
                    3 => s.sqrt(x),
                    4 => s.sin(x),
                    5 => s.cos(x),
                    6 => s.tanh(x),
                    7 => s.expm1(x),
                    8 => s.log1p(x),
                    9 => s.conj(x),
                    10 => s.sign(x),
                    11 => s.abs(x),
                    12 => s.erf(x),
                    _ => unreachable!(),
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn convert_node(&mut self, input: usize, kind: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let dtype = float_dtype(kind)?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.convert(x, dtype)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn gather_axis(&mut self, input: usize, indices: usize, axis: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let ix = self.nodes.get(indices).ok_or("invalid indices")?;
            let config = gather_axis_config(x.shape(), axis)?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.gather(x, ix, config)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn compare_node(&mut self, lhs: usize, rhs: usize, dir: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            let dir = compare_dir(dir)?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.compare(a, b, dir)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn select_node(
        &mut self,
        condition: usize,
        on_true: usize,
        on_false: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let c = self.nodes.get(condition).ok_or("invalid condition")?;
            let t = self.nodes.get(on_true).ok_or("invalid true branch")?;
            let f = self.nodes.get(on_false).ok_or("invalid false branch")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.where_select(c, t, f)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn detach_node(&mut self, input: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| s.detach(x))?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn concatenate(
        &mut self,
        ids: &[usize],
        axis: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let inputs: Vec<&EagerTensor> = ids
                .iter()
                .map(|id| self.nodes.get(*id).ok_or("invalid input"))
                .collect::<Result<_, _>>()?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.concatenate(&inputs, axis as isize)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn linalg_factor(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::EagerLinalgOps;
        let result = (|| -> FfiResult<Vec<EagerTensor>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => {
                        let r = s.svd(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![r.u, r.s, r.vh])
                    }
                    1 => {
                        let r = s.qr(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![r.q, r.r])
                    }
                    2 => {
                        let r = s.eig(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![r.values, r.vectors])
                    }
                    3 => {
                        let r = s.eigh(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![r.values, r.vectors])
                    }
                    4 => {
                        let r = s.lu(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![r.p, r.l, r.u, r.parity])
                    }
                    _ => unreachable!(),
                })?,
            )
        })();
        let values = result.map_err(|err| err.to_string())?;
        let first = self.nodes.len() as i64;
        self.nodes.extend(values);
        Ok(first)
    }

    #[julia]
    pub fn norm_node(&mut self, input: usize) -> Result<i64, String> {
        use tenferro_linalg::EagerLinalgOps;
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.norm(x)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn linalg_unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::EagerLinalgOps;
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.det(x),
                    1 => s.inv(x),
                    2 => s.pinv(x),
                    3 => s.cholesky(x),
                    _ => unreachable!(),
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn linalg_solve(
        &mut self,
        lhs: usize,
        rhs: usize,
        least_squares: bool,
    ) -> Result<i64, String> {
        use tenferro_linalg::EagerLinalgOps;
        let result = (|| -> FfiResult<EagerTensor> {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            Ok(ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                if least_squares {
                    s.lstsq(a, b)
                } else {
                    s.solve(a, b)
                }
            })?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn matrix_structure(&mut self, input: usize, op: i32, k: i64) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.tril(x, k),
                    1 => s.triu(x, k),
                    2 => s.extract_diag(x, 0, 1),
                    _ => unreachable!(),
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn slice_node(
        &mut self,
        input: usize,
        bounds: &[usize],
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let rank = x.shape().len();
            if bounds.len() != 3 * rank {
                return Err("slice bounds must contain starts, limits and strides".into());
            }
            let config = tenferro_ad::SliceConfig {
                starts: bounds[..rank].to_vec(),
                limits: bounds[rank..2 * rank].to_vec(),
                strides: bounds[2 * rank..].to_vec(),
            };
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.slice(x, config)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn broadcast_to(
        &mut self,
        input: usize,
        shape: &[usize],
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let dims: Vec<usize> = (0..x.shape().len()).collect();
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.broadcast_in_dim(x, shape, &dims)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn shape_op(
        &mut self,
        input: usize,
        args: &[usize],
        op: i32,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let axes: Vec<isize> = args.iter().map(|&axis| axis as isize).collect();
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.reshape(x, args),
                    1 => s.transpose(x, &axes),
                    2 => s.reduce_sum(x, Some(&axes)),
                    3 => s.reverse(x, &axes),
                    4 => s.reduce_prod(x, Some(&axes)),
                    5 => s.reduce_max(x, Some(&axes)),
                    6 => s.reduce_min(x, Some(&axes)),
                    7 => s.reduce_mean(x, Some(&axes)),
                    _ => unreachable!(),
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn binary(&mut self, lhs: usize, rhs: usize, op: i32) -> Result<i64, String> {
        let result = (|| -> Result<EagerTensor, Box<dyn std::error::Error>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.add(a, b),
                    1 => s.sub(a, b),
                    2 => s.mul(a, b),
                    3 => s.div(a, b),
                    4 => s.pow(a, b),
                    5 => {
                        let values =
                            tenferro_ad::extension::apply_eager_with_targeted_extension_in_session(
                                s,
                                std::sync::Arc::new(rem_extension::Remainder),
                                &[a, b],
                                |target| {
                                    rem_extension::extension_module::<tenferro_cpu::CpuEngine>(
                                        target.engine_id,
                                    )
                                    .map_err(|e| {
                                        tenferro_runtime::Error::runtime_state_source(
                                            "remainder extension",
                                            tenferro_runtime::ErrorPhase::Execution,
                                            e,
                                        )
                                    })
                                },
                            )?;
                        values.into_iter().next().ok_or_else(|| {
                            tenferro_ad::Error::Internal("missing remainder output".into())
                        })
                    }
                    6 => s.maximum(a, b),
                    7 => s.minimum(a, b),
                    _ => unreachable!(),
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn matmul(&mut self, lhs: usize, rhs: usize) -> Result<i64, String> {
        let result = (|| -> Result<EagerTensor, Box<dyn std::error::Error>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| s.matmul(a, b))?)
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn sum(&mut self, input: usize) -> Result<i64, String> {
        let result = (|| -> Result<EagerTensor, Box<dyn std::error::Error>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.reduce_sum(x, None)
                })?,
            )
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }
}

impl EagerGraph {
    fn push(&mut self, value: EagerTensor) -> i64 {
        let id = self.nodes.len();
        self.nodes.push(value);
        id as i64
    }
}

#[julia]
pub struct TracedGraph {
    trace: tenferro_runtime::TraceContext,
    nodes: Vec<tenferro_runtime::TracedTensor>,
}

#[julia]
impl TracedGraph {
    #[julia]
    pub fn new() -> Self {
        Self {
            trace: tenferro_runtime::TraceContext::new(),
            nodes: Vec::new(),
        }
    }

    #[julia]
    pub fn traced_input_array(
        &mut self,
        data: &[u8],
        shape: &[usize],
        kind: i32,
    ) -> Result<i64, String> {
        let result = (|| {
            Ok(self.trace.from_tensor_concrete_shape(
                input_tensor(data, shape, kind)?,
            )?)
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_copy_array(
        &self,
        node: usize,
        out: &mut [u8],
        kind: i32,
    ) -> Result<i32, String> {
        let result = (|| -> FfiResult<()> {
            use tenferro_runtime::{ExecutionEngine, GraphCompiler, Runtime};
            let tensor = self.nodes.get(node).ok_or("invalid node")?;
            let program = GraphCompiler::new().compile(tensor)?;
            let mut engine = tenferro_cpu::CpuEngine::serial();
            let mut builder = Runtime::builder();
            builder.register_engine(tenferro_cpu::runtime_engine_registration(&engine)?)?;
            builder.install_extension_module(rem_extension::extension_module::<
                tenferro_cpu::CpuEngine,
            >(tenferro_cpu::runtime_engine_id()?)?)?;
            builder.install_extension_module(tenferro_linalg::extension_module::<
                tenferro_cpu::CpuEngine,
            >(tenferro_cpu::runtime_engine_id()?)?)?;
            let runtime = builder.build()?;
            let prepared = runtime.prepare(&program, &[], &[engine.target()])?;
            let mut outputs = runtime.run(&mut [&mut engine], &prepared, &[])?;
            let value = outputs
                .pop()
                .ok_or("missing output")?
                .into_tensor(&mut engine)?;
            copy_tensor(&value, out, kind)
        })();
        result.map(|_| 0).map_err(|err| err.to_string())
    }

    #[julia]
    pub fn traced_unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(match op {
                0 => x.neg()?,
                1 => x.exp()?,
                2 => x.log()?,
                3 => x.sqrt()?,
                4 => x.sin()?,
                5 => x.cos()?,
                6 => x.tanh()?,
                7 => x.expm1()?,
                8 => x.log1p()?,
                9 => x.conj()?,
                10 => x.sign()?,
                11 => x.abs()?,
                12 => x.erf()?,
                _ => unreachable!(),
            })
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_convert_node(&mut self, input: usize, kind: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(x.convert(float_dtype(kind)?)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_gather_axis(
        &mut self,
        input: usize,
        indices: usize,
        axis: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let ix = self.nodes.get(indices).ok_or("invalid indices")?;
            let shape = x.try_concrete_shape().ok_or("gather requires a concrete shape")?;
            Ok(x.gather(ix, gather_axis_config(&shape, axis)?)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_compare_node(&mut self, lhs: usize, rhs: usize, dir: i32) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(a.compare(b, compare_dir(dir)?)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_select_node(
        &mut self,
        condition: usize,
        on_true: usize,
        on_false: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let c = self.nodes.get(condition).ok_or("invalid condition")?;
            let t = self.nodes.get(on_true).ok_or("invalid true branch")?;
            let f = self.nodes.get(on_false).ok_or("invalid false branch")?;
            Ok(tenferro_runtime::TracedTensor::where_select(c, t, f)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_concatenate(
        &mut self,
        ids: &[usize],
        axis: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let inputs: Vec<&tenferro_runtime::TracedTensor> = ids
                .iter()
                .map(|id| self.nodes.get(*id).ok_or("invalid input"))
                .collect::<Result<_, _>>()?;
            Ok(tenferro_runtime::TracedTensor::concatenate(
                &inputs,
                axis as isize,
            )?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_linalg_factor(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::TracedLinalgOps;
        let result = (|| -> FfiResult<Vec<tenferro_runtime::TracedTensor>> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(match op {
                0 => {
                    let r = x.svd()?;
                    vec![r.u, r.s, r.vh]
                }
                1 => {
                    let r = x.qr()?;
                    vec![r.q, r.r]
                }
                2 => {
                    let r = x.eig()?;
                    vec![r.values, r.vectors]
                }
                3 => {
                    let r = x.eigh()?;
                    vec![r.values, r.vectors]
                }
                4 => {
                    let r = x.lu()?;
                    vec![r.p, r.l, r.u, r.parity]
                }
                _ => unreachable!(),
            })
        })();
        let values = result.map_err(|err| err.to_string())?;
        let first = self.nodes.len() as i64;
        self.nodes.extend(values);
        Ok(first)
    }

    #[julia]
    pub fn traced_norm_node(&mut self, input: usize) -> Result<i64, String> {
        use tenferro_linalg::TracedLinalgOps;
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(x.norm()?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_linalg_unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::TracedLinalgOps;
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(match op {
                0 => x.det()?,
                1 => x.inv()?,
                2 => x.pinv()?,
                3 => x.cholesky()?,
                _ => unreachable!(),
            })
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_linalg_solve(
        &mut self,
        lhs: usize,
        rhs: usize,
        least_squares: bool,
    ) -> Result<i64, String> {
        use tenferro_linalg::TracedLinalgOps;
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(if least_squares {
                a.lstsq(b)?
            } else {
                a.solve(b)?
            })
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_matrix_structure(
        &mut self,
        input: usize,
        op: i32,
        k: i64,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(match op {
                0 => x.tril(k)?,
                1 => x.triu(k)?,
                2 => x.extract_diag(0, 1)?,
                _ => unreachable!(),
            })
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_slice_node(
        &mut self,
        input: usize,
        bounds: &[usize],
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let rank = x.rank();
            if bounds.len() != 3 * rank {
                return Err("slice bounds must contain starts, limits and strides".into());
            }
            let config = tenferro_runtime::SliceConfig {
                starts: bounds[..rank].to_vec(),
                limits: bounds[rank..2 * rank].to_vec(),
                strides: bounds[2 * rank..].to_vec(),
            };
            Ok(x.slice(config)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_broadcast_to(
        &mut self,
        input: usize,
        shape: &[usize],
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let dims: Vec<usize> = (0..x.rank()).collect();
            Ok(x.broadcast_in_dim(shape, &dims)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_shape_op(
        &mut self,
        input: usize,
        args: &[usize],
        op: i32,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let axes: Vec<isize> = args.iter().map(|&axis| axis as isize).collect();
            Ok(match op {
                0 => x.reshape(args)?,
                1 => x.transpose(&axes)?,
                2 => x.reduce_sum(Some(&axes))?,
                3 => x.reverse(&axes)?,
                4 => x.reduce_prod(Some(&axes))?,
                5 => x.reduce_max(Some(&axes))?,
                6 => x.reduce_min(Some(&axes))?,
                7 => x.reduce_mean(Some(&axes))?,
                _ => unreachable!(),
            })
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_binary(&mut self, lhs: usize, rhs: usize, op: i32) -> Result<i64, String> {
        let result = (|| {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(match op {
                0 => a.add(b)?,
                1 => a.sub(b)?,
                2 => a.matmul(b)?,
                3 => a.mul(b)?,
                4 => a.div(b)?,
                5 => a.pow(b)?,
                6 => tenferro_runtime::extension::apply(
                    std::sync::Arc::new(rem_extension::Remainder),
                    &[a, b],
                )?
                .into_iter()
                .next()
                .ok_or("missing remainder output")?,
                7 => a.maximum(b)?,
                8 => a.minimum(b)?,
                _ => unreachable!(),
            })
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_sum(&mut self, input: usize) -> Result<i64, String> {
        let result = (|| {
            Ok(self
                .nodes
                .get(input)
                .ok_or("invalid input")?
                .reduce_sum(None)?)
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_derivative(
        &mut self,
        output: usize,
        input: usize,
        seed: usize,
        forward: bool,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<Option<tenferro_runtime::TracedTensor>> {
            let y = self.nodes.get(output).ok_or("invalid output")?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let seed = self.nodes.get(seed).ok_or("invalid seed")?;
            let ad = tenferro_ad::AdContext::builder()
                .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
                .with_semantic_extension_rules(rem_extension::rules()?)?
                .build()?;
            Ok(if forward {
                ad.jvp_optional(y, x, seed)?
            } else {
                ad.vjp_optional(y, x, seed)?
            })
        })();
        match result {
            Ok(Some(value)) => Ok(self.push(value)),
            Ok(None) => Ok(-2),
            Err(err) => Err(err.to_string()),
        }
    }

    #[julia]
    pub fn traced_grad(&mut self, output: usize, input: usize) -> Result<i64, String> {
        let result: Result<Option<tenferro_runtime::TracedTensor>, Box<dyn std::error::Error>> =
            (|| {
                let y = self.nodes.get(output).ok_or("invalid output")?;
                let x = self.nodes.get(input).ok_or("invalid input")?;
                let ad = tenferro_ad::AdContext::builder()
                    .with_semantic_extension_rules(tenferro_linalg::semantic_ad_rules()?)?
                    .with_semantic_extension_rules(rem_extension::rules()?)?
                    .build()?;
                Ok(ad.grad_optional(y, x)?)
            })();
        match result {
            Ok(Some(value)) => Ok(self.push(value)),
            Ok(None) => Ok(-2),
            Err(err) => Err(err.to_string()),
        }
    }
}

impl TracedGraph {
    fn push(&mut self, value: tenferro_runtime::TracedTensor) -> i64 {
        let id = self.nodes.len();
        self.nodes.push(value);
        id as i64
    }
}

enum TypedData {
    Bool(tenferro_tensor::TypedTensor<bool, tenferro_tensor::DynRank, tenferro_tensor::Host>),
    F32(tenferro_tensor::TypedTensor<f32, tenferro_tensor::DynRank, tenferro_tensor::Host>),
    F64(tenferro_tensor::TypedTensor<f64, tenferro_tensor::DynRank, tenferro_tensor::Host>),
    I32(tenferro_tensor::TypedTensor<i32, tenferro_tensor::DynRank, tenferro_tensor::Host>),
    I64(tenferro_tensor::TypedTensor<i64, tenferro_tensor::DynRank, tenferro_tensor::Host>),
    C32(
        tenferro_tensor::TypedTensor<
            num_complex::Complex32,
            tenferro_tensor::DynRank,
            tenferro_tensor::Host,
        >,
    ),
    C64(
        tenferro_tensor::TypedTensor<
            num_complex::Complex64,
            tenferro_tensor::DynRank,
            tenferro_tensor::Host,
        >,
    ),
}

macro_rules! typed_match {
    ($data:expr, |$tensor:ident| $body:expr) => {
        match $data {
            TypedData::Bool($tensor) => $body,
            TypedData::F32($tensor) => $body,
            TypedData::F64($tensor) => $body,
            TypedData::I32($tensor) => $body,
            TypedData::I64($tensor) => $body,
            TypedData::C32($tensor) => $body,
            TypedData::C64($tensor) => $body,
        }
    };
}

#[julia]
pub struct TypedStorage {
    data: TypedData,
}

#[julia]
impl TypedStorage {
    #[julia]
    pub fn new(data: &[u8], shape: &[usize], kind: i32) -> Result<Self, String> {
        let result = (|| -> FfiResult<TypedData> {
            let len = element_count(shape)?;
            macro_rules! make {
                ($variant:ident) => {
                    TypedData::$variant(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        shape.to_vec(),
                        decode(data, len)?,
                    )?)
                };
            }
            Ok(match kind {
                0 => make!(F32),
                1 => make!(F64),
                2 => make!(I32),
                3 => make!(I64),
                4 => make!(C32),
                5 => make!(C64),
                6 => make!(Bool),
                _ => return Err("unsupported element type".into()),
            })
        })();
        result
            .map(|data| Self { data })
            .map_err(|err| err.to_string())
    }
    #[julia]
    pub fn typed_copy(&self, out: &mut [u8]) -> i32 {
        typed_match!(&self.data, |t| i32::from(encode(t.as_slice(), out).is_err()))
    }
    #[julia]
    pub fn typed_get(&self, index: usize, out: &mut [u8]) -> i32 {
        typed_match!(&self.data, |t| match t.as_slice().get(index) {
            Some(v) => i32::from(encode(std::slice::from_ref(v), out).is_err()),
            None => 1,
        })
    }
    #[julia]
    pub fn typed_set(&mut self, index: usize, value: &[u8]) -> i32 {
        typed_match!(&mut self.data, |t| match t.host_data_mut().get_mut(index) {
            Some(v) => match decode(value, 1) {
                Ok(decoded) => {
                    *v = decoded[0];
                    0
                }
                Err(_) => 1,
            },
            None => 1,
        })
    }
}

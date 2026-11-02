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

unsafe fn input_tensor(
    data: *const u8,
    shape: *const usize,
    rank: usize,
    kind: i32,
) -> FfiResult<Tensor> {
    let dims = if rank == 0 {
        vec![]
    } else {
        std::slice::from_raw_parts(shape, rank).to_vec()
    };
    let len = dims
        .iter()
        .try_fold(1usize, |n, d| n.checked_mul(*d))
        .ok_or("shape overflow")?;
    macro_rules! make {
        ($ty:ty) => {{
            let values = if len == 0 {
                vec![]
            } else {
                std::slice::from_raw_parts(data.cast::<$ty>(), len).to_vec()
            };
            Tensor::from_vec_col_major(dims, values)?
        }};
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

unsafe fn copy_tensor(value: &Tensor, out: *mut u8, len: usize, kind: i32) -> FfiResult<()> {
    macro_rules! copy {
        ($ty:ty) => {{
            let values = value.as_slice::<$ty>()?;
            if values.len() != len {
                return Err("shape mismatch".into());
            }
            if len > 0 {
                std::ptr::copy_nonoverlapping(values.as_ptr(), out.cast::<$ty>(), len);
            }
        }};
    }
    match kind {
        0 => copy!(f32),
        1 => copy!(f64),
        2 => copy!(i32),
        3 => copy!(i64),
        6 => copy!(bool),
        4 => copy!(num_complex::Complex32),
        5 => copy!(num_complex::Complex64),
        _ => return Err("unsupported element type".into()),
    }
    Ok(())
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
        data: *const u8,
        shape: *const usize,
        rank: usize,
        kind: i32,
    ) -> Result<i64, String> {
        let result = (|| {
            let tensor = unsafe { input_tensor(data, shape, rank, kind)? };
            let ctx = self
                .ctx
                .as_ref()
                .ok_or_else(|| self.init_error.as_str())?
                .clone();
            Ok(if matches!(kind, 0 | 1 | 4 | 5) {
                EagerTensor::requires_grad_in(tensor, ctx)?
            } else {
                EagerTensor::from_tensor_in(tensor, ctx)?
            })
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn copy_array(
        &self,
        node: usize,
        out: *mut u8,
        len: usize,
        kind: i32,
    ) -> Result<i32, String> {
        let result = (|| -> FfiResult<()> {
            let value = self
                .nodes
                .get(node)
                .ok_or("invalid node")?
                .to_tensor(&mut tenferro_cpu::CpuEngine::serial())?;
            unsafe {
                copy_tensor(&value, out, len, kind)?;
            }
            Ok(())
        })();
        result.map(|_| 0).map_err(|err| err.to_string())
    }

    #[julia]
    pub fn rule_input(&mut self, node: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let input = self.nodes.get(node).ok_or("invalid input")?;
            let mut engine = tenferro_cpu::CpuEngine::serial();
            let value = input.to_tensor(&mut engine)?;
            Ok(EagerTensor::requires_grad_in(value, ctx.clone())?)
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
            Ok(ctx.vjp_optional(&mut tenferro_cpu::CpuEngine::serial(), y, x, seed)?)
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
            Ok(ctx.jvp_optional(&mut tenferro_cpu::CpuEngine::serial(), y, x, seed)?)
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
    pub fn gather_vector(&mut self, input: usize, indices: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let ix = self.nodes.get(indices).ok_or("invalid indices")?;
            let config = tenferro_ad::GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            };
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
    pub fn concatenate(
        &mut self,
        data: *const usize,
        len: usize,
        axis: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let ids = if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(data, len) }
            };
            let inputs: Vec<&EagerTensor> = ids
                .iter()
                .map(|id| self.nodes.get(*id).ok_or("invalid input"))
                .collect::<Result<_, _>>()?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.concatenate(&inputs, axis)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn linalg_factor(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::EagerSessionLinalgExt;
        let result = (|| -> FfiResult<Vec<EagerTensor>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => {
                        let (u, v, w) = s.svd(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![u, v, w])
                    }
                    1 => {
                        let (q, r) = s.qr(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![q, r])
                    }
                    2 => {
                        let (v, w) = s.eig(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![v, w])
                    }
                    3 => {
                        let (v, w) = s.eigh(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![v, w])
                    }
                    4 => {
                        let (p, l, u, parity) = s.lu(x)?;
                        Ok::<_, tenferro_ad::Error>(vec![p, l, u, parity])
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
        use tenferro_linalg::EagerSessionLinalgExt;
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                    s.norm(x, None, None, false)
                })?,
            )
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn linalg_unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::EagerSessionLinalgExt;
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
        use tenferro_linalg::{EagerSessionLinalgExt, EagerTensorLinalgExt};
        let result = (|| -> FfiResult<EagerTensor> {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            if least_squares {
                let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
                Ok(
                    ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| {
                        s.lstsq(a, b)
                    })?,
                )
            } else {
                Ok(a.solve(&mut tenferro_cpu::CpuEngine::serial(), b)?)
            }
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
        data: *const usize,
        rank: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let values = if rank == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(data, 3 * rank) }
            };
            let config = tenferro_ad::SliceConfig {
                starts: values[..rank].to_vec(),
                limits: values[rank..2 * rank].to_vec(),
                strides: values[2 * rank..].to_vec(),
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
        shape: *const usize,
        rank: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let shape = if rank == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(shape, rank) }
            };
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
        values: *const usize,
        len: usize,
        op: i32,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<EagerTensor> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let args = if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(values, len) }
            };
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.reshape(x, args),
                    1 => s.transpose(x, args),
                    2 => s.reduce_sum(x, Some(args)),
                    3 => s.reverse(x, args),
                    4 => s.reduce_prod(x, Some(args)),
                    5 => s.reduce_max(x, Some(args)),
                    6 => s.reduce_min(x, Some(args)),
                    7 => s.reduce_mean(x, Some(args)),
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
    nodes: Vec<tenferro_runtime::TracedTensor>,
}

#[julia]
impl TracedGraph {
    #[julia]
    pub fn new() -> Self {
        Self { nodes: Vec::new() }
    }

    #[julia]
    pub fn traced_input_array(
        &mut self,
        data: *const u8,
        shape: *const usize,
        rank: usize,
        kind: i32,
    ) -> Result<i64, String> {
        let result = (|| {
            Ok(tenferro_runtime::TracedTensor::from_tensor_concrete_shape(
                unsafe { input_tensor(data, shape, rank, kind)? },
            )?)
        })();
        let value = result.map_err(|err: Box<dyn std::error::Error>| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_copy_array(
        &self,
        node: usize,
        out: *mut u8,
        len: usize,
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
            unsafe { copy_tensor(&value, out, len, kind) }
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
    pub fn traced_gather_vector(&mut self, input: usize, indices: usize) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let ix = self.nodes.get(indices).ok_or("invalid indices")?;
            let config = tenferro_runtime::GatherConfig {
                offset_dims: vec![],
                collapsed_slice_dims: vec![0],
                start_index_map: vec![0],
                index_vector_dim: 1,
                slice_sizes: vec![1],
            };
            Ok(x.gather(ix, config)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_concatenate(
        &mut self,
        data: *const usize,
        len: usize,
        axis: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let ids = if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(data, len) }
            };
            let inputs: Vec<&tenferro_runtime::TracedTensor> = ids
                .iter()
                .map(|id| self.nodes.get(*id).ok_or("invalid input"))
                .collect::<Result<_, _>>()?;
            Ok(tenferro_runtime::TracedTensor::concatenate(&inputs, axis)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_linalg_factor(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::TracedTensorLinalgExt;
        let result = (|| -> FfiResult<Vec<tenferro_runtime::TracedTensor>> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(match op {
                0 => {
                    let (u, v, w) = x.svd()?;
                    vec![u, v, w]
                }
                1 => {
                    let (q, r) = x.qr()?;
                    vec![q, r]
                }
                2 => {
                    let (v, w) = x.eig()?;
                    vec![v, w]
                }
                3 => {
                    let (v, w) = x.eigh()?;
                    vec![v, w]
                }
                4 => {
                    let (p, l, u, parity) = x.lu()?;
                    vec![p, l, u, parity]
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
        use tenferro_linalg::TracedTensorLinalgExt;
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            Ok(x.norm(None, None, false)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_linalg_unary(&mut self, input: usize, op: i32) -> Result<i64, String> {
        use tenferro_linalg::TracedTensorLinalgExt;
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
        use tenferro_linalg::TracedTensorLinalgExt;
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
        data: *const usize,
        rank: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let values = if rank == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(data, 3 * rank) }
            };
            let config = tenferro_runtime::SliceConfig {
                starts: values[..rank].to_vec(),
                limits: values[rank..2 * rank].to_vec(),
                strides: values[2 * rank..].to_vec(),
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
        shape: *const usize,
        rank: usize,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let shape = if rank == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(shape, rank) }
            };
            let dims: Vec<usize> = (0..x.rank).collect();
            Ok(x.broadcast_in_dim(shape, &dims)?)
        })();
        let value = result.map_err(|err| err.to_string())?;
        Ok(self.push(value))
    }

    #[julia]
    pub fn traced_shape_op(
        &mut self,
        input: usize,
        values: *const usize,
        len: usize,
        op: i32,
    ) -> Result<i64, String> {
        let result = (|| -> FfiResult<tenferro_runtime::TracedTensor> {
            let x = self.nodes.get(input).ok_or("invalid input")?;
            let args = if len == 0 {
                &[]
            } else {
                unsafe { std::slice::from_raw_parts(values, len) }
            };
            Ok(match op {
                0 => x.reshape(args)?,
                1 => x.transpose(args)?,
                2 => x.reduce_sum(Some(args))?,
                3 => x.reverse(args)?,
                4 => x.reduce_prod(Some(args))?,
                5 => x.reduce_max(Some(args))?,
                6 => x.reduce_min(Some(args))?,
                7 => x.reduce_mean(Some(args))?,
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
            Some(TypedData::Bool($tensor)) => $body,
            Some(TypedData::F32($tensor)) => $body,
            Some(TypedData::F64($tensor)) => $body,
            Some(TypedData::I32($tensor)) => $body,
            Some(TypedData::I64($tensor)) => $body,
            Some(TypedData::C32($tensor)) => $body,
            Some(TypedData::C64($tensor)) => $body,
            None => 1,
        }
    };
}

#[julia]
pub struct TypedStorage {
    data: Option<TypedData>,
    error: String,
}

#[julia]
impl TypedStorage {
    #[julia]
    pub fn new(data: *const u8, shape: *const usize, rank: usize, kind: i32) -> Self {
        let result = (|| -> Result<TypedData, Box<dyn std::error::Error>> {
            let dims = if rank == 0 {
                vec![]
            } else {
                unsafe { std::slice::from_raw_parts(shape, rank) }.to_vec()
            };
            let len = dims
                .iter()
                .try_fold(1usize, |n, d| n.checked_mul(*d))
                .ok_or("shape overflow")?;
            Ok(match kind {
                0 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe { std::slice::from_raw_parts(data.cast::<f32>(), len) }.to_vec()
                    };
                    TypedData::F32(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                1 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe { std::slice::from_raw_parts(data.cast::<f64>(), len) }.to_vec()
                    };
                    TypedData::F64(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                2 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe { std::slice::from_raw_parts(data.cast::<i32>(), len) }.to_vec()
                    };
                    TypedData::I32(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                3 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe { std::slice::from_raw_parts(data.cast::<i64>(), len) }.to_vec()
                    };
                    TypedData::I64(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                4 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe {
                            std::slice::from_raw_parts(data.cast::<num_complex::Complex32>(), len)
                        }
                        .to_vec()
                    };
                    TypedData::C32(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                5 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe {
                            std::slice::from_raw_parts(data.cast::<num_complex::Complex64>(), len)
                        }
                        .to_vec()
                    };
                    TypedData::C64(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                6 => {
                    let values = if len == 0 {
                        vec![]
                    } else {
                        unsafe { std::slice::from_raw_parts(data.cast::<bool>(), len) }.to_vec()
                    };
                    TypedData::Bool(tenferro_tensor::TypedTensor::from_host_vec_col_major(
                        dims, values,
                    )?)
                }
                _ => return Err("unsupported element type".into()),
            })
        })();
        match result {
            Ok(data) => Self {
                data: Some(data),
                error: String::new(),
            },
            Err(err) => Self {
                data: None,
                error: err.to_string(),
            },
        }
    }
    #[julia]
    pub fn typed_valid(&self) -> bool {
        self.data.is_some()
    }
    #[julia]
    pub fn typed_error(&self) -> String {
        self.error.clone()
    }
    #[julia]
    pub fn typed_copy(&self, out: *mut u8, len: usize) -> i32 {
        typed_match!(&self.data, |t| {
            let slice = t.as_slice();
            if slice.len() != len {
                1
            } else {
                if len > 0 {
                    unsafe { std::ptr::copy_nonoverlapping(slice.as_ptr(), out.cast(), len) };
                }
                0
            }
        })
    }
    #[julia]
    pub fn typed_get(&self, index: usize, out: *mut u8) -> i32 {
        typed_match!(&self.data, |t| match t.as_slice().get(index) {
            Some(v) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        std::ptr::from_ref(v).cast::<u8>(),
                        out,
                        std::mem::size_of_val(v),
                    )
                };
                0
            }
            None => 1,
        })
    }
    #[julia]
    pub fn typed_set(&mut self, index: usize, value: *const u8) -> i32 {
        typed_match!(&mut self.data, |t| match t.host_data_mut().get_mut(index) {
            Some(v) => {
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        value,
                        std::ptr::from_mut(&mut *v).cast::<u8>(),
                        std::mem::size_of_val(v),
                    )
                };
                0
            }
            None => 1,
        })
    }
}

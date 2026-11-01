use rustcall_julia_macros::julia;
use tenferro_ad::{EagerRuntime, EagerTensor, Tensor};

type FfiResult<T> = Result<T, Box<dyn std::error::Error>>;

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

macro_rules! copy_guard {
    ($value:expr, $out:expr, $len:expr, $kind:expr) => {{
        macro_rules! copy {
            ($ty:ty) => {{
                let values = $value.as_slice::<$ty>()?;
                if values.len() != $len {
                    return Err("shape mismatch".into());
                }
                if $len > 0 {
                    unsafe {
                        std::ptr::copy_nonoverlapping(values.as_ptr(), $out.cast::<$ty>(), $len)
                    };
                }
            }};
        }
        match $kind {
            0 => copy!(f32),
            1 => copy!(f64),
            2 => copy!(i32),
            3 => copy!(i64),
            6 => copy!(bool),
            4 => copy!(num_complex::Complex32),
            5 => copy!(num_complex::Complex64),
            _ => return Err("unsupported element type".into()),
        }
    }};
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
        match EagerRuntime::new() {
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
            let value = self.nodes.get(node).ok_or("invalid node")?.value()?;
            copy_guard!(value, out, len, kind);
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
    pub fn binary(&mut self, lhs: usize, rhs: usize, op: i32) -> Result<i64, String> {
        let result = (|| -> Result<EagerTensor, Box<dyn std::error::Error>> {
            let ctx = self.ctx.as_ref().ok_or_else(|| self.init_error.as_str())?;
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(
                ctx.with_eager_session(&mut tenferro_cpu::CpuEngine::serial(), |s| match op {
                    0 => s.add(a, b),
                    _ => s.sub(a, b),
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
    pub fn traced_binary(&mut self, lhs: usize, rhs: usize, op: i32) -> Result<i64, String> {
        let result = (|| {
            let a = self.nodes.get(lhs).ok_or("invalid lhs")?;
            let b = self.nodes.get(rhs).ok_or("invalid rhs")?;
            Ok(match op {
                0 => a.add(b)?,
                1 => a.sub(b)?,
                _ => a.matmul(b)?,
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
        use tenferro_ad::TracedTensorAdExt;
        let result: Result<Option<tenferro_runtime::TracedTensor>, Box<dyn std::error::Error>> =
            (|| {
                let y = self.nodes.get(output).ok_or("invalid output")?;
                let x = self.nodes.get(input).ok_or("invalid input")?;
                Ok(y.grad_optional(x)?)
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

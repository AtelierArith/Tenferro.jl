//! Prepared traced programs, reused across traced graphs of the same structure.
//!
//! Every `TracedContext` is a fresh trace, so each evaluation used to build a
//! `Runtime` (with every extension module) and prepare the compiled program for
//! it, although a training step traces the same structure every time: about
//! 250–300 µs of a 1.4 ms MNIST MLP step. The shared engine now owns one
//! `Runtime` and a small cache of prepared programs. A program is reused when
//! its normalized semantic structure (`SemanticProgram::semantic_eq`) and the
//! dtypes and shapes of its inputs match; it then runs on the new trace's input
//! tensors, passed explicitly in program input order.
//!
//! Only programs whose bindings are exactly their inputs, in an order the
//! caller could check (`traced_inputs` in `lib.rs`), are cached: data kept in a
//! binding that is not an input would not be replaced by a rerun. Other
//! programs are prepared on the shared runtime and run as before. Each entry
//! keeps its program, and with it the input tensors of the trace that prepared
//! it, until it is evicted; at most [`CAPACITY`] entries are kept.

use std::collections::VecDeque;

use tenferro_runtime::{CompiledGraph, ExecutionEngine, Prepared, Runtime};
use tenferro_tensor::{DType, Tensor, TensorRead};

use crate::FfiResult;

/// Prepared programs kept at most.
const CAPACITY: usize = 32;
/// Call-log entries (see [`TracedPrograms::run_call`]) kept at most.
const CALL_CAPACITY: usize = 64;

/// A call log and the leaf node ids of its program's inputs, in program input order.
pub type CallProgram = (Box<[u64]>, Box<[usize]>);

/// What a call log led to the last time it was seen.
enum CallResult {
    /// `vjp_many` returned a derivative for exactly the inputs marked `true`.
    Vjp(Box<[bool]>),
    /// Evaluating the requested nodes ran this program; input `i` is the tensor of leaf node
    /// `sources[i]`.
    Program { prepared: Prepared, sources: Box<[usize]>, inputs: Box<[(DType, Box<[usize]>)]> },
}

struct CallEntry {
    hash: u64,
    key: Box<[u64]>,
    result: CallResult,
}

fn hash_key(key: &[u64]) -> u64 {
    // FNV-1a over the words; collisions only cost a slice comparison.
    key.iter().fold(0xcbf2_9ce4_8422_2325_u64, |h, &w| (h ^ w).wrapping_mul(0x0000_0100_0000_01b3))
}

struct Entry {
    program: CompiledGraph,
    inputs: Box<[(DType, Box<[usize]>)]>,
    prepared: Prepared,
}

/// One runtime and its recently prepared traced programs, owned by the shared engine.
#[derive(Default)]
pub struct TracedPrograms {
    runtime: Option<Runtime>,
    /// Least recently used first.
    entries: VecDeque<Entry>,
    /// Call logs of traced graphs and what they led to, least recently used first.
    calls: VecDeque<CallEntry>,
    hits: u64,
    misses: u64,
}

fn signature(inputs: &[TensorRead<'_>]) -> Box<[(DType, Box<[usize]>)]> {
    inputs.iter().map(|read| (read.dtype(), read.shape().into())).collect()
}

impl TracedPrograms {
    /// `(entries, hits, misses)`.
    pub fn stats(&self) -> (usize, u64, u64) {
        (self.entries.len(), self.hits, self.misses)
    }

    /// Move the entry of `key` to the back of the LRU order; `false` when there is none.
    fn touch_call(&mut self, key: &[u64]) -> bool {
        let hash = hash_key(key);
        let Some(index) = self.calls.iter().position(|entry| entry.hash == hash && *entry.key == *key) else {
            return false;
        };
        let entry = self.calls.remove(index).expect("index is in range");
        self.calls.push_back(entry);
        true
    }

    fn store_call(&mut self, key: Box<[u64]>, result: CallResult) {
        let hash = hash_key(&key);
        if let Some(index) = self.calls.iter().position(|entry| entry.hash == hash && entry.key == key) {
            self.calls.remove(index);
        }
        if self.calls.len() >= CALL_CAPACITY {
            self.calls.pop_front();
        }
        self.calls.push_back(CallEntry { hash, key, result });
    }

    /// The derivative pattern `vjp_many` returned for the call log `key`, if it was seen.
    pub fn vjp_pattern(&mut self, key: &[u64]) -> Option<Box<[bool]>> {
        if !self.touch_call(key) {
            return None;
        }
        match &self.calls.back()?.result {
            CallResult::Vjp(pattern) => Some(pattern.clone()),
            CallResult::Program { .. } => None,
        }
    }

    /// Remember that `vjp_many` returned `pattern` for the call log `key`.
    pub fn store_vjp(&mut self, key: Box<[u64]>, pattern: Box<[bool]>) {
        self.store_call(key, CallResult::Vjp(pattern));
    }

    /// Run the program an earlier graph with the call log `key` evaluated, on the leaf tensors
    /// of `nodes`; `None` when `key` was not seen (or its leaves do not fit).
    ///
    /// Two graphs with the same call log record the same operations on leaves of the same
    /// dtypes and shapes, so they compile to the same program with the same input order, and
    /// the leaves are found by node id. Only programs whose every input is a leaf are kept.
    pub fn run_call(
        &mut self,
        engine: &mut tenferro_cpu::CpuEngine,
        key: &[u64],
        nodes: &[tenferro_runtime::TracedTensor],
    ) -> FfiResult<Option<Vec<Tensor>>> {
        if !self.touch_call(key) {
            return Ok(None);
        }
        let Some(CallEntry { result: CallResult::Program { prepared, sources, inputs: sig }, .. }) = self.calls.back()
        else {
            return Ok(None);
        };
        let mut inputs = Vec::with_capacity(sources.len());
        for &id in sources.iter() {
            match nodes.get(id).and_then(|node| node.attached_value()) {
                Some(value) => inputs.push(value.tensor_read()?),
                None => return Ok(None),
            }
        }
        if signature(&inputs) != *sig {
            return Ok(None);
        }
        let runtime = self.runtime.as_ref().expect("runtime exists once a program is prepared");
        let outputs = runtime.run(&mut [&mut *engine], prepared, &inputs)?;
        self.hits += 1;
        Ok(Some(outputs.into_iter().map(|output| Ok(output.into_tensor(engine)?)).collect::<FfiResult<Vec<_>>>()?))
    }

    fn runtime(&mut self, engine: &mut tenferro_cpu::CpuEngine) -> FfiResult<&Runtime> {
        if self.runtime.is_none() {
            self.runtime = Some(crate::build_runtime(engine)?);
        }
        Ok(self.runtime.as_ref().expect("runtime was just built"))
    }

    /// Run `program` on `engine`. `inputs` are its input tensors in program input order (then
    /// the prepared program is looked up and cached); `None` prepares and runs it on its own
    /// bound tensors without caching.
    pub fn run(
        &mut self,
        engine: &mut tenferro_cpu::CpuEngine,
        program: &CompiledGraph,
        inputs: Option<&[TensorRead<'_>]>,
        call: Option<CallProgram>,
    ) -> FfiResult<Vec<Tensor>> {
        let Some(inputs) = inputs else {
            let runtime = self.runtime(engine)?;
            let prepared = runtime.prepare(program, &[], &[engine.target()])?;
            let outputs = runtime.run(&mut [&mut *engine], &prepared, &[])?;
            return outputs.into_iter().map(|output| Ok(output.into_tensor(engine)?)).collect();
        };
        let sig = signature(inputs);
        let found = self.entries.iter().position(|entry| {
            entry.inputs == sig
                && entry.program.output_count() == program.output_count()
                && entry.program.program().semantic_fingerprint() == program.program().semantic_fingerprint()
                && entry.program.program().semantic_eq(program.program())
        });
        let entry = match found {
            Some(index) => {
                self.hits += 1;
                self.entries.remove(index).expect("index is in range")
            }
            None => {
                self.misses += 1;
                let prepared = self.runtime(engine)?.prepare(program, inputs, &[engine.target()])?;
                Entry { program: program.clone(), inputs: sig.clone(), prepared }
            }
        };
        let runtime = self.runtime.as_ref().expect("runtime exists once a program is prepared");
        let result = (|| -> FfiResult<Vec<Tensor>> {
            let outputs = runtime.run(&mut [&mut *engine], &entry.prepared, inputs)?;
            outputs.into_iter().map(|output| Ok(output.into_tensor(engine)?)).collect()
        })();
        if self.entries.len() >= CAPACITY {
            self.entries.pop_front();
        }
        // A program whose run failed is not kept.
        if result.is_ok() {
            if let Some((key, sources)) = call {
                let result = CallResult::Program {
                    prepared: entry.prepared.clone(),
                    sources,
                    inputs: entry.inputs.clone(),
                };
                self.store_call(key, result);
            }
            self.entries.push_back(entry);
        }
        result
    }
}

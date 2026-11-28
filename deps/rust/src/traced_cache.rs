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
                Entry { program: program.clone(), inputs: sig, prepared }
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
            self.entries.push_back(entry);
        }
        result
    }
}

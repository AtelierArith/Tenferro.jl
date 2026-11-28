//! The process-wide CPU execution engine shared by every bridge call.
//!
//! A `CpuEngine` owns the buffer pool that recycles outputs, the kernel plan
//! cache and (optionally) a worker pool. Building one per call threw all of that
//! away, so every call borrows this one engine instead. Calls from different
//! Julia threads serialize on the engine lock.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use tenferro_cpu::CpuEngine;

use crate::traced_cache::TracedPrograms;

/// Worker threads requested by the caller; `0` means "not set yet".
static REQUESTED_THREADS: AtomicUsize = AtomicUsize::new(0);

struct Shared {
    engine: CpuEngine,
    /// The thread count this engine was requested with (it may have fallen back to serial).
    requested: usize,
    /// The runtime and prepared programs of traced evaluations on this engine.
    programs: TracedPrograms,
}

static ENGINE: Mutex<Option<Shared>> = Mutex::new(None);

/// The thread count from `TENFERRO_NUM_THREADS`, or 1 (serial) when unset or invalid.
fn env_threads() -> usize {
    std::env::var("TENFERRO_NUM_THREADS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .filter(|&n| n >= 1)
        .unwrap_or(1)
}

/// The thread count the next engine is built with.
pub fn requested_threads() -> usize {
    match REQUESTED_THREADS.load(Ordering::Relaxed) {
        0 => {
            let threads = env_threads();
            // Keep an explicit request that raced with this first read.
            let _ = REQUESTED_THREADS.compare_exchange(0, threads, Ordering::Relaxed, Ordering::Relaxed);
            REQUESTED_THREADS.load(Ordering::Relaxed)
        }
        n => n,
    }
}

/// Request `threads` workers; the engine is rebuilt on its next use.
pub fn set_threads(threads: usize) {
    REQUESTED_THREADS.store(threads.max(1), Ordering::Relaxed);
}

/// Build an engine with `threads` workers, or a serial engine if the pool cannot be built.
fn build(threads: usize) -> Shared {
    let pool = (threads > 1)
        .then(|| tenferro_cpu::Pool::builder().threads(threads).build().ok())
        .flatten();
    Shared {
        engine: pool.map_or_else(CpuEngine::serial, CpuEngine::new),
        requested: threads,
        programs: TracedPrograms::default(),
    }
}

/// Run `f` with exclusive access to the shared engine.
///
/// The engine is (re)built when it does not exist yet, when the requested thread
/// count changed, or when an earlier panic left it broken.
pub fn with_engine<R>(f: impl FnOnce(&mut CpuEngine) -> R) -> R {
    with_engine_programs(|engine, _| f(engine))
}

/// Like [`with_engine`], also lending the engine's traced runtime and prepared programs.
pub fn with_engine_programs<R>(f: impl FnOnce(&mut CpuEngine, &mut TracedPrograms) -> R) -> R {
    let mut guard = ENGINE.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
    let threads = requested_threads();
    let stale = match guard.as_ref() {
        None => true,
        Some(shared) => shared.engine.is_broken() || shared.requested != threads,
    };
    if stale {
        *guard = Some(build(threads));
    }
    let shared = guard.as_mut().expect("engine was just built");
    f(&mut shared.engine, &mut shared.programs)
}

/// The number of worker threads of the shared engine (building it if needed).
pub fn engine_threads() -> usize {
    with_engine(|engine| engine.num_threads())
}

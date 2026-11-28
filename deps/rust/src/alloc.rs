//! A global allocator that keeps recently freed large blocks for reuse.
//!
//! Every differentiation step allocates and frees tensors of the same sizes:
//! inputs, leaf copies, residuals, gradients. With the system allocator each
//! large free returns its pages to the OS (`madvise` on macOS) and the next
//! allocation of the same size faults them in again, which costs more than the
//! copies into them. Blocks of [`MIN`]..=[`MAX`] bytes are therefore kept in a
//! small cache (at most [`SLOTS`] blocks and [`BUDGET`] bytes) and handed out
//! again for a request of the same rounded size. Everything else goes straight
//! to the system allocator.
//!
//! Measured on the MNIST training steps of TenferroLux (i9-9900K, macOS), with
//! the cache switched on and off in one process: the eager MLP step went from
//! 1.40 to 1.27 ms and the eager CNN step from 24.0 to 19.2 ms at one thread.

use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, Ordering};

/// Smallest cached block.
const MIN: usize = 32 << 10;
/// Largest cached block.
const MAX: usize = 64 << 20;
/// Cached blocks at most.
const SLOTS: usize = 64;
/// Cached bytes at most.
const BUDGET: usize = 256 << 20;
/// Alignment of every cached block (covers every Rust type's alignment we cache).
const ALIGN: usize = 64;
/// Size granularity of cached blocks.
const GRAIN: usize = 4 << 10;

pub struct CachingAlloc;

struct Cache {
    lock: AtomicBool,
    /// `(rounded size, address)`; size 0 is an empty slot.
    slots: std::cell::UnsafeCell<[(usize, usize); SLOTS]>,
    bytes: std::cell::UnsafeCell<usize>,
}

// SAFETY: `slots` and `bytes` are only touched while `lock` is held.
unsafe impl Sync for Cache {}

static CACHE: Cache = Cache {
    lock: AtomicBool::new(false),
    slots: std::cell::UnsafeCell::new([(0, 0); SLOTS]),
    bytes: std::cell::UnsafeCell::new(0),
};

impl Cache {
    fn with<R>(&self, f: impl FnOnce(&mut [(usize, usize); SLOTS], &mut usize) -> R) -> R {
        while self
            .lock
            .compare_exchange_weak(false, true, Ordering::Acquire, Ordering::Relaxed)
            .is_err()
        {
            std::hint::spin_loop();
        }
        // SAFETY: the lock is held, so this is the only access.
        let result = unsafe { f(&mut *self.slots.get(), &mut *self.bytes.get()) };
        self.lock.store(false, Ordering::Release);
        result
    }
}

/// The rounded size of a cacheable layout.
fn cached_size(layout: Layout) -> Option<usize> {
    (layout.size() >= MIN && layout.size() <= MAX && layout.align() <= ALIGN)
        .then(|| layout.size().div_ceil(GRAIN) * GRAIN)
}

fn block(size: usize) -> Layout {
    // SAFETY: `size` is a nonzero multiple of `GRAIN` below `MAX`, and `ALIGN` is a power of two.
    unsafe { Layout::from_size_align_unchecked(size, ALIGN) }
}

unsafe impl GlobalAlloc for CachingAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let Some(size) = cached_size(layout) else {
            return System.alloc(layout);
        };
        let hit = CACHE.with(|slots, bytes| {
            slots.iter_mut().find(|slot| slot.0 == size).map(|slot| {
                *bytes -= size;
                std::mem::take(slot).1
            })
        });
        match hit {
            Some(ptr) => ptr as *mut u8,
            None => System.alloc(block(size)),
        }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        if cached_size(layout).is_none() {
            return System.alloc_zeroed(layout);
        }
        let ptr = self.alloc(layout);
        if !ptr.is_null() {
            std::ptr::write_bytes(ptr, 0, layout.size());
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        let Some(size) = cached_size(layout) else {
            return System.dealloc(ptr, layout);
        };
        let kept = CACHE.with(|slots, bytes| {
            if *bytes + size > BUDGET {
                return false;
            }
            match slots.iter_mut().find(|slot| slot.0 == 0) {
                Some(slot) => {
                    *slot = (size, ptr as usize);
                    *bytes += size;
                    true
                }
                None => false,
            }
        });
        if !kept {
            System.dealloc(ptr, block(size));
        }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller guarantees `new_size` with `layout.align()` is a valid layout.
        let new_layout = Layout::from_size_align_unchecked(new_size, layout.align());
        if cached_size(layout).is_none() && cached_size(new_layout).is_none() {
            return System.realloc(ptr, layout, new_size);
        }
        let new = self.alloc(new_layout);
        if !new.is_null() {
            std::ptr::copy_nonoverlapping(ptr, new, layout.size().min(new_size));
            self.dealloc(ptr, layout);
        }
        new
    }
}

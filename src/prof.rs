//! Per-stage wall-clock accumulators for on-device profiling.
//! Each stage adds its elapsed nanoseconds; `synthesize` prints and resets
//! them once per render as a single `[PROF]` line.
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

pub static FLOWLM_STEP: AtomicU64 = AtomicU64::new(0); // transformer step only
pub static FLOWNET: AtomicU64 = AtomicU64::new(0);
pub static MIMI_UPSAMPLE: AtomicU64 = AtomicU64::new(0);
pub static MIMI_TRANSFORMER: AtomicU64 = AtomicU64::new(0);
pub static MIMI_SEANET: AtomicU64 = AtomicU64::new(0);

pub struct Timer(&'static AtomicU64, Instant);
impl Timer {
    pub fn start(slot: &'static AtomicU64) -> Self { Timer(slot, Instant::now()) }
}
impl Drop for Timer {
    fn drop(&mut self) { self.0.fetch_add(self.1.elapsed().as_nanos() as u64, Ordering::Relaxed); }
}

pub fn take_ms(slot: &'static AtomicU64) -> f64 { slot.swap(0, Ordering::Relaxed) as f64 / 1e6 }

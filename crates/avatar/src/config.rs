// Runtime configuration for avatar rendering.
// Simple global flag – using an AtomicBool avoids changing signatures throughout the code.
use std::sync::atomic::{AtomicBool, Ordering};

pub static MODEST: AtomicBool = AtomicBool::new(false);

/// Set the modest flag (e.g. from CLI).
pub fn set_modest(value: bool) {
    MODEST.store(value, Ordering::Relaxed);
}

//! Supporting utilities for the Kosmic Lab Holochain integration tests.

pub mod paths {
    use std::path::{Path, PathBuf};

    pub fn repo_root() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("..").join("..")
    }
}

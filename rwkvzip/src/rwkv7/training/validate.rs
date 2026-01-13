use anyhow::{bail, Context, Result};
use std::path::Path;

use crate::{CoderType, Compressor};

pub fn validate_roundtrip<P: AsRef<Path>>(model_path: P, input_path: P) -> Result<()> {
    let input_path = input_path.as_ref();
    let data = std::fs::read(input_path)
        .with_context(|| format!("Failed to read {}", input_path.display()))?;

    let mut compressor = Compressor::new(model_path.as_ref())?;

    let compressed = compressor
        .compress(&data, CoderType::RANS)
        .context("compress")?;
    let decompressed = compressor.decompress(&compressed).context("decompress")?;

    if decompressed != data {
        bail!("Roundtrip mismatch for {}", input_path.display());
    }
    Ok(())
}

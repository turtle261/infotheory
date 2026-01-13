use anyhow::{bail, Context, Result};
use memmap2::Mmap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha12Rng;
use std::fs::File;
use std::path::Path;
use tch::{Device, Kind, Tensor};

pub struct Enwik8Mmap {
    mmap: Mmap,
}

impl Enwik8Mmap {
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path.as_ref())
            .with_context(|| format!("Failed to open dataset file: {}", path.as_ref().display()))?;
        // Safety: file is kept alive by mmap object.
        let mmap = unsafe { Mmap::map(&file) }.context("Failed to mmap dataset")?;
        if mmap.len() < 2 {
            bail!("Dataset too small ({} bytes)", mmap.len());
        }
        Ok(Self { mmap })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.mmap.len()
    }

    /// Sample a batch of token sequences.
    ///
    /// Returns:
    /// - x: (B, T) int64
    /// - y: (B, T) int64 (next-token)
    pub fn sample_batch(
        &self,
        rng: &mut ChaCha12Rng,
        batch_size: i64,
        seq_len: i64,
        device: Device,
    ) -> Result<(Tensor, Tensor)> {
        let b = batch_size as usize;
        let t = seq_len as usize;
        if t < 1 {
            bail!("seq_len must be >= 1");
        }
        // Need T+1 bytes to form next-token targets.
        if self.len() < t + 1 {
            bail!("Dataset smaller than seq_len+1");
        }

        let max_start = self.len() - (t + 1);
        let mut x_buf = vec![0i64; b * t];
        let mut y_buf = vec![0i64; b * t];

        for bi in 0..b {
            let start = rng.gen_range(0..=max_start);
            let slice = &self.mmap[start..start + t + 1];
            for ti in 0..t {
                let x = slice[ti] as i64;
                let y = slice[ti + 1] as i64;
                x_buf[bi * t + ti] = x;
                y_buf[bi * t + ti] = y;
            }
        }

        let x = Tensor::from_slice(&x_buf)
            .to_kind(Kind::Int64)
            .view([batch_size, seq_len])
            .to_device(device);
        let y = Tensor::from_slice(&y_buf)
            .to_kind(Kind::Int64)
            .view([batch_size, seq_len])
            .to_device(device);
        Ok((x, y))
    }

    pub fn seeded_rng(seed: u64) -> ChaCha12Rng {
        ChaCha12Rng::seed_from_u64(seed)
    }
}

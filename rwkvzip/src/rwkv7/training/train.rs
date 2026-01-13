use super::data::Enwik8Mmap;
use super::export::export_safetensors;
use super::model::TrainModelConfig;
use super::model_fast::{FastTrainModel, FastTrainState};
use super::validate::validate_roundtrip;
use anyhow::{bail, Context, Result};
use std::path::PathBuf;
use std::time::Instant;
use tch::{no_grad, Device, Tensor};

// Helper functions for optimizer state saving/loading
fn tensor_to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    let t = t
        .to_device(Device::Cpu)
        .to_kind(tch::Kind::Float)
        .contiguous();
    let n = t.numel();
    let mut out = vec![0f32; n as usize];
    t.f_copy_data(&mut out, n)
        .context("Failed to copy tensor data")?;
    Ok(out)
}

fn tensor_shape_usize(t: &Tensor) -> Vec<usize> {
    t.size().iter().map(|&d| d as usize).collect()
}

#[derive(Debug, Clone)]
pub struct TrainConfig {
    pub dataset_path: PathBuf,
    pub output_model_path: PathBuf,
    pub resume_model_path: Option<PathBuf>,

    pub steps: usize,
    pub batch_size: i64,
    pub seq_len: i64,
    pub grad_accum_steps: usize,

    pub lr: f64,
    pub weight_decay: f64,
    pub beta1: f64,
    pub beta2: f64,
    pub adam_eps: f64,

    pub seed: u64,
    pub device: Option<String>,

    pub validate_roundtrip_path: Option<PathBuf>,

    pub model_cfg: TrainModelConfig,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            dataset_path: PathBuf::from("files/enwik8"),
            output_model_path: PathBuf::from("out/rwkv7_byte_small.safetensors"),
            resume_model_path: None,
            steps: 2_000,
            batch_size: 32,
            seq_len: 128,
            grad_accum_steps: 1,
            lr: 2e-4,
            weight_decay: 0.1,
            beta1: 0.9,
            beta2: 0.99,
            adam_eps: 1e-8,
            seed: 42,
            device: None,
            validate_roundtrip_path: Some(PathBuf::from("files/bench.txt")),
            model_cfg: TrainModelConfig::small_default(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TrainReport {
    pub device: String,
    pub steps: usize,
    pub tokens_per_step: i64,
    pub final_loss: f64,
    pub steps_per_sec: f64,
    pub tokens_per_sec: f64,
    pub output_model_path: PathBuf,
}

struct AdamWEntry {
    p: Tensor,
    m: Tensor,
    v: Tensor,
}

struct AdamW {
    entries: Vec<AdamWEntry>,
    beta1: f64,
    beta2: f64,
    eps: f64,
    weight_decay: f64,
    step: i64,
}

impl AdamW {
    fn new(params: Vec<Tensor>, beta1: f64, beta2: f64, eps: f64, weight_decay: f64) -> Self {
        let mut entries = Vec::with_capacity(params.len());
        for p in params {
            let m = Tensor::zeros_like(&p);
            let v = Tensor::zeros_like(&p);
            entries.push(AdamWEntry { p, m, v });
        }
        Self {
            entries,
            beta1,
            beta2,
            eps,
            weight_decay,
            step: 0,
        }
    }

    /// Create a new AdamW optimizer and optionally load state from a checkpoint
    fn new_with_state(
        params: Vec<Tensor>,
        beta1: f64,
        beta2: f64,
        eps: f64,
        weight_decay: f64,
        checkpoint_path: Option<&std::path::Path>,
    ) -> Result<Self> {
        let mut optimizer = Self::new(params, beta1, beta2, eps, weight_decay);

        // Try to load optimizer state if checkpoint is provided
        if let Some(path) = checkpoint_path {
            if path.exists() {
                optimizer.load_state(path)?;
                println!("Loaded optimizer state from checkpoint");
            } else {
                println!(
                    "Warning: Optimizer checkpoint not found, starting with fresh optimizer state"
                );
            }
        }

        Ok(optimizer)
    }

    /// Save optimizer state to a checkpoint file
    fn save_state(&self, path: &std::path::Path) -> Result<()> {
        use std::collections::BTreeMap;
        use std::fs;
        use std::io::Write;

        let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();

        // Save step count
        tensors.insert(
            "optimizer.step".to_string(),
            (vec![1], vec![self.step as f32]),
        );

        // Save momentum and variance for each parameter
        for (i, entry) in self.entries.iter().enumerate() {
            let prefix = format!("optimizer.{}", i);

            // Save momentum (m)
            tensors.insert(
                format!("{}.m", prefix),
                (tensor_shape_usize(&entry.m), tensor_to_vec_f32(&entry.m)?),
            );

            // Save variance (v)
            tensors.insert(
                format!("{}.v", prefix),
                (tensor_shape_usize(&entry.v), tensor_to_vec_f32(&entry.v)?),
            );
        }

        // Write in the same format as safetensors
        let mut cursor: usize = 0;
        let mut meta_entries: Vec<String> = Vec::with_capacity(tensors.len());
        let mut data_blobs: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());

        for (name, (shape, data_f32)) in tensors.into_iter() {
            let byte_len = data_f32.len() * 4;
            let start = cursor;
            let end = cursor + byte_len;
            cursor = end;

            let mut bytes = Vec::with_capacity(byte_len);
            for v in data_f32 {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            data_blobs.push(bytes);

            let shape_json = shape
                .iter()
                .map(|d| d.to_string())
                .collect::<Vec<_>>()
                .join(",");
            let entry = format!(
                "\"{}\":{{\"dtype\":\"F32\",\"shape\":[{}],\"data_offsets\":[{},{}]}}",
                name, shape_json, start, end
            );
            meta_entries.push(entry);
        }

        let header_json = format!("{{\"__metadata__\":{{}},{} }}", meta_entries.join(","));
        let header_bytes = header_json.as_bytes();
        let header_len = header_bytes.len() as u64;

        let mut out = Vec::with_capacity(8 + header_bytes.len() + cursor);
        out.extend_from_slice(&header_len.to_le_bytes());
        out.extend_from_slice(header_bytes);
        for blob in data_blobs {
            out.extend_from_slice(&blob);
        }

        fs::write(path, out).with_context(|| {
            format!("Failed to write optimizer checkpoint to {}", path.display())
        })?;

        Ok(())
    }

    /// Load optimizer state from a checkpoint file
    fn load_state(&mut self, path: &std::path::Path) -> Result<()> {
        use std::fs::File;
        use std::io::Read;

        let mut file = File::open(path)
            .with_context(|| format!("Failed to open optimizer checkpoint: {}", path.display()))?;

        // Read header length
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Read JSON header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str = std::str::from_utf8(&header_bytes)
            .context("Invalid UTF-8 in optimizer checkpoint header")?;

        // Parse JSON header
        let json: serde_json::Value = serde_json::from_str(header_str)
            .context("Failed to parse optimizer checkpoint JSON header")?;

        let data_offset = 8 + header_len;

        // Load step count
        if let Some(step_info) = json.get("optimizer.step") {
            let offsets = step_info["data_offsets"]
                .as_array()
                .ok_or_else(|| anyhow::anyhow!("Invalid data_offsets for step"))?
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect::<Vec<usize>>();

            let byte_len = offsets[1] - offsets[0];
            let mut raw_bytes = vec![0u8; byte_len];
            use std::io::Seek;
            file.seek(std::io::SeekFrom::Start((data_offset + offsets[0]) as u64))?;
            file.read_exact(&mut raw_bytes)?;

            // Convert bytes to f32 and then to i64
            let step_f32 =
                f32::from_le_bytes([raw_bytes[0], raw_bytes[1], raw_bytes[2], raw_bytes[3]]);
            self.step = step_f32 as i64;
        }

        // Load momentum and variance for each parameter
        for (i, entry) in self.entries.iter_mut().enumerate() {
            let prefix = format!("optimizer.{}", i);

            // Load momentum (m)
            if let Some(m_info) = json.get(&format!("{}.m", prefix)) {
                let shape = m_info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid shape for momentum"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as i64)
                    .collect::<Vec<i64>>();

                let offsets = m_info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid data_offsets for momentum"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<usize>>();

                let byte_len = offsets[1] - offsets[0];
                let mut raw_bytes = vec![0u8; byte_len];
                use std::io::Seek;
                file.seek(std::io::SeekFrom::Start((data_offset + offsets[0]) as u64))?;
                file.read_exact(&mut raw_bytes)?;

                let mut data = vec![0f32; byte_len / 4];
                for j in 0..data.len() {
                    let base = j * 4;
                    data[j] = f32::from_le_bytes([
                        raw_bytes[base],
                        raw_bytes[base + 1],
                        raw_bytes[base + 2],
                        raw_bytes[base + 3],
                    ]);
                }

                entry.m = Tensor::f_from_slice(&data)?
                    .view(&*shape)
                    .to_device(entry.m.device());
            }

            // Load variance (v)
            if let Some(v_info) = json.get(&format!("{}.v", prefix)) {
                let shape = v_info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid shape for variance"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as i64)
                    .collect::<Vec<i64>>();

                let offsets = v_info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid data_offsets for variance"))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<usize>>();

                let byte_len = offsets[1] - offsets[0];
                let mut raw_bytes = vec![0u8; byte_len];
                use std::io::Seek;
                file.seek(std::io::SeekFrom::Start((data_offset + offsets[0]) as u64))?;
                file.read_exact(&mut raw_bytes)?;

                let mut data = vec![0f32; byte_len / 4];
                for j in 0..data.len() {
                    let base = j * 4;
                    data[j] = f32::from_le_bytes([
                        raw_bytes[base],
                        raw_bytes[base + 1],
                        raw_bytes[base + 2],
                        raw_bytes[base + 3],
                    ]);
                }

                entry.v = Tensor::f_from_slice(&data)?
                    .view(&*shape)
                    .to_device(entry.v.device());
            }
        }

        Ok(())
    }

    fn zero_grad(&mut self) {
        for e in &mut self.entries {
            e.p.zero_grad();
        }
    }

    fn step(&mut self, lr: f64) {
        self.step += 1;
        let t = self.step as f64;
        let bias_c1 = 1.0 - self.beta1.powf(t);
        let bias_c2 = 1.0 - self.beta2.powf(t);
        let step_size = lr * (bias_c2.sqrt() / bias_c1);

        no_grad(|| {
            for e in &mut self.entries {
                let g = e.p.grad();
                if !g.defined() {
                    continue;
                }

                // m = b1*m + (1-b1)*g
                e.m = &e.m * self.beta1 + &g * (1.0 - self.beta1);
                // v = b2*v + (1-b2)*g^2
                e.v = &e.v * self.beta2 + g.square() * (1.0 - self.beta2);

                // denom = sqrt(v) + eps
                let denom = e.v.sqrt() + self.eps;

                // Decoupled weight decay
                if self.weight_decay != 0.0 {
                    let wd = &e.p * (-lr * self.weight_decay);
                    // p += (-lr*wd) * p
                    let _ = e.p.f_add_(&wd);
                }

                // p -= step_size * m / denom
                let update = (&e.m / denom) * (-step_size);
                let _ = e.p.f_add_(&update);
            }
        });

        self.zero_grad();
    }
}

fn pick_device(requested: &Option<String>) -> Result<Device> {
    let cuda_available = tch::Cuda::is_available();

    if let Some(s) = requested.as_deref() {
        match s {
            "cpu" => return Ok(Device::Cpu),
            "cuda" | "gpu" => {
                if cuda_available {
                    return Ok(Device::Cuda(0));
                }
                bail!("CUDA was requested but is not available. Set TORCH_CUDA_VERSION=121 (or 118) and rebuild, or pass --device cpu to force CPU.");
            }
            other => bail!("Unknown device '{}'. Use cpu or cuda", other),
        }
    }

    if cuda_available {
        Ok(Device::Cuda(0))
    } else {
        bail!(
            "CUDA not available. Training expects a CUDA-enabled libtorch. Install/rebuild with TORCH_CUDA_VERSION=121 (or 118) or rerun with --device cpu explicitly."
        )
    }
}

pub fn train_enwik8(cfg: TrainConfig) -> Result<TrainReport> {
    cfg.model_cfg.validate()?;

    let device = pick_device(&cfg.device)?;
    let device_str = match device {
        Device::Cpu => "cpu".to_string(),
        Device::Cuda(_) => {
            // Enable cuDNN benchmark mode for better performance
            tch::Cuda::cudnn_set_benchmark(true);
            "cuda:0".to_string()
        }
        _ => format!("{:?}", device),
    };

    // Ensure output dir exists
    if let Some(parent) = cfg.output_model_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create output dir {}", parent.display()))?;
    }

    let ds = Enwik8Mmap::open(&cfg.dataset_path)?;
    let mut rng = Enwik8Mmap::seeded_rng(cfg.seed);

    // Create model - either from scratch or by loading existing weights
    let model = if let Some(resume_path) = &cfg.resume_model_path {
        println!("Loading model weights from {}...", resume_path.display());
        FastTrainModel::load_from_safetensors(resume_path, cfg.model_cfg.clone(), device)
            .with_context(|| format!("Failed to load model from {}", resume_path.display()))?
    } else {
        FastTrainModel::new(cfg.model_cfg.clone(), device, cfg.seed as i64)?
    };

    let mut state = FastTrainState::new(&cfg.model_cfg, cfg.batch_size, device)?;

    let params = model.parameters();

    // Determine optimizer checkpoint path
    let optimizer_checkpoint_path = cfg.output_model_path.with_extension("opt.safetensors");

    // Create optimizer checkpoint path for loading (if resuming)
    let resume_optimizer_path = cfg
        .resume_model_path
        .as_ref()
        .map(|p| p.with_extension("opt.safetensors"));

    let mut opt = AdamW::new_with_state(
        params,
        cfg.beta1,
        cfg.beta2,
        cfg.adam_eps,
        cfg.weight_decay,
        resume_optimizer_path.as_deref(),
    )?;

    let accum_steps = cfg.grad_accum_steps.max(1);
    let tokens_per_step = cfg.batch_size * cfg.seq_len * (accum_steps as i64);
    let start = Instant::now();
    let mut last_loss = 0.0f64;

    for step in 0..cfg.steps {
        let mut step_loss_sum = 0.0f64;

        for micro_step in 0..accum_steps {
            state.reset();

            let (x, y) = ds.sample_batch(&mut rng, cfg.batch_size, cfg.seq_len, device)?;

            // Forward entire sequence at once (sequence-parallel)
            let logits = model.forward_sequence(&x, &mut state)?; // (B, T, V)
            let targets = y; // (B,T)

            // Cross-entropy over all positions, normalized by accum_steps for correct gradient scale
            let loss = logits
                .view([-1, cfg.model_cfg.vocab_size])
                .cross_entropy_for_logits(&targets.view([-1]))
                / (accum_steps as f64);

            loss.backward();

            // Accumulate loss for logging
            step_loss_sum += loss.to_device(Device::Cpu).double_value(&[]) * (accum_steps as f64);

            // Only step optimizer on last micro-step
            if micro_step == accum_steps - 1 {
                opt.step(cfg.lr);
            }
        }

        last_loss = step_loss_sum / (accum_steps as f64);

        if step % 50 == 0 {
            let elapsed = start.elapsed().as_secs_f64().max(1e-9);
            let sps = (step.max(1) as f64) / elapsed;
            let tps = sps * (tokens_per_step as f64);
            eprintln!(
                "step {:6} | loss {:8.4} | {:8.2} steps/s | {:10.0} tok/s | {}",
                step, last_loss, sps, tps, device_str
            );

            // Save optimizer checkpoint every 50 steps
            if let Err(e) = opt.save_state(&optimizer_checkpoint_path) {
                eprintln!("Warning: Failed to save optimizer checkpoint: {}", e);
            }
        }
    }

    // Export weights
    export_safetensors(&cfg.output_model_path, &cfg.model_cfg, &model.p)
        .context("export safetensors")?;

    // Save final optimizer state
    if let Err(e) = opt.save_state(&optimizer_checkpoint_path) {
        eprintln!("Warning: Failed to save final optimizer checkpoint: {}", e);
    }

    // Optional end-to-end compression validation.
    if let Some(p) = &cfg.validate_roundtrip_path {
        validate_roundtrip(&cfg.output_model_path, p)
            .with_context(|| format!("validate roundtrip using {}", p.display()))?;
    }

    let elapsed = start.elapsed().as_secs_f64().max(1e-9);
    let steps_per_sec = (cfg.steps as f64) / elapsed;
    let tokens_per_sec = steps_per_sec * (tokens_per_step as f64);

    Ok(TrainReport {
        device: device_str,
        steps: cfg.steps,
        tokens_per_step,
        final_loss: last_loss,
        steps_per_sec,
        tokens_per_sec,
        output_model_path: cfg.output_model_path,
    })
}

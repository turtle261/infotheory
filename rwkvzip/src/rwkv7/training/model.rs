use anyhow::{bail, Context, Result};
use tch::{kind::Kind, Device, Tensor};

/// Training-side RWKV7 configuration.
///
/// Must remain compatible with the inference assumptions:
/// - head_dim = 64
/// - hidden_size = num_heads * 64
#[derive(Debug, Clone)]
pub struct TrainModelConfig {
    pub vocab_size: i64,
    pub hidden_size: i64,
    pub num_layers: usize,
    pub num_heads: i64,
    pub head_dim: i64,
    pub intermediate_size: i64,

    pub layer_norm_eps: f64,
    pub group_norm_eps: f64,

    pub decay_low_rank: i64,
    pub a_low_rank: i64,
    pub v_low_rank: i64,
    pub g_low_rank: i64,
}

impl TrainModelConfig {
    pub fn small_default() -> Self {
        // ~ <10M params, byte-level
        let hidden_size = 256;
        let head_dim = 64;
        let num_heads = hidden_size / head_dim;
        Self {
            vocab_size: 256,
            hidden_size,
            num_layers: 6,
            num_heads,
            head_dim,
            intermediate_size: 1024,
            layer_norm_eps: 1e-5,
            group_norm_eps: 64e-5,
            decay_low_rank: 32,
            a_low_rank: 32,
            v_low_rank: 32,
            g_low_rank: 64,
        }
    }

    pub fn validate(&self) -> Result<()> {
        if self.vocab_size != 256 {
            bail!("This trainer is currently byte-level only (vocab_size=256)");
        }
        if self.head_dim != 64 {
            bail!("Inference path assumes head_dim=64 (got {})", self.head_dim);
        }
        if self.hidden_size != self.num_heads * self.head_dim {
            bail!(
                "hidden_size must equal num_heads*head_dim ({} != {}*{})",
                self.hidden_size,
                self.num_heads,
                self.head_dim
            );
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct LayerParams {
    pub pre_norm_w: Option<Tensor>,
    pub pre_norm_b: Option<Tensor>,

    pub attn_norm_w: Tensor,
    pub attn_norm_b: Tensor,
    pub ffn_norm_w: Tensor,
    pub ffn_norm_b: Tensor,

    // Attention mixing vectors (C,)
    pub attn_x_r: Tensor,
    pub attn_x_w: Tensor,
    pub attn_x_k: Tensor,
    pub attn_x_v: Tensor,
    pub attn_x_a: Tensor,
    pub attn_x_g: Tensor,

    // Projections (C,C)
    pub attn_r_proj: Tensor,
    pub attn_k_proj: Tensor,
    pub attn_v_proj: Tensor,
    pub attn_o_proj: Tensor,

    // LoRA low-rank blocks
    pub attn_w1: Tensor, // (D_w, C)
    pub attn_w2: Tensor, // (C, D_w)
    pub attn_w0: Tensor, // (C,)

    pub attn_a1: Tensor, // (D_a, C)
    pub attn_a2: Tensor, // (C, D_a)
    pub attn_a0: Tensor, // (C,)

    // v_lora exists for layers > 0
    pub attn_v1: Option<Tensor>, // (D_v, C)
    pub attn_v2: Option<Tensor>, // (C, D_v)
    pub attn_v0: Option<Tensor>, // (C,)

    pub attn_g1: Tensor, // (D_g, C)
    pub attn_g2: Tensor, // (C, D_g)

    pub attn_k_k: Tensor, // (C,)
    pub attn_k_a: Tensor, // (C,)
    pub attn_r_k: Tensor, // (H, N)

    pub attn_gn_w: Tensor, // (C,)
    pub attn_gn_b: Tensor, // (C,)

    // FFN
    pub ffn_x_k: Tensor,     // (C,)
    pub ffn_key_w: Tensor,   // (I, C)
    pub ffn_value_w: Tensor, // (C, I)
}

#[derive(Debug)]
pub struct TrainParams {
    pub embeddings: Tensor, // (V, C)
    pub ln_out_w: Tensor,   // (C,)
    pub ln_out_b: Tensor,   // (C,)
    pub lm_head: Tensor,    // (V, C)
    pub layers: Vec<LayerParams>,
}

impl TrainParams {
    pub fn init(cfg: &TrainModelConfig, device: Device, seed: i64) -> Result<Self> {
        cfg.validate()?;
        tch::manual_seed(seed);

        let v = cfg.vocab_size;
        let c = cfg.hidden_size;
        let h = cfg.num_heads;
        let n = cfg.head_dim;
        let i = cfg.intermediate_size;

        let d_w = cfg.decay_low_rank;
        let d_a = cfg.a_low_rank;
        let d_v = cfg.v_low_rank;
        let d_g = cfg.g_low_rank;

        let kf = Kind::Float;

        let embeddings = Tensor::randn([v, c], (kf, device)) * 0.02;
        let _ = embeddings.set_requires_grad(true);

        let ln_out_w = Tensor::ones([c], (kf, device));
        let _ = ln_out_w.set_requires_grad(true);
        let ln_out_b = Tensor::zeros([c], (kf, device));
        let _ = ln_out_b.set_requires_grad(true);

        let lm_head = Tensor::randn([v, c], (kf, device)) * 0.02;
        let _ = lm_head.set_requires_grad(true);

        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let is_first = layer_idx == 0;

            let mut pre_norm_w = None;
            let mut pre_norm_b = None;
            if is_first {
                let w = Tensor::ones([c], (kf, device));
                let _ = w.set_requires_grad(true);
                let b = Tensor::zeros([c], (kf, device));
                let _ = b.set_requires_grad(true);
                pre_norm_w = Some(w);
                pre_norm_b = Some(b);
            }

            let attn_norm_w = Tensor::ones([c], (kf, device));
            let _ = attn_norm_w.set_requires_grad(true);
            let attn_norm_b = Tensor::zeros([c], (kf, device));
            let _ = attn_norm_b.set_requires_grad(true);

            let ffn_norm_w = Tensor::ones([c], (kf, device));
            let _ = ffn_norm_w.set_requires_grad(true);
            let ffn_norm_b = Tensor::zeros([c], (kf, device));
            let _ = ffn_norm_b.set_requires_grad(true);

            // mixing vectors: initialize close to 0 (so x_mix ~= x)
            let mix_init = |c: i64| -> Tensor {
                let t = Tensor::zeros([c], (kf, device));
                let _ = t.set_requires_grad(true);
                t
            };

            let attn_x_r = mix_init(c);
            let attn_x_w = mix_init(c);
            let attn_x_k = mix_init(c);
            let attn_x_v = mix_init(c);
            let attn_x_a = mix_init(c);
            let attn_x_g = mix_init(c);

            // linear weights
            let attn_r_proj = Tensor::randn([c, c], (kf, device)) * (1.0 / (c as f64).sqrt());
            let _ = attn_r_proj.set_requires_grad(true);
            let attn_k_proj = Tensor::randn([c, c], (kf, device)) * (1.0 / (c as f64).sqrt());
            let _ = attn_k_proj.set_requires_grad(true);
            let attn_v_proj = Tensor::randn([c, c], (kf, device)) * (1.0 / (c as f64).sqrt());
            let _ = attn_v_proj.set_requires_grad(true);
            let attn_o_proj = Tensor::randn([c, c], (kf, device)) * (1.0 / (c as f64).sqrt());
            let _ = attn_o_proj.set_requires_grad(true);

            // low-rank blocks
            let attn_w1 = Tensor::randn([d_w, c], (kf, device)) * 0.02;
            let _ = attn_w1.set_requires_grad(true);
            let attn_w2 = Tensor::randn([c, d_w], (kf, device)) * 0.02;
            let _ = attn_w2.set_requires_grad(true);
            let attn_w0 = Tensor::zeros([c], (kf, device));
            let _ = attn_w0.set_requires_grad(true);

            let attn_a1 = Tensor::randn([d_a, c], (kf, device)) * 0.02;
            let _ = attn_a1.set_requires_grad(true);
            let attn_a2 = Tensor::randn([c, d_a], (kf, device)) * 0.02;
            let _ = attn_a2.set_requires_grad(true);
            let attn_a0 = Tensor::zeros([c], (kf, device));
            let _ = attn_a0.set_requires_grad(true);

            let (attn_v1, attn_v2, attn_v0) = if is_first {
                (None, None, None)
            } else {
                let v1 = Tensor::randn([d_v, c], (kf, device)) * 0.02;
                let _ = v1.set_requires_grad(true);
                let v2 = Tensor::randn([c, d_v], (kf, device)) * 0.02;
                let _ = v2.set_requires_grad(true);
                let v0 = Tensor::zeros([c], (kf, device));
                let _ = v0.set_requires_grad(true);
                (Some(v1), Some(v2), Some(v0))
            };

            let attn_g1 = Tensor::randn([d_g, c], (kf, device)) * 0.02;
            let _ = attn_g1.set_requires_grad(true);
            let attn_g2 = Tensor::randn([c, d_g], (kf, device)) * 0.02;
            let _ = attn_g2.set_requires_grad(true);

            let attn_k_k = Tensor::ones([c], (kf, device));
            let _ = attn_k_k.set_requires_grad(true);
            let attn_k_a = Tensor::zeros([c], (kf, device));
            let _ = attn_k_a.set_requires_grad(true);

            let attn_r_k = Tensor::zeros([h, n], (kf, device));
            let _ = attn_r_k.set_requires_grad(true);

            let attn_gn_w = Tensor::ones([c], (kf, device));
            let _ = attn_gn_w.set_requires_grad(true);
            let attn_gn_b = Tensor::zeros([c], (kf, device));
            let _ = attn_gn_b.set_requires_grad(true);

            // FFN
            let ffn_x_k = Tensor::zeros([c], (kf, device));
            let _ = ffn_x_k.set_requires_grad(true);

            let ffn_key_w = Tensor::randn([i, c], (kf, device)) * (1.0 / (c as f64).sqrt());
            let _ = ffn_key_w.set_requires_grad(true);
            let ffn_value_w = Tensor::randn([c, i], (kf, device)) * (1.0 / (i as f64).sqrt());
            let _ = ffn_value_w.set_requires_grad(true);

            layers.push(LayerParams {
                pre_norm_w,
                pre_norm_b,
                attn_norm_w,
                attn_norm_b,
                ffn_norm_w,
                ffn_norm_b,
                attn_x_r,
                attn_x_w,
                attn_x_k,
                attn_x_v,
                attn_x_a,
                attn_x_g,
                attn_r_proj,
                attn_k_proj,
                attn_v_proj,
                attn_o_proj,
                attn_w1,
                attn_w2,
                attn_w0,
                attn_a1,
                attn_a2,
                attn_a0,
                attn_v1,
                attn_v2,
                attn_v0,
                attn_g1,
                attn_g2,
                attn_k_k,
                attn_k_a,
                attn_r_k,
                attn_gn_w,
                attn_gn_b,
                ffn_x_k,
                ffn_key_w,
                ffn_value_w,
            });
        }

        Ok(Self {
            embeddings,
            ln_out_w,
            ln_out_b,
            lm_head,
            layers,
        })
    }

    /// Load training parameters from a safetensors file.
    ///
    /// This function loads model weights from a previously saved checkpoint
    /// and initializes them as trainable parameters on the specified device.
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        cfg: &TrainModelConfig,
        device: Device,
    ) -> Result<Self> {
        use std::collections::BTreeMap;
        use std::fs;
        use std::io::Read;

        cfg.validate()?;

        let path = path.as_ref();
        let mut file = std::fs::File::open(path)
            .with_context(|| format!("Failed to open safetensors file: {}", path.display()))?;

        // Read header length (8-byte little-endian u64)
        let mut header_len_bytes = [0u8; 8];
        file.read_exact(&mut header_len_bytes)?;
        let header_len = u64::from_le_bytes(header_len_bytes) as usize;

        // Read JSON header
        let mut header_bytes = vec![0u8; header_len];
        file.read_exact(&mut header_bytes)?;
        let header_str =
            std::str::from_utf8(&header_bytes).context("Invalid UTF-8 in safetensors header")?;

        // Parse JSON header to get tensor metadata
        let json: serde_json::Value =
            serde_json::from_str(header_str).context("Failed to parse safetensors JSON header")?;

        let data_offset = 8 + header_len;
        let kf = Kind::Float;

        // Helper function to load a tensor
        let mut load_tensor = |name: &str, expected_shape: Vec<i64>| -> Result<Tensor> {
            if let Some(tensor_info) = json.get(name) {
                let shape = tensor_info["shape"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid shape for tensor {}", name))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as i64)
                    .collect::<Vec<i64>>();

                let offsets = tensor_info["data_offsets"]
                    .as_array()
                    .ok_or_else(|| anyhow::anyhow!("Invalid data_offsets for tensor {}", name))?
                    .iter()
                    .map(|v| v.as_u64().unwrap() as usize)
                    .collect::<Vec<usize>>();

                if shape != expected_shape {
                    bail!(
                        "Shape mismatch for {}: expected {:?}, got {:?}",
                        name,
                        expected_shape,
                        shape
                    );
                }

                let byte_len = offsets[1] - offsets[0];
                let mut raw_bytes = vec![0u8; byte_len];

                use std::io::Seek;
                file.seek(std::io::SeekFrom::Start((data_offset + offsets[0]) as u64))?;
                file.read_exact(&mut raw_bytes)?;

                // Convert bytes to f32 and create tensor
                let mut data = vec![0f32; byte_len / 4];
                for i in 0..data.len() {
                    let base = i * 4;
                    data[i] = f32::from_le_bytes([
                        raw_bytes[base],
                        raw_bytes[base + 1],
                        raw_bytes[base + 2],
                        raw_bytes[base + 3],
                    ]);
                }

                let tensor = Tensor::f_from_slice(&data)?.view(&*shape).to_device(device);
                let _ = tensor.set_requires_grad(true);
                Ok(tensor)
            } else {
                bail!("Tensor {} not found in safetensors file", name);
            }
        };

        let v = cfg.vocab_size;
        let c = cfg.hidden_size;
        let h = cfg.num_heads;
        let n = cfg.head_dim;
        let i = cfg.intermediate_size;

        let d_w = cfg.decay_low_rank;
        let d_a = cfg.a_low_rank;
        let d_v = cfg.v_low_rank;
        let d_g = cfg.g_low_rank;

        // Load main parameters
        let embeddings = load_tensor("model.embeddings.weight", vec![v, c])?;
        let ln_out_w = load_tensor("model.norm.weight", vec![c])?;
        let ln_out_b = load_tensor("model.norm.bias", vec![c])?;
        let lm_head = load_tensor("lm_head.weight", vec![v, c])?;

        // Load layers
        let mut layers = Vec::with_capacity(cfg.num_layers);
        for layer_idx in 0..cfg.num_layers {
            let is_first = layer_idx == 0;
            let prefix = format!("model.layers.{}", layer_idx);

            let mut pre_norm_w = None;
            let mut pre_norm_b = None;
            if is_first {
                pre_norm_w = Some(load_tensor(
                    &format!("{}.pre_norm.weight", prefix),
                    vec![c],
                )?);
                pre_norm_b = Some(load_tensor(&format!("{}.pre_norm.bias", prefix), vec![c])?);
            }

            let attn_norm_w = load_tensor(&format!("{}.attn_norm.weight", prefix), vec![c])?;
            let attn_norm_b = load_tensor(&format!("{}.attn_norm.bias", prefix), vec![c])?;
            let ffn_norm_w = load_tensor(&format!("{}.ffn_norm.weight", prefix), vec![c])?;
            let ffn_norm_b = load_tensor(&format!("{}.ffn_norm.bias", prefix), vec![c])?;

            // Attention mixing vectors
            let attn_x_r = load_tensor(&format!("{}.attn.x_r", prefix), vec![c])?;
            let attn_x_w = load_tensor(&format!("{}.attn.x_w", prefix), vec![c])?;
            let attn_x_k = load_tensor(&format!("{}.attn.x_k", prefix), vec![c])?;
            let attn_x_v = load_tensor(&format!("{}.attn.x_v", prefix), vec![c])?;
            let attn_x_a = load_tensor(&format!("{}.attn.x_a", prefix), vec![c])?;
            let attn_x_g = load_tensor(&format!("{}.attn.x_g", prefix), vec![c])?;

            // Projections
            let attn_r_proj = load_tensor(&format!("{}.attn.r_proj.weight", prefix), vec![c, c])?;
            let attn_k_proj = load_tensor(&format!("{}.attn.k_proj.weight", prefix), vec![c, c])?;
            let attn_v_proj = load_tensor(&format!("{}.attn.v_proj.weight", prefix), vec![c, c])?;
            let attn_o_proj = load_tensor(&format!("{}.attn.o_proj.weight", prefix), vec![c, c])?;

            // LoRA blocks
            let attn_w1 = load_tensor(
                &format!("{}.attn.w_lora.lora.0.weight", prefix),
                vec![d_w, c],
            )?;
            let attn_w2 = load_tensor(
                &format!("{}.attn.w_lora.lora.2.weight", prefix),
                vec![c, d_w],
            )?;
            let attn_w0 = load_tensor(&format!("{}.attn.w_lora.lora.2.bias", prefix), vec![c])?;

            let attn_a1 = load_tensor(
                &format!("{}.attn.a_lora.lora.0.weight", prefix),
                vec![d_a, c],
            )?;
            let attn_a2 = load_tensor(
                &format!("{}.attn.a_lora.lora.2.weight", prefix),
                vec![c, d_a],
            )?;
            let attn_a0 = load_tensor(&format!("{}.attn.a_lora.lora.2.bias", prefix), vec![c])?;

            let (attn_v1, attn_v2, attn_v0) = if is_first {
                (None, None, None)
            } else {
                let v1 = Some(load_tensor(
                    &format!("{}.attn.v_lora.lora.0.weight", prefix),
                    vec![d_v, c],
                )?);
                let v2 = Some(load_tensor(
                    &format!("{}.attn.v_lora.lora.2.weight", prefix),
                    vec![c, d_v],
                )?);
                let v0 = Some(load_tensor(
                    &format!("{}.attn.v_lora.lora.2.bias", prefix),
                    vec![c],
                )?);
                (v1, v2, v0)
            };

            let attn_g1 = load_tensor(
                &format!("{}.attn.g_lora.lora.0.weight", prefix),
                vec![d_g, c],
            )?;
            let attn_g2 = load_tensor(
                &format!("{}.attn.g_lora.lora.2.weight", prefix),
                vec![c, d_g],
            )?;

            let attn_k_k = load_tensor(&format!("{}.attn.k_k", prefix), vec![c])?;
            let attn_k_a = load_tensor(&format!("{}.attn.k_a", prefix), vec![c])?;
            let attn_r_k = load_tensor(&format!("{}.attn.r_k", prefix), vec![h, n])?;

            let attn_gn_w = load_tensor(&format!("{}.attn.g_norm.weight", prefix), vec![c])?;
            let attn_gn_b = load_tensor(&format!("{}.attn.g_norm.bias", prefix), vec![c])?;

            // FFN
            let ffn_x_k = load_tensor(&format!("{}.ffn.x_k", prefix), vec![c])?;
            let ffn_key_w = load_tensor(&format!("{}.ffn.key.weight", prefix), vec![i, c])?;
            let ffn_value_w = load_tensor(&format!("{}.ffn.value.weight", prefix), vec![c, i])?;

            layers.push(LayerParams {
                pre_norm_w,
                pre_norm_b,
                attn_norm_w,
                attn_norm_b,
                ffn_norm_w,
                ffn_norm_b,
                attn_x_r,
                attn_x_w,
                attn_x_k,
                attn_x_v,
                attn_x_a,
                attn_x_g,
                attn_r_proj,
                attn_k_proj,
                attn_v_proj,
                attn_o_proj,
                attn_w1,
                attn_w2,
                attn_w0,
                attn_a1,
                attn_a2,
                attn_a0,
                attn_v1,
                attn_v2,
                attn_v0,
                attn_g1,
                attn_g2,
                attn_k_k,
                attn_k_a,
                attn_r_k,
                attn_gn_w,
                attn_gn_b,
                ffn_x_k,
                ffn_key_w,
                ffn_value_w,
            });
        }

        Ok(Self {
            embeddings,
            ln_out_w,
            ln_out_b,
            lm_head,
            layers,
        })
    }

    pub fn parameters(&self) -> Vec<Tensor> {
        let mut out = Vec::new();
        out.push(self.embeddings.shallow_clone());
        out.push(self.ln_out_w.shallow_clone());
        out.push(self.ln_out_b.shallow_clone());
        out.push(self.lm_head.shallow_clone());
        for (idx, l) in self.layers.iter().enumerate() {
            if idx == 0 {
                if let Some(w) = &l.pre_norm_w {
                    out.push(w.shallow_clone());
                }
                if let Some(b) = &l.pre_norm_b {
                    out.push(b.shallow_clone());
                }
            }
            out.push(l.attn_norm_w.shallow_clone());
            out.push(l.attn_norm_b.shallow_clone());
            out.push(l.ffn_norm_w.shallow_clone());
            out.push(l.ffn_norm_b.shallow_clone());

            out.push(l.attn_x_r.shallow_clone());
            out.push(l.attn_x_w.shallow_clone());
            out.push(l.attn_x_k.shallow_clone());
            out.push(l.attn_x_v.shallow_clone());
            out.push(l.attn_x_a.shallow_clone());
            out.push(l.attn_x_g.shallow_clone());

            out.push(l.attn_r_proj.shallow_clone());
            out.push(l.attn_k_proj.shallow_clone());
            out.push(l.attn_v_proj.shallow_clone());
            out.push(l.attn_o_proj.shallow_clone());

            out.push(l.attn_w1.shallow_clone());
            out.push(l.attn_w2.shallow_clone());
            out.push(l.attn_w0.shallow_clone());

            out.push(l.attn_a1.shallow_clone());
            out.push(l.attn_a2.shallow_clone());
            out.push(l.attn_a0.shallow_clone());

            if let Some(v1) = &l.attn_v1 {
                out.push(v1.shallow_clone());
            }
            if let Some(v2) = &l.attn_v2 {
                out.push(v2.shallow_clone());
            }
            if let Some(v0) = &l.attn_v0 {
                out.push(v0.shallow_clone());
            }

            out.push(l.attn_g1.shallow_clone());
            out.push(l.attn_g2.shallow_clone());

            out.push(l.attn_k_k.shallow_clone());
            out.push(l.attn_k_a.shallow_clone());
            out.push(l.attn_r_k.shallow_clone());
            out.push(l.attn_gn_w.shallow_clone());
            out.push(l.attn_gn_b.shallow_clone());

            out.push(l.ffn_x_k.shallow_clone());
            out.push(l.ffn_key_w.shallow_clone());
            out.push(l.ffn_value_w.shallow_clone());
        }
        out
    }
}

// NOTE: TrainState and TrainModel kept for reference; the FastTrainModel is now used.

#[derive(Debug)]
#[allow(dead_code)]
pub struct TrainState {
    pub att_x_prev: Vec<Tensor>, // (B,C)
    pub att_state: Vec<Tensor>,  // (B,H,N,N)
    pub ffn_x_prev: Vec<Tensor>, // (B,C)
    pub v_first: Tensor,         // (B,C)
    pub v_first_set: bool,
}

#[allow(dead_code)]
impl TrainState {
    pub fn new(cfg: &TrainModelConfig, batch_size: i64, device: Device) -> Result<Self> {
        cfg.validate()?;
        let b = batch_size;
        let c = cfg.hidden_size;
        let h = cfg.num_heads;
        let n = cfg.head_dim;
        let kf = Kind::Float;

        let mut att_x_prev = Vec::with_capacity(cfg.num_layers);
        let mut att_state = Vec::with_capacity(cfg.num_layers);
        let mut ffn_x_prev = Vec::with_capacity(cfg.num_layers);

        for _ in 0..cfg.num_layers {
            att_x_prev.push(Tensor::zeros([b, c], (kf, device)));
            att_state.push(Tensor::zeros([b, h, n, n], (kf, device)));
            ffn_x_prev.push(Tensor::zeros([b, c], (kf, device)));
        }

        Ok(Self {
            att_x_prev,
            att_state,
            ffn_x_prev,
            v_first: Tensor::zeros([b, c], (kf, device)),
            v_first_set: false,
        })
    }

    pub fn reset(&mut self) {
        // Important: detach recurrent tensors between optimizer steps.
        // Otherwise autograd will try to backprop through the previous step's graph.
        tch::no_grad(|| {
            for t in self.att_x_prev.iter_mut() {
                let detached = t.detach();
                *t = detached;
                let _ = t.zero_();
            }
            for t in self.att_state.iter_mut() {
                let detached = t.detach();
                *t = detached;
                let _ = t.zero_();
            }
            for t in self.ffn_x_prev.iter_mut() {
                let detached = t.detach();
                *t = detached;
                let _ = t.zero_();
            }
            self.v_first = self.v_first.detach();
            let _ = self.v_first.zero_();
        });

        self.v_first_set = false;
    }
}

#[derive(Debug)]
#[allow(dead_code)]
pub struct TrainModel {
    pub cfg: TrainModelConfig,
    pub p: TrainParams,
}

#[allow(dead_code)]
impl TrainModel {
    pub fn new(cfg: TrainModelConfig, device: Device, seed: i64) -> Result<Self> {
        let p = TrainParams::init(&cfg, device, seed)?;
        Ok(Self { cfg, p })
    }

    /// Forward one token for a batch.
    ///
    /// token: (B,) int64
    /// returns logits: (B, V)
    pub fn forward_token(&self, token: &Tensor, state: &mut TrainState) -> Result<Tensor> {
        let c = self.cfg.hidden_size;

        // embedding lookup
        let mut x = self.p.embeddings.index_select(0, token); // (B,C)

        for (layer_idx, layer) in self.p.layers.iter().enumerate() {
            if layer_idx == 0 {
                if let (Some(w), Some(b)) = (&layer.pre_norm_w, &layer.pre_norm_b) {
                    x = x.layer_norm(&[c], Some(w), Some(b), self.cfg.layer_norm_eps, false);
                }
            }

            // Attention norm
            let x_norm = x.layer_norm(
                &[c],
                Some(&layer.attn_norm_w),
                Some(&layer.attn_norm_b),
                self.cfg.layer_norm_eps,
                false,
            );

            let att_out = self.attention(layer_idx, layer, &x_norm, state)?;
            x = x + att_out;

            // FFN norm
            let x_norm = x.layer_norm(
                &[c],
                Some(&layer.ffn_norm_w),
                Some(&layer.ffn_norm_b),
                self.cfg.layer_norm_eps,
                false,
            );

            let ffn_out = self.ffn(layer_idx, layer, &x_norm, state)?;
            x = x + ffn_out;
        }

        // Output norm
        let x_norm = x.layer_norm(
            &[c],
            Some(&self.p.ln_out_w),
            Some(&self.p.ln_out_b),
            self.cfg.layer_norm_eps,
            false,
        );

        // logits = x @ lm_head.T
        Ok(x_norm.matmul(&self.p.lm_head.transpose(0, 1)))
    }

    fn token_shift(x: &Tensor, prev: &Tensor, mix: &Tensor) -> Tensor {
        // x + mix*(prev - x)
        x + mix * (prev - x)
    }

    fn attention(
        &self,
        layer_idx: usize,
        layer: &LayerParams,
        x_norm: &Tensor,
        state: &mut TrainState,
    ) -> Result<Tensor> {
        let b = x_norm.size()[0];
        let c = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;

        // Token shift mixes
        let prev = &state.att_x_prev[layer_idx];
        let xr = Self::token_shift(x_norm, prev, &layer.attn_x_r);
        let xw = Self::token_shift(x_norm, prev, &layer.attn_x_w);
        let xk = Self::token_shift(x_norm, prev, &layer.attn_x_k);
        let xv = Self::token_shift(x_norm, prev, &layer.attn_x_v);
        let xa = Self::token_shift(x_norm, prev, &layer.attn_x_a);
        let xg = Self::token_shift(x_norm, prev, &layer.attn_x_g);

        // update prev
        state.att_x_prev[layer_idx] = x_norm.shallow_clone();

        // r/k/v projections (B,C)
        let r = xr.matmul(&layer.attn_r_proj.transpose(0, 1));
        let mut k = xk.matmul(&layer.attn_k_proj.transpose(0, 1));
        let mut v = xv.matmul(&layer.attn_v_proj.transpose(0, 1));

        // w decay: w = exp(-sigmoid(w2 @ tanh(w1 @ xw) + w0) / sqrt(e))
        let tmp_w = (xw.matmul(&layer.attn_w1.transpose(0, 1))).tanh(); // (B, D_w)
        let w_pre = tmp_w.matmul(&layer.attn_w2.transpose(0, 1)) + &layer.attn_w0; // (B,C)
        let w_sig = w_pre.sigmoid();
        let inv_sqrt_e = 1.0f64 / std::f64::consts::E.sqrt();
        let w = (-w_sig * inv_sqrt_e).exp();

        // a = sigmoid(a2 @ (a1 @ xa) + a0)
        let tmp_a = xa.matmul(&layer.attn_a1.transpose(0, 1)); // (B, D_a)
        let a = (tmp_a.matmul(&layer.attn_a2.transpose(0, 1)) + &layer.attn_a0).sigmoid(); // (B,C)

        // g = (sigmoid(g1 @ xg) @ g2)
        let tmp_g = (xg.matmul(&layer.attn_g1.transpose(0, 1))).sigmoid(); // (B, D_g)
        let g = tmp_g.matmul(&layer.attn_g2.transpose(0, 1)); // (B,C)

        // Value residual
        if layer_idx == 0 {
            state.v_first = v.shallow_clone();
            state.v_first_set = true;
        } else if state.v_first_set {
            if let (Some(v1), Some(v2), Some(v0)) = (&layer.attn_v1, &layer.attn_v2, &layer.attn_v0)
            {
                let tmp = xv.matmul(&v1.transpose(0, 1)); // (B, D_v)
                let nu = (tmp.matmul(&v2.transpose(0, 1)) + v0).sigmoid(); // (B,C)
                v = &v + (&state.v_first - &v) * nu;
            }
        }

        // reshape to heads
        let r_h = r.view([b, h, n]);
        let v_h = v.view([b, h, n]);
        let w_h = w.view([b, h, n]);
        let a_h = a.view([b, h, n]);
        let g_h = g.view([b, h, n]);

        // kk = normalize(k * k_k)
        let kk = (k.view([b, c]) * &layer.attn_k_k).view([b, h, n]);
        let kk_norm = {
            let denom =
                (kk.square().sum_dim_intlist(&[-1i64][..], true, Kind::Float) + 1e-12).sqrt();
            &kk / denom
        };

        // k = k * (1 + (a - 1) * k_a)
        let scale = (a.view([b, c]) - 1.0) * &layer.attn_k_a + 1.0;
        k = (k.view([b, c]) * scale).view([b, h, n]);
        let k_h = k;

        // WKV update (batched)
        let mut s = state.att_state[layer_idx].shallow_clone(); // (B,H,N,N)
                                                                // S = S*w.T (multiply columns)
        s = s * w_h.unsqueeze(-2);
        // u = S @ kk
        let s2 = s.view([b * h, n, n]);
        let kk2 = kk_norm.view([b * h, n, 1]);
        let u = s2.bmm(&kk2).view([b, h, n]);
        let kka = &kk_norm * &a_h;
        // S = S - u*(kk*a).T + v*k.T
        let sub = u.unsqueeze(-1) * kka.unsqueeze(-2);
        let add = v_h.unsqueeze(-1) * k_h.unsqueeze(-2);
        s = s - sub + add;
        // y = S @ r
        let y = s
            .view([b * h, n, n])
            .bmm(&r_h.view([b * h, n, 1]))
            .view([b, h, n]);

        // write back state
        state.att_state[layer_idx] = s;

        // group norm per head (manual)
        let mut y_gn = {
            let mean = y.mean_dim(&[-1i64][..], true, Kind::Float);
            let yc = &y - &mean;
            let var = yc.square().mean_dim(&[-1i64][..], true, Kind::Float);
            yc / (var + self.cfg.group_norm_eps).sqrt()
        };
        let gn_w = layer.attn_gn_w.view([1, h, n]);
        let gn_b = layer.attn_gn_b.view([1, h, n]);
        y_gn = y_gn * gn_w + gn_b;

        // head-qk term: y += alpha * v
        let r_k = layer.attn_r_k.view([1, h, n]);
        let alpha = (r_h * &k_h * r_k).sum_dim_intlist(&[-1i64][..], true, Kind::Float); // (B,H,1)
        let mut y2 = y_gn + alpha * v_h;

        // apply gate
        y2 = y2 * g_h;

        // output projection
        let y_flat = y2.view([b, c]);
        Ok(y_flat.matmul(&layer.attn_o_proj.transpose(0, 1)))
    }

    fn ffn(
        &self,
        layer_idx: usize,
        layer: &LayerParams,
        x_norm: &Tensor,
        state: &mut TrainState,
    ) -> Result<Tensor> {
        let prev = &state.ffn_x_prev[layer_idx];
        let xk = Self::token_shift(x_norm, prev, &layer.ffn_x_k);
        state.ffn_x_prev[layer_idx] = x_norm.shallow_clone();

        // k = relu(xk @ key_w.T)^2
        let k = xk.matmul(&layer.ffn_key_w.transpose(0, 1)).relu();
        let k2 = &k * &k;

        // out = k @ value_w.T
        Ok(k2.matmul(&layer.ffn_value_w.transpose(0, 1)))
    }
}

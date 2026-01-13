//! High-performance sequence-parallel RWKV7 training model.
//!
//! Key optimizations:
//! - All linear projections batched across the full sequence (B,T,C) @ (C,C) = (B,T,C)
//! - Token-shift computed via tensor slicing, not per-token loops
//! - WKV uses fused kernel to minimize kernel launches
//! - Minimizes tensor allocations by reusing where possible

use anyhow::Result;
use tch::{kind::Kind, Device, IndexOp, Tensor};

use super::model::{LayerParams, TrainModelConfig, TrainParams};
use super::wkv_fused;

/// Sequence-parallel training state.
/// Unlike the token-by-token TrainState, this only holds the recurrent WKV state
/// since token-shift "prev" is handled via tensor slicing.
#[derive(Debug)]
pub struct FastTrainState {
    /// WKV state per layer: (B, H, N, N)
    pub wkv_state: Vec<Tensor>,
    /// v_first for value residual: (B, C)
    pub v_first: Option<Tensor>,
}

impl FastTrainState {
    pub fn new(cfg: &TrainModelConfig, batch_size: i64, device: Device) -> Result<Self> {
        let b = batch_size;
        let h = cfg.num_heads;
        let n = cfg.head_dim;
        let kf = Kind::Float;

        let wkv_state: Vec<Tensor> = (0..cfg.num_layers)
            .map(|_| Tensor::zeros([b, h, n, n], (kf, device)))
            .collect();

        Ok(Self {
            wkv_state,
            v_first: None,
        })
    }

    pub fn reset(&mut self) {
        tch::no_grad(|| {
            for s in &mut self.wkv_state {
                *s = s.detach();
                let _ = s.zero_();
            }
            self.v_first = None;
        });
    }
}

/// Fast sequence-parallel model wrapper.
pub struct FastTrainModel {
    pub cfg: TrainModelConfig,
    pub p: TrainParams,
}

impl FastTrainModel {
    pub fn new(cfg: TrainModelConfig, device: Device, seed: i64) -> Result<Self> {
        let p = TrainParams::init(&cfg, device, seed)?;
        Ok(Self { cfg, p })
    }

    /// Create a new FastTrainModel by loading weights from a safetensors file.
    pub fn load_from_safetensors<P: AsRef<std::path::Path>>(
        path: P,
        cfg: TrainModelConfig,
        device: Device,
    ) -> Result<Self> {
        let p = TrainParams::load_from_safetensors(path, &cfg, device)?;
        Ok(Self { cfg, p })
    }

    /// Forward pass for entire sequence.
    ///
    /// tokens: (B, T) int64
    /// Returns logits: (B, T, V)
    pub fn forward_sequence(&self, tokens: &Tensor, state: &mut FastTrainState) -> Result<Tensor> {
        let sizes = tokens.size();
        let b = sizes[0];
        let t = sizes[1];
        let c = self.cfg.hidden_size;

        // Embedding lookup: (B, T) -> (B, T, C)
        let mut x = self
            .p
            .embeddings
            .index_select(0, &tokens.view([-1]))
            .view([b, t, c]);

        for (layer_idx, layer) in self.p.layers.iter().enumerate() {
            // Pre-norm (layer 0 only)
            if layer_idx == 0 {
                if let (Some(w), Some(b_)) = (&layer.pre_norm_w, &layer.pre_norm_b) {
                    x = x.layer_norm(&[c], Some(w), Some(b_), self.cfg.layer_norm_eps, false);
                }
            }

            // Attention block
            let x_norm = x.layer_norm(
                &[c],
                Some(&layer.attn_norm_w),
                Some(&layer.attn_norm_b),
                self.cfg.layer_norm_eps,
                false,
            );

            let att_out = self.attention_seq(layer_idx, layer, &x_norm, b, t, state)?;
            x = &x + att_out;

            // FFN block
            let x_norm = x.layer_norm(
                &[c],
                Some(&layer.ffn_norm_w),
                Some(&layer.ffn_norm_b),
                self.cfg.layer_norm_eps,
                false,
            );

            let ffn_out = self.ffn_seq(layer, &x_norm, b, t)?;
            x = x + ffn_out;
        }

        // Output norm and LM head
        let x_norm = x.layer_norm(
            &[c],
            Some(&self.p.ln_out_w),
            Some(&self.p.ln_out_b),
            self.cfg.layer_norm_eps,
            false,
        );

        // logits: (B, T, C) @ (C, V) -> (B, T, V)
        Ok(x_norm.matmul(&self.p.lm_head.transpose(0, 1)))
    }

    /// Sequence-parallel token shift.
    /// x: (B, T, C), mix: (C,)
    /// Returns shifted: (B, T, C) where shifted[:,t,:] = x[:,t,:] + mix * (x[:,t-1,:] - x[:,t,:])
    /// For t=0, prev is zeros.
    #[inline]
    fn token_shift_seq(x: &Tensor, mix: &Tensor, b: i64, t: i64, c: i64) -> Tensor {
        // prev[:,t,:] = x[:,t-1,:] for t>0, zeros for t=0
        let zeros = Tensor::zeros([b, 1, c], (x.kind(), x.device()));
        let x_prev = Tensor::cat(&[zeros, x.i((.., ..t - 1, ..))], 1); // (B, T, C)
                                                                       // Use contiguous to ensure view operations work
        let x_prev = x_prev.contiguous();
        // shifted = x + mix * (prev - x)
        x + mix.view([1, 1, c]) * (&x_prev - x)
    }

    /// Attention for full sequence.
    fn attention_seq(
        &self,
        layer_idx: usize,
        layer: &LayerParams,
        x_norm: &Tensor,
        b: i64,
        t: i64,
        state: &mut FastTrainState,
    ) -> Result<Tensor> {
        let c = self.cfg.hidden_size;
        let h = self.cfg.num_heads;
        let n = self.cfg.head_dim;

        // Token shift for all mix vectors - parallel across time
        let xr = Self::token_shift_seq(x_norm, &layer.attn_x_r, b, t, c);
        let xw = Self::token_shift_seq(x_norm, &layer.attn_x_w, b, t, c);
        let xk = Self::token_shift_seq(x_norm, &layer.attn_x_k, b, t, c);
        let xv = Self::token_shift_seq(x_norm, &layer.attn_x_v, b, t, c);
        let xa = Self::token_shift_seq(x_norm, &layer.attn_x_a, b, t, c);
        let xg = Self::token_shift_seq(x_norm, &layer.attn_x_g, b, t, c);

        // All linear projections batched: (B,T,C) @ (C,C) -> (B,T,C)
        let r = xr.matmul(&layer.attn_r_proj.transpose(0, 1)); // (B,T,C)
        let k_raw = xk.matmul(&layer.attn_k_proj.transpose(0, 1));
        let mut v = xv.matmul(&layer.attn_v_proj.transpose(0, 1));

        // w decay: w = exp(-sigmoid(w2 @ tanh(w1 @ xw) + w0) / sqrt(e))
        let tmp_w = xw.matmul(&layer.attn_w1.transpose(0, 1)).tanh(); // (B,T,D_w)
        let w_pre = tmp_w.matmul(&layer.attn_w2.transpose(0, 1)) + &layer.attn_w0; // (B,T,C)
        let inv_sqrt_e = 1.0f64 / std::f64::consts::E.sqrt();
        let w = (-w_pre.sigmoid() * inv_sqrt_e).exp(); // (B,T,C)

        // a = sigmoid(a2 @ (a1 @ xa) + a0)
        let tmp_a = xa.matmul(&layer.attn_a1.transpose(0, 1)); // (B,T,D_a)
        let a = (tmp_a.matmul(&layer.attn_a2.transpose(0, 1)) + &layer.attn_a0).sigmoid(); // (B,T,C)

        // g = sigmoid(g1 @ xg) @ g2
        let tmp_g = xg.matmul(&layer.attn_g1.transpose(0, 1)).sigmoid(); // (B,T,D_g)
        let g = tmp_g.matmul(&layer.attn_g2.transpose(0, 1)); // (B,T,C)

        // Value residual (v_first)
        if layer_idx == 0 {
            // Store v at t=0 for all batch items
            state.v_first = Some(v.i((.., 0i64, ..)).shallow_clone()); // (B,C)
        } else if let Some(ref v_first) = state.v_first {
            if let (Some(v1), Some(v2), Some(v0)) = (&layer.attn_v1, &layer.attn_v2, &layer.attn_v0)
            {
                let tmp = xv.matmul(&v1.transpose(0, 1)); // (B,T,D_v)
                let nu = (tmp.matmul(&v2.transpose(0, 1)) + v0).sigmoid(); // (B,T,C)
                                                                           // v = v + (v_first - v) * nu, broadcast v_first (B,C) -> (B,1,C)
                v = &v + (v_first.unsqueeze(1) - &v) * nu;
            }
        }

        // kk = normalize(k * k_k) per head
        let kk_unnorm = (&k_raw * &layer.attn_k_k).contiguous().view([b, t, h, n]); // (B,T,H,N)
        let kk_norm = {
            let sq_sum = kk_unnorm
                .square()
                .sum_dim_intlist(&[-1i64][..], true, Kind::Float);
            &kk_unnorm / (sq_sum + 1e-12).sqrt()
        }; // (B,T,H,N)

        // k = k * (1 + (a - 1) * k_a)
        let scale = (&a - 1.0) * &layer.attn_k_a + 1.0; // (B,T,C)
        let k = (&k_raw * scale).contiguous().view([b, t, h, n]); // (B,T,H,N)

        // Reshape others for WKV - need contiguous for tensors that result from operations
        let r_h = r.contiguous().view([b, t, h, n]); // (B,T,H,N)
        let v_h = v.contiguous().view([b, t, h, n]);
        let w_h = w.contiguous().view([b, t, h, n]);
        let a_h = a.contiguous().view([b, t, h, n]);
        let g_h = g.contiguous().view([b, t, h, n]);

        // WKV recurrence - use hybrid CPU/GPU for long sequences (faster than GPU sequential)
        // For short sequences, use GPU checkpointed approach
        let (y, new_state) = if t > 128 {
            // Hybrid: move to CPU for sequential WKV, then back to GPU
            wkv_fused::wkv_hybrid(
                &r_h,
                &k,
                &v_h,
                &w_h,
                &a_h,
                &kk_norm,
                &state.wkv_state[layer_idx],
            )
        } else {
            // Checkpointed GPU approach for short sequences
            let wkv_chunk_size = t.min(32);
            wkv_fused::wkv_checkpointed(
                &r_h,
                &k,
                &v_h,
                &w_h,
                &a_h,
                &kk_norm,
                &state.wkv_state[layer_idx],
                wkv_chunk_size,
            )
        };

        // Update state
        state.wkv_state[layer_idx] = new_state;

        // Group norm per head
        let y_gn = {
            let mean = y.mean_dim(&[-1i64][..], true, Kind::Float);
            let yc = &y - &mean;
            let var = yc.square().mean_dim(&[-1i64][..], true, Kind::Float);
            let normed = yc / (var + self.cfg.group_norm_eps).sqrt();
            let gn_w = layer.attn_gn_w.view([1, 1, h, n]);
            let gn_b = layer.attn_gn_b.view([1, 1, h, n]);
            normed * gn_w + gn_b
        }; // (B,T,H,N)

        // Head-qk term: y += (r * k * r_k).sum(-1,keepdim) * v
        let r_k = layer.attn_r_k.view([1, 1, h, n]);
        let alpha = (&r_h * &k * r_k).sum_dim_intlist(&[-1i64][..], true, Kind::Float); // (B,T,H,1)
        let y2 = (&y_gn + alpha * &v_h) * &g_h; // (B,T,H,N)

        // Output projection: (B,T,C) @ (C,C) -> (B,T,C)
        let y_flat = y2.contiguous().view([b, t, c]);
        Ok(y_flat.matmul(&layer.attn_o_proj.transpose(0, 1)))
    }

    /// FFN for full sequence.
    fn ffn_seq(&self, layer: &LayerParams, x_norm: &Tensor, b: i64, t: i64) -> Result<Tensor> {
        let c = self.cfg.hidden_size;

        // Token shift
        let xk = Self::token_shift_seq(x_norm, &layer.ffn_x_k, b, t, c);

        // k = relu(xk @ key_w.T)^2
        let k = xk.matmul(&layer.ffn_key_w.transpose(0, 1)).relu();
        let k2 = &k * &k;

        // out = k2 @ value_w.T
        Ok(k2.matmul(&layer.ffn_value_w.transpose(0, 1)))
    }

    /// Get all trainable parameters (delegates to TrainParams)
    pub fn parameters(&self) -> Vec<Tensor> {
        self.p.parameters()
    }
}

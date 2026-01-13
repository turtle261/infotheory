use crate::rwkv7::training::model::{TrainModelConfig, TrainParams};
use anyhow::{Context, Result};
use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use tch::{Device, Kind, Tensor};

fn tensor_to_vec_f32(t: &Tensor) -> Result<Vec<f32>> {
    let t = t.to_device(Device::Cpu).to_kind(Kind::Float).contiguous();
    let n = t.numel();
    let mut out = vec![0f32; n as usize];
    t.f_copy_data(&mut out, n)
        .context("Failed to copy tensor data")?;
    Ok(out)
}

fn tensor_shape_usize(t: &Tensor) -> Vec<usize> {
    t.size().iter().map(|&d| d as usize).collect()
}

pub fn export_safetensors<P: AsRef<Path>>(
    path: P,
    cfg: &TrainModelConfig,
    p: &TrainParams,
) -> Result<()> {
    cfg.validate()?;

    let mut tensors: BTreeMap<String, (Vec<usize>, Vec<f32>)> = BTreeMap::new();

    tensors.insert(
        "model.embeddings.weight".to_string(),
        (
            tensor_shape_usize(&p.embeddings),
            tensor_to_vec_f32(&p.embeddings)?,
        ),
    );
    tensors.insert(
        "model.norm.weight".to_string(),
        (
            tensor_shape_usize(&p.ln_out_w),
            tensor_to_vec_f32(&p.ln_out_w)?,
        ),
    );
    tensors.insert(
        "model.norm.bias".to_string(),
        (
            tensor_shape_usize(&p.ln_out_b),
            tensor_to_vec_f32(&p.ln_out_b)?,
        ),
    );
    tensors.insert(
        "lm_head.weight".to_string(),
        (
            tensor_shape_usize(&p.lm_head),
            tensor_to_vec_f32(&p.lm_head)?,
        ),
    );

    for (i, l) in p.layers.iter().enumerate() {
        let prefix = format!("model.layers.{}", i);

        if i == 0 {
            if let (Some(w), Some(b)) = (&l.pre_norm_w, &l.pre_norm_b) {
                tensors.insert(
                    format!("{}.pre_norm.weight", prefix),
                    (tensor_shape_usize(w), tensor_to_vec_f32(w)?),
                );
                tensors.insert(
                    format!("{}.pre_norm.bias", prefix),
                    (tensor_shape_usize(b), tensor_to_vec_f32(b)?),
                );
            }
        }

        tensors.insert(
            format!("{}.attn_norm.weight", prefix),
            (
                tensor_shape_usize(&l.attn_norm_w),
                tensor_to_vec_f32(&l.attn_norm_w)?,
            ),
        );
        tensors.insert(
            format!("{}.attn_norm.bias", prefix),
            (
                tensor_shape_usize(&l.attn_norm_b),
                tensor_to_vec_f32(&l.attn_norm_b)?,
            ),
        );
        tensors.insert(
            format!("{}.ffn_norm.weight", prefix),
            (
                tensor_shape_usize(&l.ffn_norm_w),
                tensor_to_vec_f32(&l.ffn_norm_w)?,
            ),
        );
        tensors.insert(
            format!("{}.ffn_norm.bias", prefix),
            (
                tensor_shape_usize(&l.ffn_norm_b),
                tensor_to_vec_f32(&l.ffn_norm_b)?,
            ),
        );

        // Attention
        tensors.insert(
            format!("{}.attn.x_r", prefix),
            (
                tensor_shape_usize(&l.attn_x_r),
                tensor_to_vec_f32(&l.attn_x_r)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.x_w", prefix),
            (
                tensor_shape_usize(&l.attn_x_w),
                tensor_to_vec_f32(&l.attn_x_w)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.x_k", prefix),
            (
                tensor_shape_usize(&l.attn_x_k),
                tensor_to_vec_f32(&l.attn_x_k)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.x_v", prefix),
            (
                tensor_shape_usize(&l.attn_x_v),
                tensor_to_vec_f32(&l.attn_x_v)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.x_a", prefix),
            (
                tensor_shape_usize(&l.attn_x_a),
                tensor_to_vec_f32(&l.attn_x_a)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.x_g", prefix),
            (
                tensor_shape_usize(&l.attn_x_g),
                tensor_to_vec_f32(&l.attn_x_g)?,
            ),
        );

        tensors.insert(
            format!("{}.attn.r_proj.weight", prefix),
            (
                tensor_shape_usize(&l.attn_r_proj),
                tensor_to_vec_f32(&l.attn_r_proj)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.k_proj.weight", prefix),
            (
                tensor_shape_usize(&l.attn_k_proj),
                tensor_to_vec_f32(&l.attn_k_proj)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.v_proj.weight", prefix),
            (
                tensor_shape_usize(&l.attn_v_proj),
                tensor_to_vec_f32(&l.attn_v_proj)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.o_proj.weight", prefix),
            (
                tensor_shape_usize(&l.attn_o_proj),
                tensor_to_vec_f32(&l.attn_o_proj)?,
            ),
        );

        tensors.insert(
            format!("{}.attn.w_lora.lora.0.weight", prefix),
            (
                tensor_shape_usize(&l.attn_w1),
                tensor_to_vec_f32(&l.attn_w1)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.w_lora.lora.2.weight", prefix),
            (
                tensor_shape_usize(&l.attn_w2),
                tensor_to_vec_f32(&l.attn_w2)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.w_lora.lora.2.bias", prefix),
            (
                tensor_shape_usize(&l.attn_w0),
                tensor_to_vec_f32(&l.attn_w0)?,
            ),
        );

        tensors.insert(
            format!("{}.attn.a_lora.lora.0.weight", prefix),
            (
                tensor_shape_usize(&l.attn_a1),
                tensor_to_vec_f32(&l.attn_a1)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.a_lora.lora.2.weight", prefix),
            (
                tensor_shape_usize(&l.attn_a2),
                tensor_to_vec_f32(&l.attn_a2)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.a_lora.lora.2.bias", prefix),
            (
                tensor_shape_usize(&l.attn_a0),
                tensor_to_vec_f32(&l.attn_a0)?,
            ),
        );

        if let (Some(v1), Some(v2), Some(v0)) = (&l.attn_v1, &l.attn_v2, &l.attn_v0) {
            tensors.insert(
                format!("{}.attn.v_lora.lora.0.weight", prefix),
                (tensor_shape_usize(v1), tensor_to_vec_f32(v1)?),
            );
            tensors.insert(
                format!("{}.attn.v_lora.lora.2.weight", prefix),
                (tensor_shape_usize(v2), tensor_to_vec_f32(v2)?),
            );
            tensors.insert(
                format!("{}.attn.v_lora.lora.2.bias", prefix),
                (tensor_shape_usize(v0), tensor_to_vec_f32(v0)?),
            );
        }

        tensors.insert(
            format!("{}.attn.g_lora.lora.0.weight", prefix),
            (
                tensor_shape_usize(&l.attn_g1),
                tensor_to_vec_f32(&l.attn_g1)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.g_lora.lora.2.weight", prefix),
            (
                tensor_shape_usize(&l.attn_g2),
                tensor_to_vec_f32(&l.attn_g2)?,
            ),
        );

        tensors.insert(
            format!("{}.attn.k_k", prefix),
            (
                tensor_shape_usize(&l.attn_k_k),
                tensor_to_vec_f32(&l.attn_k_k)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.k_a", prefix),
            (
                tensor_shape_usize(&l.attn_k_a),
                tensor_to_vec_f32(&l.attn_k_a)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.r_k", prefix),
            (
                tensor_shape_usize(&l.attn_r_k),
                tensor_to_vec_f32(&l.attn_r_k)?,
            ),
        );

        tensors.insert(
            format!("{}.attn.g_norm.weight", prefix),
            (
                tensor_shape_usize(&l.attn_gn_w),
                tensor_to_vec_f32(&l.attn_gn_w)?,
            ),
        );
        tensors.insert(
            format!("{}.attn.g_norm.bias", prefix),
            (
                tensor_shape_usize(&l.attn_gn_b),
                tensor_to_vec_f32(&l.attn_gn_b)?,
            ),
        );

        // FFN
        tensors.insert(
            format!("{}.ffn.x_k", prefix),
            (
                tensor_shape_usize(&l.ffn_x_k),
                tensor_to_vec_f32(&l.ffn_x_k)?,
            ),
        );
        tensors.insert(
            format!("{}.ffn.key.weight", prefix),
            (
                tensor_shape_usize(&l.ffn_key_w),
                tensor_to_vec_f32(&l.ffn_key_w)?,
            ),
        );
        tensors.insert(
            format!("{}.ffn.value.weight", prefix),
            (
                tensor_shape_usize(&l.ffn_value_w),
                tensor_to_vec_f32(&l.ffn_value_w)?,
            ),
        );
    }

    // Write safetensors format directly (compatible with our native reader and HF tooling).
    // Format:
    // - u64 little-endian header length
    // - JSON header bytes
    // - raw tensor bytes (little-endian)

    // Compute contiguous byte offsets.
    let mut cursor: usize = 0;
    let mut meta_entries: Vec<String> = Vec::with_capacity(tensors.len());
    let mut data_blobs: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());

    for (name, (shape, data_f32)) in tensors.into_iter() {
        let byte_len = data_f32.len() * 4;
        let start = cursor;
        let end = cursor + byte_len;
        cursor = end;

        // Build bytes for tensor
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

    // Deterministic header with empty metadata.
    let header_json = format!("{{\"__metadata__\":{{}},{} }}", meta_entries.join(","));
    let header_bytes = header_json.as_bytes();
    let header_len = header_bytes.len() as u64;

    let mut out = Vec::with_capacity(8 + header_bytes.len() + cursor);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header_bytes);
    for blob in data_blobs {
        out.extend_from_slice(&blob);
    }

    fs::write(path.as_ref(), out).with_context(|| format!("write {}", path.as_ref().display()))?;
    Ok(())
}

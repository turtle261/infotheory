// RWKV7 WKV CUDA kernel for training with chunked checkpointing
// Based on official RWKV-v7 training implementation
//
// Forward: Saves state every CHUNK_LEN steps for backward pass
// Backward: Uses checkpoints to reconstruct state and compute gradients
//
// Input shapes (all fp32, contiguous):
//   r, w, k, v, a, b: (B, T, H, N) -> flattened to (B*T*H*N)
// Output shapes:
//   y: (B, T, H, N)
//   s: (B, H, T/CHUNK_LEN, N, N) - checkpointed states
//   sa: (B, T, H, N) - saved sa values for backward
//
// Launch config: grid(H, B), block(N) where N=head_dim=64

#include <cuda_runtime.h>
#include <math.h>

#ifndef _N_
#define _N_ 64
#endif

#ifndef _CHUNK_LEN_
#define _CHUNK_LEN_ 32
#endif

extern "C" {

// Forward kernel with checkpointing for training
__global__ void wkv7_forward_train(
    const int T,              // sequence length
    const float* __restrict__ r,      // (B, T, H, N)
    const float* __restrict__ w,      // (B, T, H, N)
    const float* __restrict__ k,      // (B, T, H, N)
    const float* __restrict__ v,      // (B, T, H, N)
    const float* __restrict__ a,      // (B, T, H, N)
    const float* __restrict__ b,      // (B, T, H, N)
    float* __restrict__ y,            // (B, T, H, N)
    float* __restrict__ s_out,        // (B, H, num_chunks, N, N) checkpoints
    float* __restrict__ sa_out        // (B, T, H, N) saved sa values
) {
    const int bb = blockIdx.y;  // batch index
    const int hh = blockIdx.x;  // head index
    const int i = threadIdx.x;  // column index 0..N-1
    const int H = gridDim.x;
    
    if (i >= _N_) return;
    
    // Each thread maintains one column of state in registers
    float state[_N_] = {0};
    
    // Shared memory for input vectors
    __shared__ float s_r[_N_], s_k[_N_], s_w[_N_], s_a[_N_], s_b[_N_];
    
    const int num_chunks = (T + _CHUNK_LEN_ - 1) / _CHUNK_LEN_;
    
    for (int t = 0; t < T; t++) {
        // Input index: (bb, t, hh, i) in (B, T, H, N) layout
        const int ind = bb * T * H * _N_ + t * H * _N_ + hh * _N_ + i;
        
        __syncthreads();
        
        // Load input vectors into shared memory
        s_r[i] = r[ind];
        s_w[i] = expf(-expf(w[ind]));  // w = exp(-exp(w_raw))
        s_k[i] = k[ind];
        s_a[i] = a[ind];
        s_b[i] = b[ind];
        
        __syncthreads();
        
        // Compute sa = sum_j a[j] * state[j] (for this column i)
        // But we need the full sa vector, so we need to reduce
        // Each thread computes partial contribution then reduces
        
        // Actually, looking at official code, sa[i] = sum_j a[j] * state[j][i]
        // where state[j][i] is row j, column i
        // Since we store state column-wise, state[j] in our code is state[j][i]
        float sa = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            sa += s_a[j] * state[j];
        }
        
        // Save sa for backward pass
        sa_out[ind] = sa;
        
        // Get v value for this position
        float vv = v[ind];
        
        // Update state and compute y
        // state[j] = state[j] * w[j] + k[j] * v + sa * b[j]
        // y = sum_j state[j] * r[j]
        float y_val = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            state[j] = state[j] * s_w[j] + s_k[j] * vv + sa * s_b[j];
            y_val += state[j] * s_r[j];
        }
        
        // Write output
        y[ind] = y_val;
        
        // Checkpoint state every CHUNK_LEN steps
        if ((t + 1) % _CHUNK_LEN_ == 0) {
            int chunk_idx = t / _CHUNK_LEN_;
            // s_out shape: (B, H, num_chunks, N, N)
            // Index: bb * (H * num_chunks * N * N) + hh * (num_chunks * N * N) + chunk_idx * (N * N) + row * N + col
            int base = bb * H * num_chunks * _N_ * _N_ + hh * num_chunks * _N_ * _N_ + chunk_idx * _N_ * _N_;
            #pragma unroll
            for (int j = 0; j < _N_; j++) {
                s_out[base + j * _N_ + i] = state[j];
            }
        }
    }
    
    // Save final state if not already saved (when T is not multiple of CHUNK_LEN)
    if (T % _CHUNK_LEN_ != 0) {
        int chunk_idx = num_chunks - 1;
        int base = bb * H * num_chunks * _N_ * _N_ + hh * num_chunks * _N_ * _N_ + chunk_idx * _N_ * _N_;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            s_out[base + j * _N_ + i] = state[j];
        }
    }
}

// Backward kernel using saved checkpoints
__global__ void wkv7_backward_train(
    const int T,
    const float* __restrict__ r,
    const float* __restrict__ w,
    const float* __restrict__ k,
    const float* __restrict__ v,
    const float* __restrict__ a,
    const float* __restrict__ b,
    const float* __restrict__ dy,     // gradient from output
    const float* __restrict__ s_in,   // saved state checkpoints
    const float* __restrict__ sa_in,  // saved sa values
    float* __restrict__ dr,
    float* __restrict__ dw,
    float* __restrict__ dk,
    float* __restrict__ dv,
    float* __restrict__ da,
    float* __restrict__ db
) {
    const int bb = blockIdx.y;
    const int hh = blockIdx.x;
    const int i = threadIdx.x;
    const int H = gridDim.x;
    
    if (i >= _N_) return;
    
    const int num_chunks = (T + _CHUNK_LEN_ - 1) / _CHUNK_LEN_;
    
    // Gradient accumulators for state
    float dstate[_N_] = {0};
    float state[_N_] = {0};  // Reconstructed state
    
    __shared__ float s_r[_N_], s_k[_N_], s_w[_N_], s_a[_N_], s_b[_N_], s_dy[_N_], s_v[_N_], s_sa[_N_];
    __shared__ float s_dSb[_N_];  // For da computation reduction
    
    // Process backward through time
    for (int t = T - 1; t >= 0; t--) {
        const int ind = bb * T * H * _N_ + t * H * _N_ + hh * _N_ + i;
        
        // Load checkpoint if at chunk boundary
        if ((t + 1) % _CHUNK_LEN_ == 0 || t == T - 1) {
            int chunk_idx = t / _CHUNK_LEN_;
            int base = bb * H * num_chunks * _N_ * _N_ + hh * num_chunks * _N_ * _N_ + chunk_idx * _N_ * _N_;
            #pragma unroll
            for (int j = 0; j < _N_; j++) {
                state[j] = s_in[base + j * _N_ + i];
            }
        }
        
        __syncthreads();
        
        // Load inputs
        s_r[i] = r[ind];
        float wi_fac = -expf(w[ind]);
        s_w[i] = expf(wi_fac);
        s_k[i] = k[ind];
        s_a[i] = a[ind];
        s_b[i] = b[ind];
        s_v[i] = v[ind];
        s_dy[i] = dy[ind];
        s_sa[i] = sa_in[ind];
        
        __syncthreads();
        
        // Compute dr = state @ dy (for this column)
        float dr_val = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            dr_val += state[j] * s_dy[j];
        }
        dr[ind] = dr_val;
        
        // Invert state update: state_prev = (state - k*v - sa*b) / w
        float iwi = 1.0f / s_w[i];
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            state[j] = (state[j] - s_k[i] * s_v[j] - s_b[i] * s_sa[j]) * iwi;
            dstate[j] += s_dy[i] * s_r[j];
        }
        
        // Compute gradients
        float dw_val = 0, dk_val = 0, dv_val = 0, db_val = 0, dSb = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            dw_val += dstate[j] * state[j];
            dk_val += dstate[j] * s_v[j];
            dv_val += dstate[j] * s_k[j];
            dSb += dstate[j] * s_b[j];
            db_val += dstate[j] * s_sa[j];
        }
        
        dw[ind] = dw_val * s_w[i] * wi_fac;
        dk[ind] = dk_val;
        dv[ind] = dv_val;
        db[ind] = db_val;
        
        // Compute da using reduction
        __syncthreads();
        s_dSb[i] = dSb;
        __syncthreads();
        
        float da_val = 0;
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            da_val += state[j] * s_dSb[j];
        }
        da[ind] = da_val;
        
        // Update dstate for next timestep
        #pragma unroll
        for (int j = 0; j < _N_; j++) {
            dstate[j] = dstate[j] * s_w[j] + dSb * s_a[j];
        }
    }
}

// Host wrapper functions
void wkv7_cuda_forward_train(
    int B, int T, int H, int N,
    const float* r, const float* w, const float* k, const float* v,
    const float* a, const float* b,
    float* y, float* s_out, float* sa_out
) {
    dim3 grid(H, B);
    dim3 block(N);
    wkv7_forward_train<<<grid, block>>>(T, r, w, k, v, a, b, y, s_out, sa_out);
}

void wkv7_cuda_backward_train(
    int B, int T, int H, int N,
    const float* r, const float* w, const float* k, const float* v,
    const float* a, const float* b, const float* dy,
    const float* s_in, const float* sa_in,
    float* dr, float* dw, float* dk, float* dv, float* da, float* db
) {
    dim3 grid(H, B);
    dim3 block(N);
    wkv7_backward_train<<<grid, block>>>(T, r, w, k, v, a, b, dy, s_in, sa_in, dr, dw, dk, dv, da, db);
}

} // extern "C"

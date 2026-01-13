// RWKV7 WKV CUDA kernel for fast training
// Processes entire sequence in one kernel launch with state in registers
//
// This kernel implements the RWKV7 linear attention:
//   S = S * w  (decay)
//   u = S @ kk
//   S = S - outer(u, kk*a) + outer(v, k)
//   y = S @ r
//
// Input shapes (all fp32, contiguous):
//   r, w, k, v, a, kk: (B*H, T, N) - batch*heads merged
//   state: (B*H, N, N) - initial state
// Output shapes:
//   y: (B*H, T, N)
//   state_out: (B*H, N, N)
//
// Launch config: grid(B*H), block(N) where N=head_dim=64
// Each thread handles one column of the NxN state matrix

#include <cuda_runtime.h>

// N must be defined at compile time (e.g. -D_N_=64)
#ifndef _N_
#define _N_ 64
#endif

extern "C" {

// Forward kernel for training (fp32)
// Each block processes one (batch, head) pair
// Each thread handles one column of the state matrix
__global__ void wkv7_forward_kernel(
    const int T,
    const float* __restrict__ r,      // (BH, T, N)
    const float* __restrict__ w,      // (BH, T, N)
    const float* __restrict__ k,      // (BH, T, N)
    const float* __restrict__ v,      // (BH, T, N)
    const float* __restrict__ a,      // (BH, T, N)
    const float* __restrict__ kk,     // (BH, T, N)
    const float* __restrict__ state_in,  // (BH, N, N)
    float* __restrict__ y,            // (BH, T, N)
    float* __restrict__ state_out     // (BH, N, N)
) {
    const int bh = blockIdx.x;        // batch*head index
    const int col = threadIdx.x;      // column index (0..N-1)
    
    if (col >= _N_) return;
    
    // Each thread maintains one column of the NxN state matrix in registers
    // state_col[row] = state[row][col]
    float state_col[_N_];
    
    // Load initial state column
    const int state_base = bh * _N_ * _N_;
    #pragma unroll
    for (int row = 0; row < _N_; row++) {
        state_col[row] = state_in[state_base + row * _N_ + col];
    }
    
    // Shared memory for input vectors at current timestep
    __shared__ float s_r[_N_], s_k[_N_], s_w[_N_], s_v[_N_], s_a[_N_], s_kk[_N_];
    
    const int TN = T * _N_;  // stride for time dimension
    
    for (int t = 0; t < T; t++) {
        // Input index for this batch-head and timestep
        const int in_idx = bh * TN + t * _N_;
        
        __syncthreads();
        
        // Collaboratively load input vectors into shared memory
        // Each thread loads one element
        s_r[col] = r[in_idx + col];
        s_w[col] = w[in_idx + col];
        s_k[col] = k[in_idx + col];
        s_v[col] = v[in_idx + col];
        s_a[col] = a[in_idx + col];
        s_kk[col] = kk[in_idx + col];
        
        __syncthreads();
        
        // Step 1: S = S * w (column-wise decay)
        // state_col *= w[col]
        const float w_col = s_w[col];
        #pragma unroll
        for (int row = 0; row < _N_; row++) {
            state_col[row] *= w_col;
        }
        
        // Step 2: u = S @ kk (for this column, we compute u[row] = sum_j state[row][j] * kk[j])
        // But each thread only has state_col (one column)
        // Need reduction across threads
        // u[row] = sum over all columns j: state[row][j] * kk[j]
        // Each thread computes partial: state_col[row] * kk[col]
        // Then reduce across threads
        
        // For simplicity, compute u fully within each thread's register (requires shared mem)
        // Alternative: use warp shuffle for reduction
        
        // Compute u[row] for each row
        // u[row] = sum_j S[row][j] * kk[j]
        // We need to communicate state values across threads
        
        // Use shared memory for state (one row at a time)
        __shared__ float s_state_row[_N_];
        __shared__ float s_u[_N_];
        
        // Compute u by iterating over rows
        #pragma unroll 4
        for (int row = 0; row < _N_; row++) {
            // Each thread writes its state value for this row
            s_state_row[col] = state_col[row];
            __syncthreads();
            
            // Thread 0 computes u[row] = sum_j s_state_row[j] * s_kk[j]
            if (col == 0) {
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < _N_; j++) {
                    sum += s_state_row[j] * s_kk[j];
                }
                s_u[row] = sum;
            }
            __syncthreads();
        }
        
        // Now s_u contains u vector
        
        // Step 3: S = S - outer(u, kk*a) + outer(v, k)
        // state[row][col] = state[row][col] - u[row] * kk[col] * a[col] + v[row] * k[col]
        const float kka_col = s_kk[col] * s_a[col];
        const float k_col = s_k[col];
        #pragma unroll
        for (int row = 0; row < _N_; row++) {
            state_col[row] = state_col[row] - s_u[row] * kka_col + s_v[row] * k_col;
        }
        
        // Step 4: y = S @ r
        // y[row] = sum_j S[row][j] * r[j]
        // Similar to u computation, need reduction
        
        #pragma unroll 4
        for (int row = 0; row < _N_; row++) {
            s_state_row[col] = state_col[row];
            __syncthreads();
            
            if (col == 0) {
                float sum = 0.0f;
                #pragma unroll
                for (int j = 0; j < _N_; j++) {
                    sum += s_state_row[j] * s_r[j];
                }
                // Write y[bh][t][row]
                y[bh * TN + t * _N_ + row] = sum;
            }
            __syncthreads();
        }
    }
    
    // Save final state
    #pragma unroll
    for (int row = 0; row < _N_; row++) {
        state_out[state_base + row * _N_ + col] = state_col[row];
    }
}

// Host wrapper function
void wkv7_cuda_forward(
    int BH,              // batch * heads
    int T,               // sequence length
    int N,               // head dimension (should be _N_)
    const float* r,
    const float* w,
    const float* k,
    const float* v,
    const float* a,
    const float* kk,
    const float* state_in,
    float* y,
    float* state_out
) {
    // Launch kernel
    dim3 grid(BH);
    dim3 block(N);
    wkv7_forward_kernel<<<grid, block>>>(T, r, w, k, v, a, kk, state_in, y, state_out);
}

} // extern "C"

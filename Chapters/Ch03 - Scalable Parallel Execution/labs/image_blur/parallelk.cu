#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <vector>

// Constants for CUDA kernels
#define BLOCK_SIZE 16
#define CHECK_CUDA_ERROR(err)                                \
    if (err != cudaSuccess)                                  \
    {                                                        \
        printf("CUDA error: %s\n", cudaGetErrorString(err)); \
        exit(-1);                                            \
    }

// CUDA kernel for matrix multiplication with serial-K reduction
__global__ void serialKReductionKernel(
    const float *A_chunk, // Current chunk of matrix A
    const float *B_chunk, // Current chunk of matrix B
    float *C,             // Result matrix (accumulates results)
    const int M,          // Rows of A
    const int K_prime,    // Chunk size
    const int N,          // Columns of B
    const int K_offset    // Current offset in K dimension
)
{
    // Calculate thread indices
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory for tiling
    __shared__ float As[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE][BLOCK_SIZE];

    // Accumulator for this thread
    float sum = 0.0f;

    // Handle tiles in K_prime dimension
    for (int t = 0; t < (K_prime + BLOCK_SIZE - 1) / BLOCK_SIZE; ++t)
    {
        // Load tile into shared memory
        if (row < M && (t * BLOCK_SIZE + threadIdx.x) < K_prime)
            As[threadIdx.y][threadIdx.x] = A_chunk[row * K_prime + t * BLOCK_SIZE + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0f;

        if ((t * BLOCK_SIZE + threadIdx.y) < K_prime && col < N)
            Bs[threadIdx.y][threadIdx.x] = B_chunk[(t * BLOCK_SIZE + threadIdx.y) * N + col];
        else
            Bs[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Compute partial dot product for this tile
        for (int k = 0; k < BLOCK_SIZE; ++k)
        {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }

        __syncthreads();
    }

    // Accumulate result in global memory
    if (row < M && col < N)
    {
        atomicAdd(&C[row * N + col], sum);
    }
}

// CUDA kernel for parallel-K reduction
__global__ void parallelKReductionKernel(
    const float *A,   // Full matrix A
    const float *B,   // Full matrix B
    float *C_partial, // Partial results for this chunk
    const int M,
    const int K,
    const int N,
    const int K_prime,
    const int chunk_idx)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N)
    {
        float sum = 0.0f;
        int k_start = chunk_idx * K_prime;
        int k_end = min(k_start + K_prime, K);

        // Compute partial result for this chunk
        for (int k = k_start; k < k_end; ++k)
        {
            sum += A[row * K + k] * B[k * N + col];
        }

        // Store in partial results
        C_partial[row * N + col] = sum;
    }
}

class CudaKReduction
{
public:
    // Serial-K reduction implementation
    static void serialKReduction(
        const std::vector<float> &h_A,
        const std::vector<float> &h_B,
        std::vector<float> &h_C,
        int M, int K, int N,
        int K_prime)
    {
        // Allocate device memory for result
        float *d_C;
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemset(d_C, 0, M * N * sizeof(float));

        // Allocate device memory for chunks
        float *d_A_chunk, *d_B_chunk;
        cudaMalloc(&d_A_chunk, M * K_prime * sizeof(float));
        cudaMalloc(&d_B_chunk, K_prime * N * sizeof(float));

        // Set up grid and block dimensions
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Process each chunk sequentially
        for (int k = 0; k < K; k += K_prime)
        {
            int current_K_prime = min(K_prime, K - k);

            // Copy current chunks to device
            cudaMemcpy(d_A_chunk, &h_A[k], M * current_K_prime * sizeof(float),
                       cudaMemcpyHostToDevice);
            cudaMemcpy(d_B_chunk, &h_B[k * N], current_K_prime * N * sizeof(float),
                       cudaMemcpyHostToDevice);

            // Launch kernel for this chunk
            serialKReductionKernel<<<gridDim, blockDim>>>(
                d_A_chunk, d_B_chunk, d_C,
                M, current_K_prime, N, k);
        }

        // Copy result back to host
        cudaMemcpy(h_C.data(), d_C, M * N * sizeof(float), cudaMemcpyDeviceToHost);

        // Cleanup
        cudaFree(d_A_chunk);
        cudaFree(d_B_chunk);
        cudaFree(d_C);
    }

    // Parallel-K reduction implementation
    static void parallelKReduction(
        const std::vector<float> &h_A,
        const std::vector<float> &h_B,
        std::vector<float> &h_C,
        int M, int K, int N,
        int K_prime)
    {
        int num_chunks = (K + K_prime - 1) / K_prime;

        // Allocate device memory
        float *d_A, *d_B;
        cudaMalloc(&d_A, M * K * sizeof(float));
        cudaMalloc(&d_B, K * N * sizeof(float));

        // Copy input matrices to device
        cudaMemcpy(d_A, h_A.data(), M * K * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B.data(), K * N * sizeof(float), cudaMemcpyHostToDevice);

        // Allocate memory for partial results
        float *d_C_partials;
        cudaMalloc(&d_C_partials, num_chunks * M * N * sizeof(float));

        // Set up grid and block dimensions
        dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridDim(
            (N + BLOCK_SIZE - 1) / BLOCK_SIZE,
            (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

        // Launch kernels for all chunks in parallel
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            parallelKReductionKernel<<<gridDim, blockDim>>>(
                d_A, d_B,
                d_C_partials + (chunk * M * N),
                M, K, N, K_prime, chunk);
        }

        // Allocate and initialize result matrix
        float *d_C;
        cudaMalloc(&d_C, M * N * sizeof(float));
        cudaMemset(d_C, 0, M * N * sizeof(float));

        // Sum up partial results (could be done with another kernel)
        std::vector<float> h_C_partial(M * N);
        for (int chunk = 0; chunk < num_chunks; ++chunk)
        {
            cudaMemcpy(h_C_partial.data(),
                       d_C_partials + (chunk * M * N),
                       M * N * sizeof(float),
                       cudaMemcpyDeviceToHost);

            // Accumulate on CPU (could be optimized with a CUDA reduction kernel)
            for (int i = 0; i < M * N; ++i)
            {
                h_C[i] += h_C_partial[i];
            }
        }

        // Cleanup
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
        cudaFree(d_C_partials);
    }
};

int main() {
    // Define matrix dimensions
    int M = 10; // Number of rows
    int K = 10; // Number of columns in A and rows in B
    int N = 10; // Number of columns in B
    int K_prime = 2; // Chunk size

    // Initialize host matrices
    std::vector<float> h_A(M * K, 1.0f); // Example initialization
    std::vector<float> h_B(K * N, 2.0f); // Example initialization
    std::vector<float> h_C(M * N, 0.0f); // Result matrix

    // Call the parallel K reduction
    CudaKReduction::parallelKReduction(h_A, h_B, h_C, M, K, N, K_prime);

    // Optionally, print the result or check for correctness
    for (int i = 0; i < M * N; ++i) {
        printf("%f ", h_C[i]);
    }

    return 0;
}
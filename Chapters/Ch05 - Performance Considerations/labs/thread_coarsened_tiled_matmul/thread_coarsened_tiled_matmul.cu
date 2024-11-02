/*
cd /home/lzydaphne/Programming-Massively-Parallel-Processors/Chapters/"Ch05 - Performance Considerations"/labs/thread_coarsened_tiled_matmul
nvcc -o thread_coarsened_tiled_matmul thread_coarsened_tiled_matmul.cu
./thread_coarsened_tiled_matmul
*/

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_WIDTH 2
#define COARSE_FACTOR 2
#define BLOCK_DIM (TILE_WIDTH/COARSE_FACTOR)

// Utility function to check CUDA errors
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while (0)

// Kernel definition
__global__ void matrixMulKernel(float* M, float* N, float* P, int width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];
    
    int bx = blockIdx.x;  int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;
    
    // Identify the row and column of the P element to work on
    int row = by*TILE_WIDTH + ty;
    int colStart = bx*TILE_WIDTH*COARSE_FACTOR + tx;
    
    // Initialize Pvalue for all output elements
    float Pvalue[COARSE_FACTOR];
    for(int c = 0; c < COARSE_FACTOR; ++c) {
        Pvalue[c] = 0.0f;
    }
    
    // Loop over the M and N tiles required to compute P element
    for(int ph = 0; ph < width/TILE_WIDTH; ++ph) {
        // Collaborative loading of M tile into shared memory
        Mds[ty][tx] = M[row*width + ph*TILE_WIDTH + tx];
        
        for(int c = 0; c < COARSE_FACTOR; ++c) {
            int col = colStart + c*TILE_WIDTH;
            
            // Collaborative loading of N tile into shared memory
            Nds[ty][tx] = N[(ph*TILE_WIDTH + ty)*width + col];
            __syncthreads();
            
            for(int k = 0; k < TILE_WIDTH; ++k) {
                Pvalue[c] += Mds[ty][k]*Nds[k][tx];
            }
            __syncthreads();
        }
    }
    
    for(int c = 0; c < COARSE_FACTOR; ++c) {
        int col = colStart + c*TILE_WIDTH;
        P[row*width + col] = Pvalue[c];
    }
}

// CPU matrix multiplication for verification
void matrixMulCPU(float* M, float* N, float* P, int width) {
    for(int row = 0; row < width; ++row) {
        for(int col = 0; col < width; ++col) {
            float sum = 0.0f;
            for(int k = 0; k < width; ++k) {
                sum += M[row*width + k] * N[k*width + col];
            }
            P[row*width + col] = sum;
        }
    }
}

// Verify results
bool verifyResults(float* cpu_result, float* gpu_result, int width) {
    const float epsilon = 1e-5;
    for(int i = 0; i < width*width; ++i) {
        if(abs(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at position %d: CPU=%f, GPU=%f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int width = 4; // Matrix dimension
    // int width = 16; // Matrix dimension
    size_t size = width * width * sizeof(float);
    
    // Allocate host memory
    float *h_M = (float*)malloc(size);
    float *h_N = (float*)malloc(size);
    float *h_P = (float*)malloc(size);
    float *h_CPU_P = (float*)malloc(size);
    
    // Initialize input matrices with random values
    for(int i = 0; i < width*width; ++i) {
        h_M[i] = rand() / (float)RAND_MAX;
        h_N[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_M, *d_N, *d_P;
    CHECK_CUDA_ERROR(cudaMalloc(&d_M, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_N, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_P, size));
    
    // Copy input matrices from host to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_M, h_M, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_N, h_N, size, cudaMemcpyHostToDevice));
    
    // Set up execution configuration
    // dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    // dim3 gridDim(width/(TILE_WIDTH*COARSE_FACTOR), width/TILE_WIDTH);
    // Launch kernel
    dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
    dim3 gridDim(ceil((float)width / (blockDim.x * COARSE_FACTOR)), ceil((float)width / blockDim.y));
    
    // Launch kernel
    printf("Launching kernel with grid(%d,%d) block(%d,%d)\n", 
           gridDim.x, gridDim.y, blockDim.x, blockDim.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    matrixMulKernel<<<gridDim, blockDim>>>(d_M, d_N, d_P, width);
    cudaEventRecord(stop);
    
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Kernel execution time: %f ms\n", milliseconds);
    
    // Copy result from device to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_P, d_P, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU result for verification
    printf("Computing CPU result for verification...\n");
    matrixMulCPU(h_M, h_N, h_CPU_P, width);
    
    // Verify results
    if(verifyResults(h_CPU_P, h_P, width)) {
        printf("Test PASSED\n");
    } else {
        printf("Test FAILED\n");
        //print out result from host and device
        printf("GPU result:\n");
        for(int i = 0; i < width; ++i) {
            for(int j = 0; j < width; ++j) {
                printf("%f ", h_P[i*width + j]);
            }
            printf("\n");
        }
        printf("\n");

        printf("CPU result:\n");
        for(int i = 0; i < width; ++i) {
            for(int j = 0; j < width; ++j) {
                printf("%f ", h_CPU_P[i*width + j]);
            }
            printf("\n");
        }
        printf("\n");
    }
    
    
    // Calculate and print performance metrics
    float numOperations = 2.0f * width * width * width; // multiply-adds
    float gflops = (numOperations * 1e-9f) / (milliseconds * 1e-3f);
    printf("Performance: %.2f GFLOPS\n", gflops);
    
    // Free device memory
    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
    
    // Free host memory
    free(h_M);
    free(h_N);
    free(h_P);
    free(h_CPU_P);
    
    return 0;
}
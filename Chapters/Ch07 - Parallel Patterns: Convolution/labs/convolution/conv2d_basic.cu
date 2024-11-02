/*
nvcc -o conv2d_basic conv2d_basic.cu
./conv2d_basic
*/
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Constants
#define IN_TILE_DIM 32
#define FILTER_RADIUS 4
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
#define BLOCK_DIM 16
#define CHECK_CUDA_ERROR(err) \
    if (err != cudaSuccess) { \
        printf("CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
        exit(1); \
    }

// Kernel definition from the image
__constant__ float Filter[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_tiled_2D_const_mem_kernel(float *N, float *P,
                                                    int width, int height) {
    int col = blockIdx.x*OUT_TILE_DIM + threadIdx.x - FILTER_RADIUS;
    int row = blockIdx.y*OUT_TILE_DIM + threadIdx.y - FILTER_RADIUS;
    
    // Loading input tile
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    
    if(row>=0 && row<height && col>=0 && col<width) {
        N_s[threadIdx.y][threadIdx.x] = N[row*width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    
    __syncthreads();
    
    // Calculating output elements
    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;
    
    // turning off the threads at the edges of the block
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol>=0 && tileCol<OUT_TILE_DIM && tileRow>=0 && tileRow<OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2*FILTER_RADIUS+1; fRow++) {
                for (int fCol = 0; fCol < 2*FILTER_RADIUS+1; fCol++) {
                    Pvalue += Filter[fRow][fCol]*N_s[tileRow+fRow][tileCol+fCol];
                }
            }
            P[row*width+col] = Pvalue;
        }
    }
}

// CPU version for verification
void convolution_cpu(float *N, float *P, float *Filter, int width, int height) {
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            float sum = 0.0f;
            for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                    int r = row + i;
                    int c = col + j;
                    float value = 0.0f;
                    if(r >= 0 && r < height && c >= 0 && c < width) {
                        value = N[r*width + c];
                    }
                    sum += value * Filter[(i+FILTER_RADIUS)*(2*FILTER_RADIUS+1) + (j+FILTER_RADIUS)];
                }
            }
            P[row*width + col] = sum;
        }
    }
}

// Verification function
bool verify_results(float *cpu_result, float *gpu_result, int width, int height) {
    const float epsilon = 1e-5;
    for(int i = 0; i < width * height; i++) {
        if(abs(cpu_result[i] - gpu_result[i]) > epsilon) {
            printf("Mismatch at position %d: CPU = %f, GPU = %f\n", 
                   i, cpu_result[i], gpu_result[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Problem dimensions
    int width = 1024;
    int height = 1024;
    size_t size = width * height * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    
    // Create test filter
    float h_filter[(2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)];
    for(int i = 0; i < (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1); i++) {
        h_filter[i] = 1.0f / ((2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1));
    }
    
    // Initialize input with random values
    for(int i = 0; i < width * height; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, size);
    cudaMalloc(&d_output, size);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(Filter, h_filter, (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1)*sizeof(float));
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Calculate grid and block dimensions
    dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
                 (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    
    // Launch kernel
    convolution_tiled_2D_const_mem_kernel<<<gridDim, blockDim>>>(d_input, d_output, width, height);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);
    CHECK_CUDA_ERROR(cudaGetLastError());
    
    // Compute CPU version
    convolution_cpu(h_input, h_output_cpu, h_filter, width, height);
    
    // Verify results
    bool passed = verify_results(h_output_cpu, h_output_gpu, width, height);
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    //if passed, print the first 10 elements of the output
    if (passed) {
        for (int i = 0; i < 10; i++) {
            printf("CPU: %f, GPU: %f\n", h_output_cpu[i], h_output_gpu[i]);
        }
    }

    
    // Free memory
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    
    return 0;
}
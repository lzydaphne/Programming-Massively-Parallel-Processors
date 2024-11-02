/*
nvcc -o conv2d_cache_for_halo conv2d_cache_for_halo.cu
./conv2d_cache_for_halo
*/
#include <stdio.h>
#include <stdlib.h>

#define TILE_DIM 32
#define FILTER_RADIUS 4
#define BLOCK_SIZE TILE_DIM
#define checkCudaErrors(call) { gpuAssert((call), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s %s %d\n", cudaGetErrorString(code), file, line);
        exit(code);
    }
}

// Constant memory declaration for filter
__constant__ float Filter[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1];

__global__ void convolution_cached_tiled_2D_const_mem_kernel(float *N, float *P,
                                                             int width,
                                                             int height) {
  int row = blockIdx.y * TILE_DIM + threadIdx.y;
  int col = blockIdx.x * TILE_DIM + threadIdx.x;

  __shared__ float N_s[TILE_DIM][TILE_DIM];

//   if (col >= 0 && col < width && row >= 0 && row <= height) {
  if (col < width && row < height) {
    N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
  } else {
    N_s[threadIdx.y][threadIdx.x] = 0.0;
  }
  __syncthreads();
  
  int tileRow = threadIdx.y - FILTER_RADIUS;
  int tileCol = threadIdx.x - FILTER_RADIUS;

  if (col < width && row < height) {
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
      for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
        // calulating elements in shared memory, check for halo cells
        // tests whether the input element falls within the interior of the input tile.
        // If it does, the element is read from shared memory.
        int ty = threadIdx.y - FILTER_RADIUS + fRow;
        int tx = threadIdx.x - FILTER_RADIUS + fCol;
        
        // If within shared memory bounds, use shared memory
        if (ty >= 0 && ty < TILE_DIM && tx >= 0 && tx < TILE_DIM) {
            Pvalue += Filter[fRow][fCol] * N_s[ty][tx];
        }
        // Otherwise, access global memory for halo cells
        else {
            int global_row = row - FILTER_RADIUS + fRow;
            int global_col = col - FILTER_RADIUS + fCol;
            
            if (global_row >= 0 && global_row < height &&
                global_col >= 0 && global_col < width) {
                Pvalue += Filter[fRow][fCol] * N[global_row * width + global_col];
            }
        }
        // if (tileCol + fCol >= 0 &&
        //     tileCol + fCol < TILE_DIM &&
        //     tileRow + fRow >= 0 &&
        //     tileRow + fRow < TILE_DIM) {
        //   Pvalue +=
        //       Filter[fRow][fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
        // }


        // calulating elements not in shared memory but probably cached in L2 cache
        // if not, check whether the halo cells are ghost cells. 
        //If so, no action is taken for the ele-ment, since we assume that the ghost values are 0.
        // Otherwise, the element is accessed from the global memory.
        // else {
        //   if (row - FILTER_RADIUS + fRow >= 0 &&
        //       row - FILTER_RADIUS + fRow < height &&
        //       col - FILTER_RADIUS + fCol >= 0 &&
        //       col - FILTER_RADIUS + fCol < width) {
        //     Pvalue += Filter[fRow][fCol] * N[(row - FILTER_RADIUS + fRow) * width +
        //                                   col - FILTER_RADIUS + fCol];
        //   }
        // }
      }
      P[row * width + col] = Pvalue;
    }
  }
}


// CPU reference implementation
void convolution2D_CPU(float *N, float *P, float *F, int width, int height) {
    for(int row = 0; row < height; row++) {
        for(int col = 0; col < width; col++) {
            float sum = 0.0f;
            for(int i = -FILTER_RADIUS; i <= FILTER_RADIUS; i++) {
                for(int j = -FILTER_RADIUS; j <= FILTER_RADIUS; j++) {
                    if(row + i >= 0 && row + i < height && 
                       col + j >= 0 && col + j < width) {
                        sum += F[(i+FILTER_RADIUS)*(2*FILTER_RADIUS+1) + (j+FILTER_RADIUS)] * 
                              N[(row+i)*width + (col+j)];
                    }
                }
            }
            P[row*width + col] = sum;
        }
    }
}

// Verify results
bool verify_results(float *cpu_results, float *gpu_results, int width, int height) {
    const float epsilon = 1e-5;
    for(int i = 0; i < width * height; i++) {
        if(abs(cpu_results[i] - gpu_results[i]) > epsilon) {
            printf("Mismatch at position %d: CPU = %f, GPU = %f\n", 
                   i, cpu_results[i], gpu_results[i]);
            return false;
        }
    }
    return true;
}

int main() {
    // Problem dimensions
    int width = 1024;
    int height = 1024;
    
    // Allocate host memory
    size_t size = width * height * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    
    // Create filter
    const int filterSize = (2*FILTER_RADIUS+1)*(2*FILTER_RADIUS+1);
    float *h_filter = (float*)malloc(filterSize * sizeof(float));
    
    // Initialize input and filter
    for(int i = 0; i < width * height; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    
    // Initialize filter (example: Gaussian-like)
    for(int i = 0; i < 2*FILTER_RADIUS+1; i++) {
        for(int j = 0; j < 2*FILTER_RADIUS+1; j++) {
            int dist = (i-FILTER_RADIUS)*(i-FILTER_RADIUS) + 
                      (j-FILTER_RADIUS)*(j-FILTER_RADIUS);
            h_filter[i*(2*FILTER_RADIUS+1) + j] = exp(-dist/10.0f);
        }
    }
    
    // Allocate device memory
    float *d_input, *d_output;
    checkCudaErrors(cudaMalloc(&d_input, size));
    checkCudaErrors(cudaMalloc(&d_output, size));
    
    // Copy data to device
    checkCudaErrors(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyToSymbol(Filter, h_filter, filterSize * sizeof(float)));
    
    // Launch kernel
    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid((width + dimBlock.x - 1) / dimBlock.x,
                 (height + dimBlock.y - 1) / dimBlock.y);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    convolution_cached_tiled_2D_const_mem_kernel<<<dimGrid, dimBlock>>>(
        d_input, d_output, width, height);
    cudaEventRecord(stop);
    
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    
    // Copy result back to host
    checkCudaErrors(cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost));
    
    // Compute CPU reference
    convolution2D_CPU(h_input, h_output_cpu, h_filter, width, height);
    
    // Verify results
    bool passed = verify_results(h_output_cpu, h_output_gpu, width, height);
    printf("Test %s\n", passed ? "PASSED" : "FAILED");
    printf("GPU Execution time: %f ms\n", milliseconds);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);
    free(h_filter);
    
    return 0;
}
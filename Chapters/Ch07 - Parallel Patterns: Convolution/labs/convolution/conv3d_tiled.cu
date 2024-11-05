/*
let us think up a kernel to optimize the performance, do not output the final kernel directly, i want step by step discussion
*/
/*
cd /home/lzydaphne/Programming-Massively-Parallel-Processors/Chapters/"Ch07 - Parallel Patterns: Convolution"/labs/convolution
nvcc -o conv3d_tiled conv3d_tiled.cu
./conv3d_tiled
*/
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

// Constants
#define IN_TILE_DIM 8
#define FILTER_RADIUS 3
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2*(FILTER_RADIUS))
#define BLOCK_DIM IN_TILE_DIM

// Declare constant memory for filter
__constant__ float Filter[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1];

// Error checking macro
#define CHECK_CUDA_ERROR(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA Error: %s at line %d\n", cudaGetErrorString(err), __LINE__); \
            exit(1); \
        } \
    } while(0)

// Fixed kernel with corrections
__global__ void convolution_tiled_3D_const_mem_kernel(float *N, float *P, int32_t width, int32_t height, int32_t length) {
    // Thread indices within the block
    int32_t tx = threadIdx.x;
    int32_t ty = threadIdx.y;
    int32_t tz = threadIdx.z;
    
    // Global indices
    int32_t col = blockIdx.x * OUT_TILE_DIM + tx;
    int32_t row = blockIdx.y * OUT_TILE_DIM + ty;
    int32_t level = blockIdx.z * OUT_TILE_DIM + tz;
    
    // Shared memory - note the correct dimension ordering
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM][IN_TILE_DIM];
    
    // Load input tile to shared memory
    for(int32_t z = tz; z < IN_TILE_DIM; z += BLOCK_DIM) {
        for(int32_t y = ty; y < IN_TILE_DIM; y += BLOCK_DIM) {
            for(int32_t x = tx; x < IN_TILE_DIM; x += BLOCK_DIM) {
                int32_t global_z = blockIdx.z * OUT_TILE_DIM + z - FILTER_RADIUS;
                int32_t global_y = blockIdx.y * OUT_TILE_DIM + y - FILTER_RADIUS;
                int32_t global_x = blockIdx.x * OUT_TILE_DIM + x - FILTER_RADIUS;
                
                if(global_z >= 0 && global_z < length &&
                   global_y >= 0 && global_y < height &&
                   global_x >= 0 && global_x < width) {
                    N_s[z][y][x] = N[(global_z * height + global_y) * width + global_x];
                } else {
                    N_s[z][y][x] = 0.0f;
                }
            }
        }
    }
    
    __syncthreads();
    
    // Compute output only if this thread is responsible for an output element
    if(tx < OUT_TILE_DIM && ty < OUT_TILE_DIM && tz < OUT_TILE_DIM &&
       col < width && row < height && level < length) {
        float Pvalue = 0.0f;
        
        // Convolution computation with correct shared memory indexing
        for(int32_t fz = 0; fz < 2 * FILTER_RADIUS + 1; fz++) {
            for(int32_t fy = 0; fy < 2 * FILTER_RADIUS + 1; fy++) {
                for(int32_t fx = 0; fx < 2 * FILTER_RADIUS + 1; fx++) {
                    Pvalue += Filter[fz][fy][fx] * 
                             N_s[tz + fz][ty + fy][tx + fx];
                }
            }
        }
        
        P[(level * height + row) * width + col] = Pvalue;
    }
}

// Host function to initialize the filter with sample values
void initialize_filter(float *filter)
{
    for (int z = 0; z < 2 * FILTER_RADIUS + 1; z++) {
        for (int y = 0; y < 2 * FILTER_RADIUS + 1; y++) {
            for (int x = 0; x < 2 * FILTER_RADIUS + 1; x++) {
                // Simple Gaussian-like filter
                float dist = (x - FILTER_RADIUS) * (x - FILTER_RADIUS) + 
                           (y - FILTER_RADIUS) * (y - FILTER_RADIUS) +
                           (z - FILTER_RADIUS) * (z - FILTER_RADIUS);
                int idx = z * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) +
                         y * (2 * FILTER_RADIUS + 1) + x;
                filter[idx] = exp(-dist / (2.0f * FILTER_RADIUS * FILTER_RADIUS));
            }
        }
    }
}

// Host function to verify results
void verify_result(float *input, float *output, float *filter, 
                  int width, int height, int length)
{
    float *gold = (float *)malloc(width * height * length * sizeof(float));
    
    // Compute golden result on CPU
    for (int z = 0; z < length; z++) {
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                float sum = 0.0f;
                for (int fz = 0; fz < 2 * FILTER_RADIUS + 1; fz++) {
                    for (int fy = 0; fy < 2 * FILTER_RADIUS + 1; fy++) {
                        for (int fx = 0; fx < 2 * FILTER_RADIUS + 1; fx++) {
                            int iz = z + fz - FILTER_RADIUS;
                            int iy = y + fy - FILTER_RADIUS;
                            int ix = x + fx - FILTER_RADIUS;
                            
                            float value = 0.0f;
                            if (iz >= 0 && iz < length && 
                                iy >= 0 && iy < height && 
                                ix >= 0 && ix < width) {
                                value = input[(iz * height + iy) * width + ix];
                            }
                            
                            int fidx = fz * (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) +
                                     fy * (2 * FILTER_RADIUS + 1) + fx;
                            sum += filter[fidx] * value;
                        }
                    }
                }
                gold[(z * height + y) * width + x] = sum;
            }
        }
    }
    
    // Compare results
    bool correct = true;
    for (int i = 0; i < width * height * length; i++) {
        if (fabs(gold[i] - output[i]) > 1e-4) {
            printf("Mismatch at position %d: Expected %f, Got %f\n", 
                   i, gold[i], output[i]);
            correct = false;
            break;
        }
    }
    
    if (correct) {
        printf("Test PASSED!\n");
    } else {
        printf("Test FAILED!\n");
    }
    
    free(gold);
}

int main()
{
    const int width = 64;
    const int height = 64;
    const int length = 2;
    const int size = width * height * length * sizeof(float);
    const int filter_size = (2 * FILTER_RADIUS + 1) * (2 * FILTER_RADIUS + 1) * 
                           (2 * FILTER_RADIUS + 1) * sizeof(float);
    
    // Allocate host memory
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    float *h_filter = (float *)malloc(filter_size);
    
    // Initialize input data and filter
    for (int i = 0; i < width * height * length; i++) {
        h_input[i] = rand() / (float)RAND_MAX;
    }
    initialize_filter(h_filter);
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA_ERROR(cudaMalloc(&d_input, size));
    CHECK_CUDA_ERROR(cudaMalloc(&d_output, size));
    
    // Copy data to device
    CHECK_CUDA_ERROR(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpyToSymbol(Filter, h_filter, filter_size));
    
    // Calculate grid and block dimensions
    dim3 blockDim(BLOCK_DIM, BLOCK_DIM, BLOCK_DIM);
    dim3 gridDim(
        (width + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM,
        (length + OUT_TILE_DIM - 1) / OUT_TILE_DIM
    );
    
    // Launch kernel
    convolution_tiled_3D_const_mem_kernel<<<gridDim, blockDim>>>(
        d_input, d_output, width, height, length);
    
    // Check for kernel launch errors
    CHECK_CUDA_ERROR(cudaGetLastError());
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());
    
    // Copy result back to host
    CHECK_CUDA_ERROR(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    
    // Verify results
    verify_result(h_input, h_output, h_filter, width, height, length);
    
    // Free memory
    free(h_input);
    free(h_output);
    free(h_filter);
    CHECK_CUDA_ERROR(cudaFree(d_input));
    CHECK_CUDA_ERROR(cudaFree(d_output));
    
    return 0;
}
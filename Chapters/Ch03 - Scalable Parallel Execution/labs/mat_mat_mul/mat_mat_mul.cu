#include <stdio.h>
#include <cuda.h>

#define DEBUG 1
#define BLOCK_SIZE 2  // Typical block size for matrix multiplication

__global__
void matrix_multiply(float *A, float *B, float *C, int width)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

int main(int argc, char **argv)
{
    int n;
    float *h_A, *h_B, *h_C;
    float *d_A, *d_B, *d_C;

    if (argc != 2) {
        printf("Usage: ./a.out <n>\n");
        return 1;
    }

    n = atoi(argv[1]);

    // Allocate memory on host
    h_A = (float *)malloc(n * n * sizeof(float));  // Changed to n*n for matrix
    h_B = (float *)malloc(n * n * sizeof(float));
    h_C = (float *)malloc(n * n * sizeof(float));  // Changed to n*n for result matrix

    // Initialize host memory
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            h_A[i * n + j] = 2.0f;  // Initialize matrix A
            h_B[i * n + j] = 3.0f;  // Initialize matrix B
            h_C[i * n + j] = 0.0f;  // Initialize result matrix C
        }
    }

    // Allocate memory on device
    cudaMalloc((void **)&d_A, n * n * sizeof(float));
    cudaMalloc((void **)&d_B, n * n * sizeof(float));
    cudaMalloc((void **)&d_C, n * n * sizeof(float));

    // Copy host memory to device memory
    cudaMemcpy(d_A, h_A, n * n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, n * n * sizeof(float), cudaMemcpyHostToDevice);

    // Set up execution configuration
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize((n + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                  (n + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Block size: %d x %d\n", blockSize.x, blockSize.y);
    printf("Grid size: %d x %d\n", gridSize.x, gridSize.y);

    // Launch the kernel
    matrix_multiply<<<gridSize, blockSize>>>(d_A, d_B, d_C, n);

    // Copy result back to host
    cudaMemcpy(h_C, d_C, n * n * sizeof(float), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    // Print the result (first few elements)
    printf("Result matrix (showing first few elements):\n");
    for (int i = 0; i < min(n, 4); i++) {
        for (int j = 0; j < min(n, 4); j++) {
            printf("%.2f ", h_C[i * n + j]);
        }
        printf("\n");
    }
#endif

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
#include <iostream>
#include <cstdlib>
#include <cassert>
#include <cmath>
#include <cuda_runtime.h>
#include <zlib.h>
#include <png.h>

#define Z 2
#define Y 5
#define X 5
#define xBound X / 2
#define yBound Y / 2
#define SCALE 8

#define MASK_N 2
#define MASK_X 5
#define MASK_Y 5
#define X_BOUND 2
#define Y_BOUND 2
#define CHANNELS 3            // RGB
// #define IN_TILE_DIM 16
// #define FILTER_RADIUS 2


#define FILTER_RADIUS 2
#define BLOCK_DIM_X 20
#define BLOCK_DIM_Y 20
#define OUT_TILE_DIM_X 16
#define OUT_TILE_DIM_Y 16
#define IN_TILE_DIM_X 20  // BLOCK_DIM_X = OUT_TILE_DIM_X + 2*FILTER_RADIUS
#define IN_TILE_DIM_Y 20  // BLOCK_DIM_Y = OUT_TILE_DIM_Y + 2*FILTER_RADIUS





#define MAX_ERROR_THRESHOLD 1.0  // Maximum allowed difference between outputs
inline __device__ int bound_check(int val, int lower, int upper) {
    if (val >= lower && val < upper)
        return 1;
    else
        return 0;
}

// Error checking macro
#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        printf("Error: %s:%d, ", __FILE__, __LINE__);\
        printf("code:%d, reason: %s\n", error, cudaGetErrorString(error));\
        exit(1);\
    }\
}

// Conversion to use lower precision
typedef unsigned char uchar;
typedef unsigned int uint;


int read_png(const char* filename, unsigned char** image, unsigned* height, 
             unsigned* width, unsigned* channels) {

    unsigned char sig[8];
    FILE* infile;
    infile = fopen(filename, "rb");

    fread(sig, 1, 8, infile);
    if (!png_check_sig(sig, 8))
        return 1;   /* bad signature */

    png_structp png_ptr;
    png_infop info_ptr;

    png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (!png_ptr)
        return 4;   /* out of memory */
  
    info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr) {
        png_destroy_read_struct(&png_ptr, NULL, NULL);
        return 4;   /* out of memory */
    }

    png_init_io(png_ptr, infile);
    png_set_sig_bytes(png_ptr, 8);
    png_read_info(png_ptr, info_ptr);
    int bit_depth, color_type;
    png_get_IHDR(png_ptr, info_ptr, width, height, &bit_depth, &color_type, NULL, NULL, NULL);

    png_uint_32  i, rowbytes;
    png_bytep  row_pointers[*height];
    png_read_update_info(png_ptr, info_ptr);
    rowbytes = png_get_rowbytes(png_ptr, info_ptr);
    *channels = (int) png_get_channels(png_ptr, info_ptr);

    if ((*image = (unsigned char *) malloc(rowbytes * *height)) == NULL) {
        png_destroy_read_struct(&png_ptr, &info_ptr, NULL);
        return 3;
    }

    for (i = 0;  i < *height;  ++i)
        row_pointers[i] = *image + i * rowbytes;
    png_read_image(png_ptr, row_pointers);
    png_read_end(png_ptr, NULL);
    return 0;
}

void write_png(const char* filename, png_bytep image, const unsigned height, const unsigned width, 
               const unsigned channels) {
    FILE* fp = fopen(filename, "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_infop info_ptr = png_create_info_struct(png_ptr);
    png_init_io(png_ptr, fp);
    png_set_IHDR(png_ptr, info_ptr, width, height, 8,
                 PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
                 PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);
    png_set_filter(png_ptr, 0, PNG_NO_FILTERS);
    png_write_info(png_ptr, info_ptr);
    png_set_compression_level(png_ptr, 1);

    png_bytep row_ptr[height];
    for (int i = 0; i < height; ++ i) {
        row_ptr[i] = image + i * width * channels * sizeof(unsigned char);
    }
    png_write_image(png_ptr, row_ptr);
    png_write_end(png_ptr, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

__constant__ char mask[Z][Y][X] = {
    {{ -1, -4, -6, -4, -1},
     { -2, -8,-12, -8, -2},
     {  0,  0,  0,  0,  0},
     {  2,  8, 12,  8,  2},
     {  1,  4,  6,  4,  1}},
    {{ -1, -2,  0,  2,  1},
     { -4, -8,  0,  8,  4},
     { -6,-12,  0, 12,  6},
     { -4, -8,  0,  8,  4},
     { -1, -2,  0,  2,  1}}
};

// Original Sobel kernel for validation
__global__ void sobel_original(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {

    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    double val[Z][3];
    if (tid >= height) return;

    int y = tid;
    for (int x = 0; x < width; ++x) {
        /* Z axis of mask */
        for (int i = 0; i < Z; ++i) {

            val[i][2] = 0.;
            val[i][1] = 0.;
            val[i][0] = 0.;

            /* Y and X axis of mask */
            for (int v = -yBound; v <= yBound; ++v) {
                for (int u = -xBound; u <= xBound; ++u) {
                    if (bound_check(x + u, 0, width) && bound_check(y + v, 0, height)) {
                        const unsigned char R = s[channels * (width * (y + v) + (x + u)) + 2];
                        const unsigned char G = s[channels * (width * (y + v) + (x + u)) + 1];
                        const unsigned char B = s[channels * (width * (y + v) + (x + u)) + 0];
                        val[i][2] += R * mask[i][u + xBound][v + yBound];
                        val[i][1] += G * mask[i][u + xBound][v + yBound];
                        val[i][0] += B * mask[i][u + xBound][v + yBound];
                    }
                }
            }
        }
        double totalR = 0.;
        double totalG = 0.;
        double totalB = 0.;
        for (int i = 0; i < Z; ++i) {
            totalR += val[i][2] * val[i][2];
            totalG += val[i][1] * val[i][1];
            totalB += val[i][0] * val[i][0];
        }
        totalR = sqrt(totalR) / SCALE;
        totalG = sqrt(totalG) / SCALE;
        totalB = sqrt(totalB) / SCALE;
        const unsigned char cR = (totalR > 255.) ? 255 : totalR;
        const unsigned char cG = (totalG > 255.) ? 255 : totalG;
        const unsigned char cB = (totalB > 255.) ? 255 : totalB;
        t[channels * (width * y + x) + 2] = cR;
        t[channels * (width * y + x) + 1] = cG;
        t[channels * (width * y + x) + 0] = cB;
    }
}


// Optimized Sobel kernel
// __global__ void sobel_opt(unsigned char* input, unsigned char* output, 
//                          unsigned height, unsigned width, unsigned channels) {
//     // Thread indexing
//     int tx = threadIdx.x;
//     int ty = threadIdx.y;
//     int channel = ty / IN_TILE_DIM;
//     int y_within_channel = ty % IN_TILE_DIM;
    
//     // Global position
//     int row = blockIdx.y * OUT_TILE_DIM + y_within_channel - FILTER_RADIUS;
//     int col = blockIdx.x * OUT_TILE_DIM + tx - FILTER_RADIUS;
    
//     __shared__ unsigned char smem[IN_TILE_DIM*3][IN_TILE_DIM];
    
//     // Load data into shared memory
//     if(row >= 0 && row < height && col >= 0 && col < width) {
//         int input_idx = channels * (width * row + col) + channel;
//         smem[y_within_channel*3 + channel][tx] = input[input_idx];
//     } else {
//         smem[y_within_channel*3 + channel][tx] = 0;
//     }
    
//     __syncthreads();
    
//     // Process only interior pixels
//     if (tx >= FILTER_RADIUS && tx < IN_TILE_DIM-FILTER_RADIUS &&
//         y_within_channel >= FILTER_RADIUS && y_within_channel < IN_TILE_DIM-FILTER_RADIUS) {
        
//         int out_row = blockIdx.y * OUT_TILE_DIM + (y_within_channel - FILTER_RADIUS);
//         int out_col = blockIdx.x * OUT_TILE_DIM + (tx - FILTER_RADIUS);
        
//         if(out_row < height && out_col < width) {
//             double val_h = 0.0, val_v = 0.0;
            
//             for(int dy = -FILTER_RADIUS; dy <= FILTER_RADIUS; dy++) {
//                 for(int dx = -FILTER_RADIUS; dx <= FILTER_RADIUS; dx++) {
//                     unsigned char pixel = smem[(y_within_channel+dy)*3 + channel][tx+dx];
//                     val_h += pixel * mask[0][dy+FILTER_RADIUS][dx+FILTER_RADIUS];
//                     val_v += pixel * mask[1][dy+FILTER_RADIUS][dx+FILTER_RADIUS];
//                 }
//             }
            
//             double total = sqrt(val_h*val_h + val_v*val_v) / SCALE;
//             int output_idx = channels * (width * out_row + out_col) + channel;
//             output[output_idx] = (unsigned char)(min(255.0, total));
//         }
//     }
// }



// __global__ void sobel_final(unsigned char* input, unsigned char* output,
//                            unsigned height, unsigned width, unsigned channels) {

// 	// Thread indexing
//     const int tx = threadIdx.x;
//     const int ty = threadIdx.y;
//     const int row = blockIdx.y * blockDim.y + ty;
//     const int col = blockIdx.x * blockDim.x + tx;
    
//     // Shared memory - separate arrays for each channel
//     __shared__ unsigned char smem[3][IN_TILE_DIM][IN_TILE_DIM];
    
//     // Load data into shared memory
//     if (row < height && col < width) {
//         const int gindex = channels * (row * width + col);
//         // Load BGR values
//         for (int c = 0; c < 3; c++) {
//             smem[c][ty][tx] = input[gindex + c];
//         }
//     } else {
//         // Zero padding for out-of-bounds
//         for (int c = 0; c < 3; c++) {
//             smem[c][ty][tx] = 0;
//         }
//     }
    
//     __syncthreads();
    
//     // Only process valid pixels
//     if (row < height && col < width) {
//         float gradients[3][2] = {{0.0f}}; // [channel][x/y]
        
//         // Process each channel separately
//         for (int c = 0; c < 3; c++) {
//             // Apply Sobel operators
//             if (ty >= 2 && ty < IN_TILE_DIM-2 && tx >= 2 && tx < IN_TILE_DIM-2) {
//                 float grad_x = 0.0f;
//                 float grad_y = 0.0f;
                
//                 #pragma unroll
//                 for (int i = -2; i <= 2; i++) {
//                     #pragma unroll
//                     for (int j = -2; j <= 2; j++) {
//                         const unsigned char pixel = smem[c][ty+i][tx+j];
//                         grad_x += pixel * mask[0][i+2][j+2];
//                         grad_y += pixel * mask[1][i+2][j+2];
//                     }
//                 }
                
//                 gradients[c][0] = grad_x;
//                 gradients[c][1] = grad_y;
//             }
//         }
        
//         // Calculate magnitude for each channel
//         const int out_idx = channels * (row * width + col);
//         for (int c = 0; c < 3; c++) {
//             float magnitude = sqrtf(gradients[c][0] * gradients[c][0] + 
//                                   gradients[c][1] * gradients[c][1]) / SCALE;
            
//             // Threshold the magnitude to create more pronounced edges
//             float threshold = 30.0f; // Adjust this value to control edge sensitivity
//             if (magnitude < threshold) {
//                 magnitude = 0.0f;
//             }
            
//             // Clamp the value
//             magnitude = min(255.0f, magnitude);
            
//             // Write output
//             output[out_idx + c] = (unsigned char)magnitude;
//         }
//     }
// }


__global__ void sobel_opt_1(unsigned char *s, unsigned char *t, unsigned height, unsigned width, unsigned channels) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Use float2 for better memory alignment and vectorized operations
    float2 val[Z][2];  // Keep using float for main computations
    if (col >= width) return;
    int row = blockIdx.y;
    __shared__ unsigned char shared_R[5][516];
    __shared__ unsigned char shared_G[5][516];
    __shared__ unsigned char shared_B[5][516];

    // Load data to shared memory (keep this part in regular integers as it's memory operations)
    #pragma unroll 5
    for (int v = -yBound; v <= yBound; ++v) {
        if(bound_check(row + v, 0, height)){
            int idx;
            int idx_y = v + yBound;
            if(threadIdx.x == 0){
                if(col - 2 >= 0){
                    idx = channels * (width * (row + v) + (col - 2));
                    shared_R[idx_y][0] = s[idx + 2];
                    shared_G[idx_y][0] = s[idx + 1];
                    shared_B[idx_y][0] = s[idx + 0];
                }
                if(col - 1 >= 0){
                    idx = channels * (width * (row + v) + (col - 1));
                    shared_R[idx_y][1] = s[idx + 2];
                    shared_G[idx_y][1] = s[idx + 1];
                    shared_B[idx_y][1] = s[idx + 0];
                }
            }
            idx = channels * (width * (row + v) + col);
            shared_R[idx_y][threadIdx.x + xBound] = s[idx + 2];
            shared_G[idx_y][threadIdx.x + xBound] = s[idx + 1];
            shared_B[idx_y][threadIdx.x + xBound] = s[idx + 0];

            if(threadIdx.x == blockDim.x - 1){
                if(col + 2 < width){
                    idx = channels * (width * (row + v) + col + 2);
                    shared_R[idx_y][threadIdx.x + xBound + 2] = s[idx + 2];
                    shared_G[idx_y][threadIdx.x + xBound + 2] = s[idx + 1];
                    shared_B[idx_y][threadIdx.x + xBound + 2] = s[idx + 0];
                }
                if(col + 1 < width){
                    idx = channels * (width * (row + v) + col + 1);
                    shared_R[idx_y][threadIdx.x + xBound + 1] = s[idx + 2];
                    shared_G[idx_y][threadIdx.x + xBound + 1] = s[idx + 1];
                    shared_B[idx_y][threadIdx.x + xBound + 1] = s[idx + 0];
                }
            }
        }
    }
    __syncthreads();

    // Process in vectorized form
    #pragma unroll 2
    for (int i = 0; i < Z; ++i) {
        val[i][0] = make_float2(0.f, 0.f);  // RG
        val[i][1] = make_float2(0.f, 0.f);  // BA

        for (int v = -yBound; v <= yBound; ++v) {
            for (int u = -xBound; u <= xBound; ++u) {
                if (bound_check(col + u, 0, width) && bound_check(row + v, 0, height)) {
                    int idx = threadIdx.x + xBound;
                    float mask_val = mask[i][u + xBound][v + yBound];
                    
                    // Process RG together
                    val[i][0].x += shared_R[v + yBound][idx + u] * mask_val;
                    val[i][0].y += shared_G[v + yBound][idx + u] * mask_val;
                    // Process B
                    val[i][1].x += shared_B[v + yBound][idx + u] * mask_val;
                }
            }
        }
    }

    // Compute final values using vectorized operations
    float2 totalRG = make_float2(0.f, 0.f);
    float2 totalB = make_float2(0.f, 0.f);

    for (int i = 0; i < Z; ++i) {
        totalRG.x += val[i][0].x * val[i][0].x;  // R
        totalRG.y += val[i][0].y * val[i][0].y;  // G
        totalB.x += val[i][1].x * val[i][1].x;   // B
    }

    // Final calculations
    totalRG.x = sqrtf(totalRG.x) / SCALE;
    totalRG.y = sqrtf(totalRG.y) / SCALE;
    totalB.x = sqrtf(totalB.x) / SCALE;

    const unsigned char cR = (totalRG.x > 255.f) ? 255 : totalRG.x;
    const unsigned char cG = (totalRG.y > 255.f) ? 255 : totalRG.y;
    const unsigned char cB = (totalB.x > 255.f) ? 255 : totalB.x;

    t[channels * (width * row + col) + 2] = cR;
    t[channels * (width * row + col) + 1] = cG;
    t[channels * (width * row + col) + 0] = cB;
}


// Optimized Sobel kernel
__global__ void sobel_opt_2(unsigned char* input, unsigned char* output, 
                         unsigned height, unsigned width, unsigned channels) {

    // Keep existing position calculations
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int row = blockIdx.y * OUT_TILE_DIM_Y + ty - FILTER_RADIUS;
    int col = blockIdx.x * OUT_TILE_DIM_X + tx - FILTER_RADIUS;
    
    __shared__ unsigned char s_tile[IN_TILE_DIM_Y][IN_TILE_DIM_X][3];
    
    // Loading part stays the same
    if (row >= 0 && row < height && col >= 0 && col < width) {
        int idx = channels * (width * row + col);
        s_tile[ty][tx][0] = input[idx];
        s_tile[ty][tx][1] = input[idx + 1];
        s_tile[ty][tx][2] = input[idx + 2];
    } else {
        s_tile[ty][tx][0] = 0;
        s_tile[ty][tx][1] = 0;
        s_tile[ty][tx][2] = 0;
    }
    
    __syncthreads();
    
    // Computation part with optimizations
    if (tx >= FILTER_RADIUS && tx < IN_TILE_DIM_X - FILTER_RADIUS && 
        ty >= FILTER_RADIUS && ty < IN_TILE_DIM_Y - FILTER_RADIUS) {
        
        int out_row = blockIdx.y * OUT_TILE_DIM_Y + (ty - FILTER_RADIUS);
        int out_col = blockIdx.x * OUT_TILE_DIM_X + (tx - FILTER_RADIUS);
        
        if (out_row < height && out_col < width) {
            // Change to float for better performance
            float val[Z][3] = {{0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}};
            
            // Unroll the loops for 5x5 filter
            #pragma unroll
            for (int v = -2; v <= 2; ++v) {
                #pragma unroll
                for (int u = -2; u <= 2; ++u) {
                    int ty_offset = ty + v;
                    int tx_offset = tx + u;
                    
                    // Load mask values once for both gx and gy
                    char mx = mask[1][u + 2][v + 2];  // gx mask
                    char my = mask[0][u + 2][v + 2];  // gy mask
                    
                    // Process all channels together
                    #pragma unroll
                    for (int c = 0; c < 3; ++c) {
                        float pixel = s_tile[ty_offset][tx_offset][c];
                        val[0][c] += pixel * my;  // gy
                        val[1][c] += pixel * mx;  // gx
                    }
                }
            }
            
            // Calculate final values for all channels
            int out_idx = channels * (width * out_row + out_col);
            
            #pragma unroll
            for (int c = 0; c < 3; ++c) {
                float gx = val[1][c];
                float gy = val[0][c];
                float mag = sqrtf(gx * gx + gy * gy) / SCALE;  // Using sqrtf instead of sqrt
                output[out_idx + c] = (unsigned char)(min(max(mag, 0.0f), 255.0f));
            }
        }
    }
    
}



// Validation function
float validate_results(unsigned char* ref, unsigned char* opt, int size) {
    float max_error = 0.0f;
    float avg_error = 0.0f;
    int num_errors = 0;
    
    for(int i = 0; i < size; i++) {
        float error = abs(ref[i] - opt[i]);
        avg_error += error;
        max_error = max(max_error, error);
        if(error > MAX_ERROR_THRESHOLD) {
            num_errors++;
        }
    }
    
    avg_error /= size;
    printf("Validation Results:\n");
    printf("Max Error: %.2f\n", max_error);
    printf("Avg Error: %.2f\n", avg_error);
    printf("Number of errors above threshold: %d\n", num_errors);
    
    return max_error;
}

int main() {
    const char* input_file = "candy.png";
    unsigned height, width, channels;
    unsigned char *src = NULL, *dst_orig, *dst_opt;
    unsigned char *dsrc, *ddst_orig, *ddst_opt;

    /* read the image to src, and get height, width, channels */
     printf("Reading input image...\n");
    if (read_png(input_file, &src, &height, &width, &channels)) {
        std::cerr << "Error in read png" << std::endl;
        return -1;
    }

    printf("Image dimensions: %dx%d with %d channels\n", width, height, channels);

    // Allocate host memory for both original and optimized outputs
    size_t image_size = height * width * channels * sizeof(unsigned char);

    dst_orig = (unsigned char *)malloc(image_size);
    dst_opt = (unsigned char *)malloc(image_size);
    
    // Register source image for faster transfers
    cudaHostRegister(src, image_size, cudaHostRegisterDefault);

    // Allocate device memory
    CHECK(cudaMalloc(&dsrc, image_size));
    CHECK(cudaMalloc(&ddst_orig, image_size));
    CHECK(cudaMalloc(&ddst_opt, image_size));

    // Copy source image to device
    CHECK(cudaMemcpy(dsrc, src, image_size, cudaMemcpyHostToDevice));

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float elapsed_time;

    // Run original Sobel kernel
    const int threads_per_block = 256;
    const int num_blocks = (height + threads_per_block - 1) / threads_per_block;
    
    printf("\nRunning original Sobel kernel...\n");
    cudaEventRecord(start);
    sobel_original<<<num_blocks, threads_per_block>>>(dsrc, ddst_orig, height, width, channels);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Original kernel time: %.3f ms\n", elapsed_time);

    // Run optimized Sobel kernel
    // dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM);
    // dim3 blockDim(IN_TILE_DIM, IN_TILE_DIM*3);  // 16x48 threads
    // dim3 gridDim((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, 
    //              (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    // const int threads_per_block = 256;
	const int number_of_blocks = (width + threads_per_block - 1) / threads_per_block;

    //* -----------------------------------------------------
    // printf("\nRunning mid_optimized_1 Sobel kernel...\n");
    

    // cudaFuncSetCacheConfig(sobel_opt_1, cudaFuncCachePreferL1);


    // const int num_threads = 512;
    // dim3 num_block(width / num_threads + 1, height);
    // // printf("num_blocks.x = %d, num_blocks.y = %d\n", num_blocks.x, num_blocks.y);
    // printf("\nRunning optimized Sobel kernel...\n");
    

    // cudaFuncSetCacheConfig(sobel_opt_1, cudaFuncCachePreferL1);
   

    // cudaEventRecord(start);
    // sobel_opt_1 << <num_block, num_threads>>> (dsrc, ddst_opt, height, width, channels);
    // cudaEventRecord(stop);
    // cudaEventSynchronize(stop);
    // cudaEventElapsedTime(&elapsed_time, start, stop);
    // printf("Optimized kernel time: %.3f ms\n", elapsed_time);


    //* -----------------------------------------------------
    printf("\nRunning optimized Sobel kernel...\n");
    

    cudaFuncSetCacheConfig(sobel_opt_2, cudaFuncCachePreferL1);

    dim3 blockDim(BLOCK_DIM_X, BLOCK_DIM_Y);
    dim3 gridDim((width + OUT_TILE_DIM_X - 1) / OUT_TILE_DIM_X, 
             (height + OUT_TILE_DIM_Y - 1) / OUT_TILE_DIM_Y);

    printf("Grid dimensions: %dx%d\n", gridDim.x, gridDim.y);
    printf("Block dimensions: %dx%d\n", blockDim.x, blockDim.y);

    cudaEventRecord(start);
    //! 
	// sobel_2<<<number_of_blocks, threads_per_block>>>(dsrc, ddst_opt, height, width, channels);
    // sobel_opt<<<gridDim, blockDim>>>(dsrc, ddst_opt, height, width, channels);
              
    // Launch kernel with L1 cache preference
    
    // sobel_final<<<grid, block>>>(dsrc, ddst_opt, height, width, channels);

    sobel_opt_2<<<gridDim, blockDim>>>(dsrc, ddst_opt, height, width, channels);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsed_time, start, stop);
    printf("Optimized kernel time: %.3f ms\n", elapsed_time);
    //* ------------------------------------------------------

    // Check for kernel errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return -1;
    }

    // Copy results back to host
    CHECK(cudaMemcpy(dst_orig, ddst_orig, image_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(dst_opt, ddst_opt, image_size, cudaMemcpyDeviceToHost));

    // Validate results
    printf("\nValidating results...\n");
    long total_diff = 0;
    long max_diff = 0;
    const long total_pixels = height * width * channels;
    
    for (long i = 0; i < total_pixels; i++) {
        long diff = abs((long)dst_orig[i] - (long)dst_opt[i]);
        total_diff += diff;
        max_diff = std::max(max_diff, diff);
    }
    
    double avg_diff = (double)total_diff / total_pixels;
    printf("Validation Results:\n");
    printf("Maximum pixel difference: %ld\n", max_diff);
    printf("Average pixel difference: %.2f\n", avg_diff);

    // Write output images
    write_png("output_original.png", dst_orig, height, width, channels);
    write_png("output_optimized.png", dst_opt, height, width, channels);
    printf("\nOutput images written to 'output_original.png' and 'output_optimized.png'\n");

    // Cleanup
    cudaHostUnregister(src);
    free(src);
    free(dst_orig);
    free(dst_opt);
    cudaFree(dsrc);
    cudaFree(ddst_orig);
    cudaFree(ddst_opt);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}
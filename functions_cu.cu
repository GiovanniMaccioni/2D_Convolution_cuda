#include <cuda.h>
#include<cuda_runtime.h>

__global__ void convolution_2D(float* d_img, int rows, int columns, int channels,
                              float* d_kernel, int kernel_width,
                              float* d_new_img) 
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int half_ker_width = kernel_width / 2;
    int half_ker_height = kernel_width / 2;

    if (x < columns && y < rows) {
        for (int c = 0; c < channels; c++) 
        {
            float sum = 0;
            for (int k_i = 0; k_i < kernel_width; k_i++) {
                for (int k_j = 0; k_j < kernel_width; k_j++) {
                    int img_x = x + (k_j - half_ker_width);
                    int img_y = y + (k_i - half_ker_height);
                    
                    if (img_x >= 0 && img_x < columns && img_y >= 0 && img_y < rows) {
                        sum += d_img[(img_y * columns + img_x) * channels + c] * d_kernel[k_i * kernel_width + k_j];
                    }
                }
            }
            d_new_img[(y * columns + x) * channels + c] = min(max((int)sum, 0), 255);
        }
    }
}

__global__ void convolution_2D_shared(float* d_img, int rows, int columns, int channels,
                                     float* d_kernel, int kernel_width,
                                     float* d_new_img)
{

    extern __shared__ float shared_img[];

    int half_ker_width = kernel_width / 2;
    int half_ker_height = kernel_width / 2;

    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    int shared_x = threadIdx.x + half_ker_width;
    int shared_y = threadIdx.y + half_ker_height;

    int shared_width = blockDim.x + 2 * half_ker_width;

    for (int c = 0; c < channels; c++) {

        if (x < columns && y < rows) 
        {
            shared_img[(shared_y * shared_width + shared_x) * channels + c] = d_img[(y * columns + x) * channels + c];
        }

        if (threadIdx.x < half_ker_width)
        {
            int left_x = max(x - half_ker_width, 0);
            shared_img[(shared_y * shared_width + threadIdx.x) * channels + c] = d_img[(y * columns + left_x) * channels + c];
        }
        if (threadIdx.x >= blockDim.x - half_ker_width)
        {
            int right_x = min(x + half_ker_width, columns - 1);
            shared_img[(shared_y * shared_width + (threadIdx.x + 2 * half_ker_width)) * channels + c] = d_img[(y * columns + right_x) * channels + c];
        }
        if (threadIdx.y < half_ker_height) 
        {
            int top_y = max(y - half_ker_height, 0);
            shared_img[(threadIdx.y * shared_width + shared_x) * channels + c] = d_img[(top_y * columns + x) * channels + c];
        }
        if (threadIdx.y >= blockDim.y - half_ker_height)
        {
            int bottom_y = min(y + half_ker_height, rows - 1);
            shared_img[((threadIdx.y + 2 * half_ker_height) * shared_width + shared_x) * channels + c] = d_img[(bottom_y * columns + x) * channels + c];
        }
    }

    __syncthreads();

    if (x < columns && y < rows) 
    {
        for (int c = 0; c < channels; c++)
        {
            float sum = 0;
            for (int k_i = 0; k_i < kernel_width; k_i++) {
                for (int k_j = 0; k_j < kernel_width; k_j++) {
                    sum += shared_img[((shared_y + k_i - half_ker_height) * shared_width + (shared_x + k_j - half_ker_width)) * channels + c] *
                           d_kernel[k_i * kernel_width + k_j];
                }
            }
            d_new_img[(y * columns + x) * channels + c] = min(max((int)sum, 0), 255);
        }
    }
}
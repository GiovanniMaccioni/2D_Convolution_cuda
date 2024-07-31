#include "functions_cu.cu"
#include <opencv2/opencv.hpp>
#include <iostream>
#include "functions.cpp"

#define BLOCK_DIM 16

int main()
{
    // Load the image from file
    std::string image_name;
    std::string image_in;
    std::string image_out;

    image_name = "4K";
    image_in = image_name+".jpg";
    image_out = image_name+"_parallel_result.jpg";

    cv::Mat image = cv::imread(image_in, cv::IMREAD_COLOR);

    // Check if the image was loaded successfully
    if (image.empty())
    {
        std::cerr << "Error: Could not load image!" << std::endl;
        return -1;
    }

    if (!image.isContinuous())
    {
        image = image.clone();
    }

    int rows = image.rows;
    int columns = image.cols;
    int channels = image.channels();

    // Create a multidimensional array to store the image data
    float*** img = new float**[rows];
    for (int i = 0; i < rows; ++i) {
        img[i] = new float*[columns];
        for (int j = 0; j < columns; ++j) {
            img[i][j] = new float[channels];
        }
    }

    // Copy data from cv::Mat to the array
    for (int i = 0; i < rows; ++i)
    {
        for (int j = 0; j < columns; ++j) 
        {
            for (int c = 0; c < channels; ++c)
            {
                img[i][j][c] = (float)image.at<cv::Vec3b>(i, j)[c];
            }
        }
    }

    int kernel_width = 7;
    //int kernel_channels = 3;

    float** kernel = new float*[kernel_width];
    for (int i = 0; i < kernel_width; ++i)
        kernel[i] = new float[kernel_width];

    //gaussian_blur_3x3(kernel, kernel_width);
    //gaussian_blur_5x5(kernel, kernel_width);
    gaussian_blur_7x7(kernel, kernel_width);

    float* flat_kernel = new float[kernel_width*kernel_width];
    flatten_kernel(kernel, kernel_width, flat_kernel);
    

    float* flat_new_img = new float[rows*columns*channels];

    float* flat_image = new float[rows*columns*channels];
    flatten_image(img, rows, columns, channels, flat_image);


    auto start = std::chrono::high_resolution_clock::now();


    // Allocate memory on the GPU
    float* d_img;
    float* d_new_img;

    float* d_kernel;

    cudaMalloc((void**)&d_img, rows*columns*channels*sizeof(float));
    cudaMemcpy(d_img, flat_image, rows*columns*channels*sizeof(float), cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_new_img, rows*columns*channels*sizeof(float));

    cudaMalloc((void**)&d_kernel, kernel_width*kernel_width*sizeof(float));
    cudaMemcpy(d_kernel, flat_kernel, kernel_width*kernel_width*sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Error after kernel launch: %s\n", cudaGetErrorString(err));
    }

    // Block and grid dimensions
    dim3 blockSize(BLOCK_DIM, BLOCK_DIM);
    dim3 gridSize((columns + blockSize.x - 1) / blockSize.x, (rows + blockSize.y - 1) / blockSize.y);


    //Uncomment for Vanilla
    //convolution_2D<<<gridSize, blockSize>>>(d_img, rows, columns, channels, d_kernel, kernel_width, d_new_img);

    //Uncomment for Shared Memory + Tiling
    size_t shared_mem_size = (blockSize.x + 2 * (kernel_width / 2)) * (blockSize.y + 2 * (kernel_width / 2)) * channels * sizeof(float);
    convolution_2D_shared<<<gridSize, blockSize, shared_mem_size>>>(d_img, rows, columns, channels, d_kernel, kernel_width, d_new_img);

    cudaMemcpy(flat_new_img, d_new_img, rows*columns*channels * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_new_img);
    cudaFree(d_kernel);


    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("Work took %ld milliseconds\n", duration.count());

    float*** new_img = new float**[rows];
    for (int i = 0; i < rows; ++i) {
        new_img[i] = new float*[columns];
        for (int j = 0; j < columns; ++j) {
            new_img[i][j] = new float[channels];
        }
    }

    unflatten_image(flat_new_img, rows, columns, channels, new_img);


    // Convert the array back to cv::Mat
    cv::Mat img_out(rows, columns, CV_8UC3);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            for (int c = 0; c < channels; ++c) {
                img_out.at<cv::Vec3b>(i, j)[c] = (unsigned char)(new_img[i][j][c]);
            }
        }
    }


    if (!cv::imwrite(image_out, img_out)) {
        std::cerr << "Error: Could not save the image!" << std::endl;
        return -1;
    }
}



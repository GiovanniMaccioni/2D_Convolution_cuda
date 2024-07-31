#include <opencv2/opencv.hpp>
#include <iostream>
#include "functions.cpp"

int main()
{
    // Load the image from file
    std::string image_name;
    std::string image_in;
    std::string image_out;

    image_name = "FULLHD";
    image_in = image_name+".jpg";
    image_out = image_name+"_sequential_result.jpg";

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
            for (int k = 0; k < channels; ++k)
            {
                img[i][j][k] = (float)image.at<cv::Vec3b>(i, j)[k];
            }
        }
    }

    int kernel_width = 5;
    //int kernel_channels = 3;

    float** kernel = new float*[kernel_width];
    for (int i = 0; i < kernel_width; ++i)
        kernel[i] = new float[kernel_width];
    
    //gaussian_blur_3x3(kernel, kernel_width);
    gaussian_blur_5x5(kernel, kernel_width);
    //gaussian_blur_7x7(kernel, kernel_width);



    float*** new_img = new float**[rows];
    for (int i = 0; i < rows; ++i) {
        new_img[i] = new float*[columns];
        for (int j = 0; j < columns; ++j) {
            new_img[i][j] = new float[channels];
        }
    }


    auto start = std::chrono::high_resolution_clock::now();
    
    //Same Convolutions
    convolution_2D_seq(img, rows, columns, channels, kernel, kernel_width, new_img);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    printf("Work took %ld milliseconds\n", duration.count());


    // Convert the array back to cv::Mat
    cv::Mat reconstructedImage(rows, columns, CV_8UC3);
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            for (int k = 0; k < channels; ++k) {
                reconstructedImage.at<cv::Vec3b>(i, j)[k] = (unsigned char)round(new_img[i][j][k]);
            }
        }
    }


    if (!cv::imwrite(image_out, reconstructedImage)) {
        std::cerr << "Error: Could not save the image!" << std::endl;
        return -1;
    }
}



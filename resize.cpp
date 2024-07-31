#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    // Load the 4K image
    cv::Mat original_image = cv::imread("./4K.jpg");
    if (original_image.empty()) {
        std::cerr << "Error: Could not open or find the image." << std::endl;
        return -1;
    }

    // Define the target size for 720p
    int target_width = 1920;
    int target_height = 1080;

    // Resize the image to 720p
    cv::Mat resized_image;
    cv::resize(original_image, resized_image, cv::Size(target_width, target_height), 0, 0, cv::INTER_LANCZOS4);

    // Save the resized image
    cv::imwrite("./FULLHD.jpg", resized_image);

    return 0;
}
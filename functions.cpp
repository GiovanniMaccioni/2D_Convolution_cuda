
void gaussian_blur_3x3(float** kernel, int kernel_width)
{
    kernel[0][0] = 1./16; kernel[0][1] = 2./16; kernel[0][2] = 1./16;
    kernel[1][0] = 2./16; kernel[1][1] = 4./16; kernel[1][2] = 2./16;
    kernel[2][0] = 1./16; kernel[2][1] = 2./16; kernel[2][2] = 1./16;
}

void gaussian_blur_5x5(float** kernel, int kernel_width)
{
    kernel[0][0] = 1./273; kernel[0][1] = 4./273; kernel[0][2] = 7./273; kernel[0][3] = 4./273; kernel[0][4] = 1./273;
    kernel[1][0] = 4./273; kernel[1][1] = 16./273; kernel[1][2] = 26./273; kernel[1][3] = 16./273; kernel[1][4] = 4./273;
    kernel[2][0] = 7./273; kernel[2][1] = 26./273; kernel[2][2] = 41./273; kernel[2][3] = 26./273; kernel[2][4] = 7./273;
    kernel[3][0] = 4./273; kernel[3][1] = 16./273; kernel[3][2] = 26./273; kernel[3][3] = 16./273; kernel[3][4] = 4./273;
    kernel[4][0] = 1./273; kernel[4][1] = 4./273; kernel[4][2] = 7./273; kernel[4][3] = 4./273; kernel[4][4] = 1./273;
}

void gaussian_blur_7x7(float** kernel, int kernel_width)
{
    kernel[0][0] = 1./1003; kernel[0][1] =  6./1003; kernel[0][2] = 15./1003; kernel[0][3] = 20./1003; kernel[0][4] = 15./1003; kernel[0][5] = 6./1003; kernel[0][6] = 1./1003;
    kernel[1][0] = 6./1003; kernel[1][1] = 36./1003; kernel[1][2] = 90./1003; kernel[1][3] = 120./1003; kernel[1][4] = 90./1003; kernel[1][5] = 36./1003; kernel[1][6] = 6./1003;
    kernel[2][0] = 15./1003; kernel[2][1] = 90./1003; kernel[2][2] = 225./1003; kernel[2][3] = 300./1003; kernel[2][4] = 225./1003; kernel[2][5] = 90./1003; kernel[2][6] = 15./1003;
    kernel[3][0] = 20./1003; kernel[3][1] = 120./1003; kernel[3][2] = 300./1003; kernel[3][3] = 400./1003; kernel[3][4] = 300./1003; kernel[3][5] = 120./1003; kernel[3][6] = 20./1003;
    kernel[4][0] = 15./1003; kernel[4][1] = 90./1003; kernel[4][2] = 225./1003; kernel[4][3] = 300./1003; kernel[4][4] = 225./1003; kernel[4][5] = 90./1003; kernel[4][6] = 15./1003;
    kernel[5][0] = 6./1003; kernel[5][1] = 36./1003; kernel[5][2] = 90./1003; kernel[5][3] = 120./1003; kernel[5][4] = 90./1003; kernel[5][5] = 36./1003; kernel[5][6] = 6./1003;
    kernel[6][0] = 1./1003; kernel[6][1] = 6./1003; kernel[6][2] = 15./1003; kernel[6][3] = 20./1003; kernel[6][4] = 15./1003; kernel[6][5] = 6./1003; kernel[6][6] = 1./1003;
}


void convolution_2D_seq(float*** img, int rows, int columns, int channels,
                    float** kernel, int kernel_width,
                    float*** new_img) 
{
    int k_center_x = kernel_width / 2;
    int k_center_y = kernel_width / 2;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            for (int c = 0; c < channels; ++c) {
                float sum = 0;
                for (int k_i = 0; k_i < kernel_width; ++k_i)
                {
                    for (int k_j = 0; k_j < kernel_width; ++k_j)
                    {
                        int img_x = j + (k_j - k_center_x);
                        int img_y = i + (k_i - k_center_y);

                        if (img_x >= 0 && img_x < columns && img_y >= 0 && img_y < rows) {
                            sum += img[img_y][img_x][c] * kernel[k_i][k_j];
                        }
                    }
                }

                // Ensure the result fits within 0-255
                new_img[i][j][c] = std::min(std::max((int)sum, 0), 255);
            }
        }
    }
}


void flatten_image(float*** image, int rows, int columns, int channels, float* flat_image) 
{
    int index = 0;
    for (int i = 0; i < rows; ++i)
        for (int j = 0; j < columns; ++j)
            for(int c = 0; c < channels; ++c)
            {
                flat_image[index] = image[i][j][c];
                index++;
            }
}

void unflatten_image(float* flat_image, int rows, int columns, int channels, float*** image) 
{
    int index = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < columns; ++j) {
            for (int c = 0; c < channels; ++c) {
                image[i][j][c] = flat_image[index];
                index++;
            }
        }
    }
}

void flatten_kernel(float** kernel, int kernel_width, float* flat_kernel) 
{
    int index = 0;
    for (int i = 0; i < kernel_width; ++i)
        for (int j = 0; j < kernel_width; ++j)
        {
            flat_kernel[index] = kernel[i][j];
            index++;
        }
}
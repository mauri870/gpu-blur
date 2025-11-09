#include "gpu_blur.hip.hpp"

void GPUBlur::Apply(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels)
{
    for (int i = 0; i < width * height * channels; ++i) {
        outputImage[i] = inputImage[i];
    }
}

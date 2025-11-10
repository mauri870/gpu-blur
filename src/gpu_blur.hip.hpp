#pragma once
#include <iostream>
#include <vector>
#include <hip/hip_runtime.h>

#define HIP_CHECK(expression)                  \
{                                              \
    const hipError_t status = expression;      \
    if(status != hipSuccess){                  \
        std::cerr << "HIP error "              \
                  << status << ": "            \
                  << hipGetErrorString(status) \
                  << " at " << __FILE__ << ":" \
                  << __LINE__ << std::endl;    \
    }                                          \
}

std::vector<float> makeGaussianKernel(int radius, float sigma);

__global__ void blurKernel(const unsigned char *dInputImage, unsigned char *dOutputImage, int width, int height, int channels, int radius, const float *dGaussianKernel);

class GPUBlur {
public:
    GPUBlur() {};
    ~GPUBlur() {};

    void Apply(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int radius);
};
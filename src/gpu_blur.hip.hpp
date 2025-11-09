#pragma once
#include <iostream>
#include <hip/hip_runtime.h>

#define HIP_CHECK(op)                                                      \
    do {                                                                    \
        hipError_t e = op;                                                 \
        if (e != hipSuccess) {                                              \
            std::cerr << "HIP error: " << hipGetErrorString(e)              \
                      << " (" << __FILE__ << ":" << __LINE__ << ")"         \
                      << std::endl;                                         \
            std::exit(EXIT_FAILURE);                                        \
        }                                                                   \
    } while (0)

__global__ void blurKernel(const unsigned char *dInputImage, unsigned char *dOutputImage, int width, int height, int channels, int radius);

class GPUBlur {
public:
    GPUBlur() {};
    ~GPUBlur() {};

    void Apply(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels, int radius);
};
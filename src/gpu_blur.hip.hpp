#pragma once
#include <hip/hip_runtime.h>

class GPUBlur {
public:
    GPUBlur() {};
    ~GPUBlur() {};

    void Apply(const unsigned char* inputImage, unsigned char* outputImage, int width, int height, int channels);
private:
};
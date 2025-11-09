#include <hip/hip_runtime.h>
#include "gpu_blur.hip.hpp"

void GPUBlur::Apply(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, int radius)
{
    unsigned char *d_inputImage, *d_outputImage;
    HIP_CHECK(hipMalloc(&d_inputImage, width * height * channels));
    HIP_CHECK(hipMalloc(&d_outputImage, width * height * channels));

    HIP_CHECK(hipMemcpy(d_inputImage, inputImage, width * height * channels, hipMemcpyHostToDevice));

    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    hipLaunchKernelGGL(
        blurKernel,
        gridSize,
        blockSize,
        0,
        0,
        d_inputImage, d_outputImage, width, height, channels, radius
    );

    HIP_CHECK(hipMemcpy(outputImage, d_outputImage, width * height * channels, hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_inputImage));
    HIP_CHECK(hipFree(d_outputImage));
}

__global__ void blurKernel(const unsigned char *dInputImage, unsigned char *dOutputImage, int width, int height, int channels, int radius)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    float3 sum;
    int count = 0;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x +dx, 0), width- 1);
            int ny = min(max(y + dy, 0), height - 1);
            int idx = (ny * width + nx) * channels;
            sum.x += dInputImage[idx + 0];
            sum.y += dInputImage[idx + 1];
            sum.z += dInputImage[idx + 2];
            count++;
        }
    }

    int pixelIndex = (y * width + x) * channels;
    dOutputImage[pixelIndex+0] = sum.x / count;
    dOutputImage[pixelIndex+1] = sum.y / count;
    dOutputImage[pixelIndex+2] = sum.z / count;
}

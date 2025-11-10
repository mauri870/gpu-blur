#include <vector>
#include <cmath>
#include <hip/hip_runtime.h>
#include "gpu_blur.hip.hpp"

void GPUBlur::Apply(const unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels, int radius)
{
    int sigma = radius / 2.0f;
    std::vector<float> hKernel = makeGaussianKernel(radius, sigma);

    float *dKernel;
    size_t kernelSize = hKernel.size() * sizeof(float);
    HIP_CHECK(hipMalloc(&dKernel, kernelSize));
    HIP_CHECK(hipMemcpy(dKernel, hKernel.data(), kernelSize, hipMemcpyHostToDevice));

    unsigned char *dInputImage, *dOutputImage;
    HIP_CHECK(hipMalloc(&dInputImage, width * height * channels));
    HIP_CHECK(hipMalloc(&dOutputImage, width * height * channels));
    HIP_CHECK(hipMemcpy(dInputImage, inputImage, width * height * channels, hipMemcpyHostToDevice));

    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, 
                  (height + blockSize.y - 1) / blockSize.y);
    
    hipLaunchKernelGGL(
        blurKernel,
        gridSize,
        blockSize,
        0,
        0,
        dInputImage, dOutputImage, width, height, channels, radius, dKernel
    );
    HIP_CHECK(hipGetLastError());
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipMemcpy(outputImage, dOutputImage, width * height * channels, hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dInputImage));
    HIP_CHECK(hipFree(dOutputImage));
    HIP_CHECK(hipFree(dKernel));
}

__global__ void blurKernel(const unsigned char *dInputImage, unsigned char *dOutputImage, int width, int height, int channels, int radius, const float *dGaussianKernel)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height) return;

    const int kernelSize = 2 * radius + 1;
    float3 sum = {0.0f, 0.0f, 0.0f};
    float weightSum = 0.0f;

    const int basePixelIndex = (y * width + x) * channels;

    for (int dy = -radius; dy <= radius; ++dy) {
        for (int dx = -radius; dx <= radius; ++dx) {
            int nx = min(max(x + dx, 0), width - 1);
            int ny = min(max(y + dy, 0), height - 1);
            int idx = (ny * width + nx) * channels;

            int kernelIndex = (dy + radius) * kernelSize + (dx + radius);
            float weight = dGaussianKernel[kernelIndex];

            sum.x += weight * dInputImage[idx];
            if (channels > 1) sum.y += weight * dInputImage[idx + 1];
            if (channels > 2) sum.z += weight * dInputImage[idx + 2];

            weightSum += weight;
        }
    }

    // Normalize so brightness is preserved
    dOutputImage[basePixelIndex]     = (unsigned char)(sum.x / weightSum);
    if (channels > 1)
        dOutputImage[basePixelIndex + 1] = (unsigned char)(sum.y / weightSum);
    if (channels > 2)
        dOutputImage[basePixelIndex + 2] = (unsigned char)(sum.z / weightSum);
}

std::vector<float> makeGaussianKernel(int radius, float sigma)
{
    int size = 2 * radius + 1;
    std::vector<float> kernel(size * size);
    float sum = 0.0f;
    float invTwoSigma2 = 1.0f / (2.0f * sigma * sigma);

    for (int y = -radius; y <= radius; ++y) {
        for (int x = -radius; x <= radius; ++x) {
            float weight = expf(-(x * x + y * y) * invTwoSigma2);
            kernel[(y + radius) * size + (x + radius)] = weight;
            sum += weight;
        }
    }

    // Normalize to 1.0
    for (float &w : kernel)
        w /= sum;

    return kernel;
}

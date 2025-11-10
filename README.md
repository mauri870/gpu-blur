# gpu-blur

This is a tool to apply a Gaussian blur filter to images using the GPU. This is the equivalent of ImageMagick's `convert -blur`, but runs much faster, especially with larger image sizes.

The GPU kernel is written in [HIP](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/what_is_hip.html) for portability, so it supports both AMD and NVIDIA GPUs.

Accepts the same input formats supported by `stb_image`, and outputs PNG only.

## Requirements

- CMake 3.x
- C++17 compiler
- [HIP Toolkit](https://rocm.docs.amd.com/projects/HIP/en/docs-develop/install/install.html)

## Installation

```bash
git clone --recursive https://github.com/mauri870/gpu-blur.git
cd gpu-blur

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j $(nproc) --config Release
```

## Usage

```bash
./build/gpu-blur <input_image> <output_image.png> [blur_intensity]
```

- `input_image`: Path to the input image file
- `output_image.png`: Path for the output image (PNG format)
- `blur_intensity` (optional): Blur radius in pixels (default: 5)

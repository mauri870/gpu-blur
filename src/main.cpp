#include <iostream>
#include <fstream>
#include <string>

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"

#include "gpu_blur.hip.hpp"

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image.png> <blur_intensity>\n";
        return 1;
    }

    std::string input_image = argv[1];
    std::string output_image = argv[2];
    std::string radius_str = (argc >=4) ? argv[3] : "5";
    int radius = std::stoi(radius_str);
    if (radius < 1) {
        std::cerr << "Blur intensity must be at least 1.\n";
        return 1;
    }

    int w, h, comp;
    unsigned char *img = stbi_load(input_image.c_str(), &w, &h, &comp, 3);
    if (img == nullptr) {
        std::cerr << "Error loading image: " << stbi_failure_reason() << "\n";
        return 1;
    }


    unsigned char* out = new unsigned char[w * h * comp];

    GPUBlur kernel;
    kernel.Apply(img, out, w, h, comp, radius);


    if (!stbi_write_png(output_image.c_str(), w, h, comp, out, w * comp)) {
        std::cerr << "Error writing image: " << stbi_failure_reason() << "\n";
        stbi_image_free(img);
        return 1;
    }

    stbi_image_free(img);
    delete[] out;
    return 0;
}

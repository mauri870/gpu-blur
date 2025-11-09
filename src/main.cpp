#include <iostream>
#include <fstream>
#include <string>

#include "gpu_blur.hip.hpp"

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " <input_image> <output_image.png>\n";
        return 1;
    }

    std::string input_image = argv[1];
    std::string output_image = argv[2];

    int w, h, comp;
    int radius = 5;
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

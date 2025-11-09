cmake_minimum_required(VERSION 3.15)
set(CMAKE_TOOLCHAIN_FILE external/vcpkg/scripts/buildsystems/vcpkg.cmake CACHE STRING "")

project(gpu-blur LANGUAGES CXX HIP)

find_package(hip REQUIRED)
find_package(Stb REQUIRED)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_HIP_STANDARD 17)

file (GLOB_RECURSE Sources CONFIGURE_DEPENDS "src/*.cpp")
file (GLOB_RECURSE Headers CONFIGURE_DEPENDS "src/*.hpp")
file (GLOB_RECURSE HipSources CONFIGURE_DEPENDS "src/*.hip.cpp")
file (GLOB_RECURSE HipHeaders CONFIGURE_DEPENDS "src/*.hip.hpp")

add_executable(${PROJECT_NAME} ${Sources} ${Headers} ${HipSources} ${HipHeaders})
target_include_directories(${PROJECT_NAME} PRIVATE ${Stb_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} hip::host)

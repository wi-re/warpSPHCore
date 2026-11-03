cmake_minimum_required(VERSION 3.21)
project(warpSPHCoreExample LANGUAGES CXX)
# Locate the standalone installation; its exported target supplies headers and C++17.
find_package(warpSPHCore CONFIG REQUIRED)
add_executable(sph_example main.cpp)
# No Python, Torch or Warp package is needed by this application.
target_link_libraries(sph_example PRIVATE warpSPHCore::core)

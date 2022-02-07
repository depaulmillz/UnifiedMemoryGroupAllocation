# UnifiedMemoryGroupAllocation
This is a memory allocator for CUDA Unified Memory.

## Building

Requires CMake >= 3.18, conan, and a CUDA version in 11.0 to 11.6.

It is easiest to get conan through pip.

Build and package by:
```
mkdir build
cd build
conan install --build missing ..
conan create ..
```

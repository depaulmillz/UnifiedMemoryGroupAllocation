# Unified Memory Group Allocation
This is a memory allocator for CUDA Unified Memory.

### OS and Hardware

Code is tested and developed on Ubuntu although it should be supported on any Linux distro that supports CUDA.
The functionality of Unified Memory (UM) used is supported solely on Linux.
The code works for Pascal or later, although Volta, sm\_70, or later is ideal. 

### Building

Requires CMake >= 3.18, conan, and a CUDA version in 11.0 to 11.6.

It is easiest to get conan through pip.

Build and package by:
```
mkdir build
cd build
conan install --build missing ..
conan create ..
```

### Code Organization

- include/GroupAllocator contains the include files
- groupallocator will include everything needed to utilize the code
- Allocator.h includes a basic global allocator to allocate UM with groups
- GroupAllocator.h contains the group allocator class to allocate groups
- Helper.h contains assertion methods for making sure errors are handled
- InPageAllocator.h allocator for allocating in pages
- ListAllocator.h thread safe list allocator that allocates with the list seperate from the memory used
- MultiPageAllocator.h allocator for allocating across multiple pages
 

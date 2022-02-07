cmake_minimum_required(VERSION 3.18)

project(UnifiedMemoryGroupAllocation LANGUAGES CXX)

option(USING_CONAN "Set if building through the conan script" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(cmake/config.cmake)

find_package(CUDAToolkit REQUIRED)

check_cuda_version(11.0 11.6)

set_consistent_standard(11)
setup_conan()

foreach(LIB ${CONAN_LIBS})
    message(STATUS "Using library ${LIB}")
endforeach()

if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "70;75")
    message(STATUS "Setting CUDA ARCHITECTURES to ${CMAKE_CUDA_ARCHITECTURES}")
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CUDA_FLAGS " -G -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "Debug")

if (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    string(APPEND CMAKE_CUDA_FLAGS " -lineinfo -pg")
endif (CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")

string(APPEND CMAKE_CUDA_FLAGS " -Xptxas=-v")
string(APPEND CMAKE_CUDA_FLAGS " -Wreorder")

add_compile_options("$<$<CONFIG:DEBUG>:-DDEBUG>")

foreach(ARCH "${CMAKE_CUDA_ARCHITECTURES}")
    if("${ARCH}" MATCHES "^5.$")
        add_compile_options("-DDISABLE_PREFETCH")
        break()
    endif()
endforeach()

enable_testing()

add_library(allocator INTERFACE)
target_include_directories(allocator INTERFACE 
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>)
target_link_libraries(allocator INTERFACE CUDA::cudart)

add_subdirectory(test)

## Installation

install(TARGETS allocator
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include)

install(DIRECTORY ${PROJECT_SOURCE_DIR}/include/ DESTINATION include)

set(CPACK_GENERATOR "ZIP")
set(CPACK_PACKAGE_NAME "UMGroupAllocation")
include(CPack)

cmake_minimum_required(VERSION 3.18)

project(UnifiedMemoryGroupAllocation LANGUAGES CUDA CXX)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Release")
endif()

include(cmake/config.cmake)

check_cuda_version(11.0 11.5)

set_consistent_standard(11)
setup_conan()

foreach(LIB in ${CONAN_LIBS})
    message(STATUS ${LIB})
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
enable_testing()

add_library(allocator INTERFACE)
target_include_directories(allocator INTERFACE include)

add_subdirectory(test)

set(TARGETLIST allocator)

install(TARGETS ${TARGETLIST}
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
        RUNTIME DESTINATION bin
        INCLUDES DESTINATION include
        )

set(CPACK_PACKAGE_VENDOR "dePaul Miller et. al.")
SET(CPACK_GENERATOR "DEB")
SET(CPACK_DEBIAN_PACKAGE_MAINTAINER "dePaul Miller")
set(CPACK_SOURCE_GENERATOR "TGZ;ZIP")
set(CPACK_SOURCE_IGNORE_FILES
        /.git
        /dist
        /.*build.*
        /\\\\.DS_Store
        )
include(CPack)

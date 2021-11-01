function(CHECK_CUDA_VERSION CUDA_MIN_VERSION CUDA_MAX_VERSION)
    if(NOT CUDA_VERSION)
        set(CUDA_VERSION ${CMAKE_CUDA_COMPILER_VERSION} PARENT_SCOPE)
    endif()
    if(NOT CUDA_TOOLKIT_ROOT_DIR)
        get_filename_component(CUDA_TOOLKIT_ROOT_DIR "${CMAKE_CUDA_COMPILER}/../.." ABSOLUTE)
    endif()
    
    if(CUDA_VERSION VERSION_LESS CUDA_MIN_VERSION AND CUDA_VERSION VERSION_GREATER CUDA_MAX_VERSION)
        message(FATAL_ERROR "CUDA ${CUDA_MIN_VERSION}-${CUDA_MAX_VERSION} required, found ${CUDA_VERSION}")
    endif()
endfunction()

macro(SET_CONSISTENT_STANDARD STANDARD)
    set(CMAKE_CXX_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD ${STANDARD})
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
endmacro()

macro(SETUP_CONAN)
    message(STATUS "Running conan install -s build_type=${CMAKE_BUILD_TYPE} ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing")
    message(STATUS "USING ${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION} standard ${CMAKE_CXX_STANDARD}")

    if(NOT USING_CONAN)
        execute_process(
            COMMAND conan install -s compiler.libcxx=libstdc++11 -s build_type=${CMAKE_BUILD_TYPE} ${CMAKE_SOURCE_DIR} -if ${CMAKE_BINARY_DIR} --build=missing
            RESULT_VARIABLE conan_code)
        if(NOT "${conan_code}" STREQUAL "0")
            message(FATAL_ERROR "Conan failed ${conan_code}")
        endif()
    endif()
    include(${CMAKE_BINARY_DIR}/conanbuildinfo.cmake)
    conan_basic_setup()
endmacro()

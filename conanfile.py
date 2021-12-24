from conans import ConanFile, CMake

class UnifiedMemoryGroupAllocationConan(ConanFile):
    name = "unifiedmemorygroupallocation"
    version = "1.1"
    author = "dePaul Miller"
    url = "https://github.com/depaulmillz/UnifiedMemoryGroupAllocation"
    license = "MIT"
    settings="os", "compiler", "build_type", "arch"
    build_requires="gtest/1.10.0"
    generators="cmake"
    description = """The Unified Memory Group Allocator was designed in the paper
    "KVCG: A Heterogeneous Key-Value Store for Skewed Workloads" by dePaul Miller, 
    Jacob Nelson, Ahmed Hassan, and Roberto Palmieri." It allows for allocating
    Unified Memory with extra metadata to limit thrashing."""
    topic = ("unified memory", "gpu programming", "allocator")
 
    options = {"cuda_arch" : "ANY", "cuda_compiler" : "ANY"}
    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "test/*", "LICENSE"
    
    def configure(self):
        if self.options.cuda_arch == None:
            self.options.cuda_arch = '70;75'
        if self.options.cuda_compiler == None:
            self.options.cuda_compiler = "nvcc"

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["USING_CONAN"] = "ON"
        cmake.definitions["CMAKE_CUDA_COMPILER"] = str(self.options.cuda_compiler)
        cmake.definitions["CMAKE_EXPORT_COMPILE_COMMANDS"] = "ON"
        cmake.definitions["CMAKE_CUDA_ARCHITECTURES"] = self.options.cuda_arch
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def package_id(self):
        self.info.header_only()

    def package_info(self):
        self.cpp_info.names["cmake_find_package"] = "UnifiedMemoryGroupAllocation"
        self.cpp_info.names["cmake_find_package_multi"] = "UnifiedMemoryGroupAllocation"

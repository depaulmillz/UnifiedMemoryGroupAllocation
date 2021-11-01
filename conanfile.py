from conans import ConanFile, CMake

class UnifiedMemoryGroupAllocationConan(ConanFile):
    name = "UMGroupAllocation"
    version = "1.0"
    settings="os", "compiler", "build_type", "arch"
    requires="gtest/1.10.0"
    generators="cmake"
    
    exports_sources = "CMakeLists.txt", "cmake/*", "include/*", "test/*"
    

    def _configure_cmake(self):
        cmake = CMake(self)
        cmake.definitions["USING_CONAN"] = "ON"
        cmake.configure()
        return cmake

    def build(self):
        cmake = self._configure_cmake()
        cmake.build()
        cmake.test()

    def package(self):
        cmake = self._configure_cmake()
        cmake.install()

    def package_info(self):
        self.cpp_info.libs = ["UMGroupAllocation"]
        self.info.header_only()

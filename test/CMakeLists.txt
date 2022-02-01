add_executable(allocatortest allocatortest.cc)
target_link_libraries(allocatortest PRIVATE ${CONAN_LIBS} allocator)

add_executable(asanAllocatortest allocatortest.cc)
target_link_libraries(asanAllocatortest PRIVATE asan ${CONAN_LIBS} allocator)
target_compile_options(asanAllocatortest PRIVATE -fsanitize=address)

add_test(NAME allocatortest COMMAND allocatortest) 

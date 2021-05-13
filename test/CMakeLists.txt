
add_executable(allocatortest allocatortest.cu)
target_link_libraries(allocatortest gtest gmock gtest_main allocator)

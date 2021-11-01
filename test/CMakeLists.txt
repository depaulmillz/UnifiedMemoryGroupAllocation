add_executable(allocatortest allocatortest.cu)
target_link_libraries(allocatortest PRIVATE ${CONAN_LIBS} allocator)

add_test(NAME allocatortest COMMAND allocatortest) 

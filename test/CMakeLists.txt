add_executable(allocatortest allocatortest.cc)
target_link_libraries(allocatortest PRIVATE ${CONAN_LIBS} allocator)

add_test(NAME allocatortest COMMAND allocatortest) 

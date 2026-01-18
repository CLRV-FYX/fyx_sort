cmake_minimum_required(VERSION 3.15)
project(fyx_sort VERSION 2.0.0 LANGUAGES CXX)

# 设置C++标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# 编译选项
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -Wpedantic -Werror)
    if(FYX_ENABLE_COVERAGE)
        add_compile_options(--coverage)
        add_link_options(--coverage)
    endif()
endif()

# 主库（头文件库）
add_library(fyx_sort INTERFACE)
target_include_directories(fyx_sort INTERFACE 
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<INSTALL_INTERFACE:include>
)

# 测试程序
add_executable(test_fyx)
target_sources(test_fyx PRIVATE 
    # 创建一个简单的测试源文件
    ${CMAKE_CURRENT_SOURCE_DIR}/test/test_main.cpp
)
target_compile_definitions(test_fyx PRIVATE FYX_MAIN)
target_link_libraries(test_fyx PRIVATE fyx_sort)

# 基准测试程序
add_executable(bench_fyx)
target_sources(bench_fyx PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/bench/bench_main.cpp
)
target_compile_definitions(bench_fyx PRIVATE FYX_MAIN)
target_link_libraries(bench_fyx PRIVATE fyx_sort)

# 安装规则
install(TARGETS fyx_sort EXPORT fyx_sort-targets)
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} DESTINATION include
    FILES_MATCHING PATTERN "*.hpp"
)
install(EXPORT fyx_sort-targets
    FILE fyx_sort-config.cmake
    DESTINATION lib/cmake/fyx_sort
)

# 测试
enable_testing()
add_test(NAME fyx_sort_test COMMAND test_fyx)

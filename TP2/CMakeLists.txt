cmake_minimum_required(VERSION 3.0)

project(tp2)
enable_testing()

option(BUILD_TEST "Builds a suite of tests")

set(CMAKE_CXX_FLAGS "-ggdb -std=c++2a -ldl")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif ()

macro(add_files name)
    file(GLOB ${name} src/${name}/*.h src/${name}/*.cpp src/${name}/*.c)
    include_directories(src/${name})
endmacro()

add_files(power_iteration)
include_directories(src)
include_directories(eigen)
add_executable(tp2 src/main.cpp ${power_iteration})

if (UNITY) 
    message("-- Unity build - enabled")
    set_target_properties(tp2 PROPERTIES UNITY_BUILD ON)
else() 
    message("-- Unity build - disabled")
endif()

if (BUILD_TESTS)
    macro(add_custom_test name file)
        string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" bare_name ${file})
        add_executable(${name} ${file} ${ARGN})
        include_directories(src)
        add_test(NAME ${name} COMMAND ${name}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${name} ${LIBRARIES} "-lpthread")
    endmacro()

    add_custom_test(test_eigen tests/test_eigen.cpp ${power_iteration})
endif()


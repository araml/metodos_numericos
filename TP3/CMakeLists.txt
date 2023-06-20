cmake_minimum_required(VERSION 3.0)

project(tp3)
enable_testing()

option(BUILD_TEST "Builds a suite of tests")

set(CMAKE_CXX_FLAGS "-O2 -ggdb -std=c++2a -fopenmp")

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
   add_compile_options (-fdiagnostics-color=always)
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
   add_compile_options (-fcolor-diagnostics)
endif ()

macro(add_files name)
    file(GLOB ${name} src/${name}/*.h src/${name}/*.cpp src/${name}/*.c)
    include_directories(src/${name})
endmacro()

add_files(metodos_iterativos)
include_directories(src)
include_directories(eigen)
include_directories(doctest)
add_subdirectory(pybind11)
add_subdirectory(doctest)

if (BUILD_TESTS)
    macro(add_custom_test name file)
        string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" bare_name ${file})
        add_executable(${name} ${file} ${ARGN})
        include_directories(src)
        include_directories(tests)
        add_test(NAME ${name} COMMAND ${name}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${name} ${LIBRARIES} "-lpthread" doctest)
    endmacro()

    # get_filename_component(TEST_FILE_PATH tests/test_data ABSOLUTE)
    # configure_file(tests/config_tests.h.in ../tests/config_tests.h)

    add_custom_test(test_jacobi tests/test_jacobi.cpp ${metodos_iterativos})
    add_custom_test(test_gs tests/test_gs.cpp ${metodos_iterativos})
    
    # Add python tests
    # Uncomment when we have python tests..
    #add_test(NAME python-tests COMMAND 
    #    ${PYTHON_EXECUTABLE} -m unittest discover "${CMAKE_CURRENT_SOURCE_DIR}/py_stuff")
endif()


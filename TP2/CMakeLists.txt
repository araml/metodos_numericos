cmake_minimum_required(VERSION 3.0)

project(tp2)
enable_testing()

option(BUILD_TEST "Builds a suite of tests")

set(CMAKE_CXX_FLAGS "-O2 -ggdb -std=c++2a")

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
add_subdirectory(pybind11)
pybind11_add_module(deflate src/main.cpp ${power_iteration})

get_filename_component(FACES_PATH caras ABSOLUTE)
get_filename_component(MATRICES_PATH matrices ABSOLUTE)
get_filename_component(FIGURES_PATH figures ABSOLUTE)
get_filename_component(FIGURES_EXPERIMENTS_PATH "caras_experimentos" ABSOLUTE)
configure_file(py_stuff/data_paths.in.py ../py_stuff/data_paths.py)

#TODO (create power iteration as a lib that links the tests/python lib)
#if (UNITY) 
#    message("-- Unity build - enabled")
#    set_target_properties(tp2 PROPERTIES UNITY_BUILD ON)
#else() 
#    message("-- Unity build - disabled")
#endif()

if (BUILD_TESTS)
    macro(add_custom_test name file)
        string(REGEX REPLACE "(^.*/|\\.[^.]*$)" "" bare_name ${file})
        add_executable(${name} ${file} ${ARGN})
        include_directories(src)
        include_directories(tests)
        add_test(NAME ${name} COMMAND ${name}
                 WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR})
        target_link_libraries(${name} ${LIBRARIES} "-lpthread")
    endmacro()

    get_filename_component(TEST_FILE_PATH tests/test_data ABSOLUTE)
    configure_file(tests/config_tests.h.in ../tests/config_tests.h)

    add_custom_test(test_eigen tests/test_eigen.cpp ${power_iteration})
    #add_custom_test(test_read_matrix tests/test_read_matrix.cpp ${power_iteration})
    
    # Add python tests
    add_test(NAME python-tests COMMAND 
        ${PYTHON_EXECUTABLE} -m unittest discover "${CMAKE_CURRENT_SOURCE_DIR}/py_stuff")
endif()


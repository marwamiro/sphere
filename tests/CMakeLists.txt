
find_package(PythonInterp REQUIRED)

add_test(io_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/io_tests.py)

add_test(porosity_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/porosity_tests.py)

add_test(memory_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/memcheck_tests.py)

add_test(bond_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/bond_tests.py)

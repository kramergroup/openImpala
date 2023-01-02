cmake_minimum_required(VERSION 3.13)


set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(CMAKE_FORTRAN_COMPILER ifort)

add_compile_options($<$<COMPILE_LANG_AND_ID:CXX,Intel>:-diag-disable=10441>)
add_compile_options($<$<COMPILE_LANG_AND_ID:C,Intel>:-diag-disable=10441>)
add_link_options($<$<COMPILE_LANG_AND_ID:CXX,Intel>:-diag-disable=10441>)
add_link_options($<$<COMPILE_LANG_AND_ID:C,Intel>:-diag-disable=10441>)

enable_language(Fortran)

project(openImpala)

set(CMAKE_PREFIX_PATH ${HYPRE_DIR} ${CMAKE_PREFIX_PATH})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

include(CTest)

# Help IDEs to find libraries and includes etc. 
# See: https://stackoverflow.com/questions/36122358/visual-studio-code-c-include-path
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Properties 
add_subdirectory(src/props)

# Filter
add_subdirectory(src/filter)
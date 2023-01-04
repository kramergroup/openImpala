cmake_minimum_required(VERSION 3.5)

# Find python and Boost - both are required dependencies
find_package(PythonLibs 3.9 REQUIRED)
find_package(Boost 1.81.0 COMPONENTS python REQUIRED)

# Without this, any build libraries automatically have names "lib{x}.so"
set(CMAKE_SHARED_MODULE_PREFIX "")

# Find other required dependencies
find_package(AMReX REQUIRED)
find_package(TIFF)

# Add a shared module - modules are intended to be imported at runtime.
# - This is where you add the source files
add_library(pyoimp MODULE pyoimp.cpp)

target_compile_options(pyoimp PUBLIC -fPIC)

# Set up the libraries and header search paths for this target
target_link_libraries(pyoimp PRIVATE ${Boost_LIBRARIES} ${PYTHON_LIBRARIES})
target_link_libraries(pyoimp PRIVATE AMReX::amrex )
target_link_libraries(pyoimp PRIVATE tiff)

target_include_directories(pyoimp PRIVATE ${PYTHON_INCLUDE_DIRS})


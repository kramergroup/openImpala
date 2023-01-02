find_package(AMReX REQUIRED)

find_package(HYPRE REQUIRED)
find_package(TIFF REQUIRED)

# Tortuosity Test
file(GLOB TORTUOSITY_SOURCES "Tortuosity*.cpp" "Tortuosity*.F90")
add_executable(testTortuosity tTortuosity.cpp ${TORTUOSITY_SOURCES} VolumeFraction.cpp ../io/TiffReader.cpp) 
target_link_libraries(testTortuosity PUBLIC AMReX::amrex )
target_link_libraries(testTortuosity PUBLIC HYPRE )
target_link_libraries(testTortuosity PUBLIC tiff)

add_test(TortuosityTest mpirun -np 4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testTortuosity ${CMAKE_SOURCE_DIR}/data/SampleData_2Phase.tif)

# Volume Fraction Test
add_executable(testVolumeFraction tVolumeFraction.cpp VolumeFraction.cpp ../io/TiffReader.cpp) 
target_link_libraries(testVolumeFraction PUBLIC AMReX::amrex )
target_link_libraries(testVolumeFraction PUBLIC HYPRE )
target_link_libraries(testVolumeFraction PUBLIC tiff)

add_test(VolumeFractionTest mpirun -np 4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testVolumeFraction ${CMAKE_SOURCE_DIR}/data/SampleData_2Phase.tif)

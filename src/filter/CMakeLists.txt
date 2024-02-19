find_package(AMReX REQUIRED)

find_package(TIFF)

# BoxCrop Test
add_executable(testBoxCrop tBoxCrop.cpp  BoxCrop.cpp Threshold.cpp ../io/TiffReader.cpp) 
target_link_libraries(testBoxCrop PUBLIC AMReX::amrex )
target_link_libraries(testBoxCrop PUBLIC tiff)

add_test(BoxCropTest mpirun -np 4 ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/testBoxCrop ${CMAKE_SOURCE_DIR}/data/SampleData_2Phase.tif ${CMAKE_SOURCE_DIR}/data/CropData_2Phase)

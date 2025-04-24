# -*- coding: utf-8 -*-
"""
Generates a 3D homogeneous structure image (100x100x100) using
porespy.generators.blobs. Saves the output data using the base name
"SampleData_2Phase" in various formats, including a 1-bit TIFF stack
and a uint8 raw binary file.

Current Date/Time: Thursday, April 24, 2025 at 10:45:10 AM BST

Key Steps:
1. Define parameters (3D image shape, target porosity, blobiness).
2. Generate the 3D image using porespy.generators.blobs.
3. Calculate and print the actual porosity.
4. Visualize a 2D slice (middle Z-plane).
5. Save the data in various formats using the specified base name:
   - Full 3D image as VTK.
   - Full 3D image as HDF5 + XDMF.
   - The 2D slice as a standard 8-bit TIFF file (using imageio).
   - The full 3D image as a 1-bit, uncompressed multi-page TIFF (stack) (using tifffile).
   - The full 3D image as a flat uint8 raw binary file (.raw). <--- Added
   - The 2D slice as a separate HDF5 file.

Libraries Used:
- numpy: For array manipulation.
- porespy: For generation, metrics, and VTK saving.
- matplotlib: For visualization (2D plot).
- h5py: For saving HDF5 files.
- imageio: For saving 2D TIFF slice.
- tifffile: For saving 3D TIFF stack with specific options.
- time: To time the generation.
- platform: To show Python version.
"""

import numpy as np
import porespy as ps
import matplotlib.pyplot as plt
import h5py
import imageio
import tifffile # For saving 1-bit TIFF stack
import time
import platform # To show Python version

# Print Python and PoreSpy version for debugging
print(f"Running on Python {platform.python_version()}")
try:
    print(f"Using PoreSpy version: {ps.__version__}")
except AttributeError:
    print("Could not determine PoreSpy version.")
# (Optional: check tifffile version if needed)

# 1. Define Parameters
# --------------------
final_shape = [100, 100, 100] # (X, Y, Z) dimensions
gen_shape = final_shape
porosity_target = 0.40
blobiness_level = 1.2 # Lower value aims for larger features

print(f"Generating structure in {gen_shape[0]}x{gen_shape[1]}x{gen_shape[2]} domain using ps.generators.blobs.")
print(f"Target Porosity (Void=False): {porosity_target:.2f}")
print(f"Blobiness level: {blobiness_level}")


# 2. Generate Structure using porespy.generators.blobs
# -----------------------------------------------------
start_time = time.time()
im_3d = ps.generators.blobs(shape=gen_shape,
                             porosity=porosity_target,
                             blobiness=blobiness_level)
end_time = time.time()
generation_time = end_time - start_time
print(f"\nFinished generation in {generation_time:.2f} seconds.")

if im_3d.dtype != bool:
    print(f"Warning: Generated image type is {im_3d.dtype}, converting to boolean.")
    im_3d = im_3d.astype(bool)

# 3. Calculate Actual Porosity
# ----------------------------
actual_porosity = ps.metrics.porosity(im_3d)
actual_solid_fraction = 1 - actual_porosity
print(f"Actual Porosity (Void=False): {actual_porosity:.3f}")
print(f"Actual Solid Fraction (Solid=True): {actual_solid_fraction:.3f}")


# 4. Visualize the Result (2D Slice Plot only)
# --------------------------------------------
slice_index_z = gen_shape[2] // 2
slice_to_plot = im_3d[:, :, slice_index_z]

print(f"\nPreparing 2D plot of Z-slice {slice_index_z}...")
fig_2d, ax_2d = plt.subplots(1, 1, figsize=(6, 6))
ax_2d.imshow(slice_to_plot.T, origin='lower', cmap='gray', interpolation='none')
ax_2d.set_xlabel('X axis')
ax_2d.set_ylabel('Y axis')
ax_2d.set_title(f'Middle Z-Slice (Z={slice_index_z}) - Blobiness={blobiness_level}')
plt.show()


# 5. Save the Images
# ------------------
base_filename = "SampleData_2Phase"

# Define specific filenames using the base name
vtk_filename = f"{base_filename}_3d.vtk"
tiff_slice_filename = f"{base_filename}_slice_z{slice_index_z}_8bit.tif"
tiff_stack_filename = f"{base_filename}_stack_3d_1bit.tif"
hdf5_filename_3d = f"{base_filename}_3d.hdf5"
xmf_filename = f"{base_filename}_3d.xmf"
hdf5_filename_2d = f"{base_filename}_slice_z{slice_index_z}.hdf5"
raw_filename = f"{base_filename}_stack_3d_uint8.raw" # <-- Added RAW filename
hdf5_dataset_name = "image"

print(f"\nSaving output files using base name: '{base_filename}'...")

# --- Save full 3D image as VTK ---
# (Saves solid=1, void=0 as uint8)
try:
    ps.io.to_vtk(im_3d.astype(np.uint8), filename=vtk_filename)
    print(f"Successfully saved 3D VTK image to: {vtk_filename}")
except AttributeError:
    print(f"Skipping VTK save: ps.io.to_vtk not found or PoreSpy not fully installed.")
except Exception as e:
    print(f"Error saving VTK image: {e}")

# --- Save full 3D image to HDF5 and generate corresponding XDMF file ---
# (Saves solid=1, void=0 as uint8)
# (HDF5 saving code remains the same)
try:
    with h5py.File(hdf5_filename_3d, 'w') as f:
        im_3d_to_save = im_3d.astype(np.uint8)
        f.create_dataset(hdf5_dataset_name, data=im_3d_to_save)
    print(f"Successfully saved 3D HDF5 file to: {hdf5_filename_3d} (Dataset: '{hdf5_dataset_name}')")
    # Generate XDMF (code remains the same, uses hdf5_filename_3d)
    numpy_shape = im_3d_to_save.shape
    size_X, size_Y, size_Z = numpy_shape[0], numpy_shape[1], numpy_shape[2]
    if im_3d_to_save.dtype == np.uint8: xdmf_type, xdmf_precision = "UChar", "1"
    else: print(f"Warning: Unexpected dtype {im_3d_to_save.dtype} for XDMF. Using UChar."); xdmf_type, xdmf_precision = "UChar", "1"
    xdmf_content = f"""\
<?xml version="1.0" ?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0">
  <Domain>
    <Grid Name="StructureGrid" GridType="Uniform">
      <Topology TopologyType="3DCoRectMesh" Dimensions="{size_Z} {size_Y} {size_X}"/>
      <Geometry GeometryType="ORIGIN_DXDYDZ">
        <DataItem Name="Origin" Format="XML" Dimensions="3">0 0 0</DataItem>
        <DataItem Name="Spacing" Format="XML" Dimensions="3">1 1 1</DataItem>
      </Geometry>
      <Attribute Name="Phase" AttributeType="Scalar" Center="Cell">
        <DataItem Format="HDF" Dimensions="{size_X} {size_Y} {size_Z}" NumberType="{xdmf_type}" Precision="{xdmf_precision}" >
          {hdf5_filename_3d}:/{hdf5_dataset_name}
        </DataItem>
      </Attribute>
    </Grid>
  </Domain>
</Xdmf>
"""
    try:
        with open(xmf_filename, 'w') as f: f.write(xdmf_content)
        print(f"Successfully generated XDMF file: {xmf_filename}")
    except Exception as e: print(f"Error writing XDMF file: {e}")
except Exception as e: print(f"Error saving 3D HDF5 file or generating XDMF: {e}")


# --- Save the plotted 2D slice as 8-bit TIFF ---
# (Uses imageio, saves standard 8-bit grayscale 0/255)
try:
    im_slice_uint8 = (slice_to_plot.astype(np.uint8) * 255)
    imageio.imwrite(tiff_slice_filename, im_slice_uint8)
    print(f"Successfully saved 2D TIFF image slice (8-bit) to: {tiff_slice_filename}")
except Exception as e:
    print(f"Error saving 2D TIFF image slice: {e}")

# --- Save the full 3D image as a 1-bit multi-page TIFF (stack) using tifffile ---
# (Saves True=1, False=0)
try:
    tifffile.imwrite(
        tiff_stack_filename,
        im_3d,
        photometric='minisblack',
        compression='None',
    )
    print(f"Successfully saved 3D TIFF image stack (1-bit) to: {tiff_stack_filename}")
except NameError:
     print(f"Skipping 3D TIFF stack save: 'tifffile' library not found or imported.")
except ImportError:
     print(f"Skipping 3D TIFF stack save: 'tifffile' library not installed? Try 'pip install tifffile'.")
except Exception as e:
    print(f"Error saving 3D TIFF image stack using tifffile: {e}")

# --- Save the full 3D image as a raw binary file (uint8) --- <-- NEW BLOCK
# (Saves True=1, False=0 as raw bytes in XYZ order)
try:
    print(f"Attempting to save 3D raw binary file (uint8, XYZ order) to: {raw_filename}")
    # Convert boolean data to uint8 (0 or 1)
    im_3d_uint8 = im_3d.astype(np.uint8)
    # Ensure data is in C-contiguous order (XYZ layout required by RawReader)
    if not im_3d_uint8.flags['C_CONTIGUOUS']:
        im_3d_uint8 = np.ascontiguousarray(im_3d_uint8)
        print("  (Data converted to C-contiguous order for raw saving)")

    # Open file in binary write mode and save raw bytes
    with open(raw_filename, 'wb') as f:
        f.write(im_3d_uint8.tobytes()) # Write the raw byte representation

    # Verification: Check file size (optional)
    import os
    file_size = os.path.getsize(raw_filename)
    expected_size = np.prod(im_3d_uint8.shape) * im_3d_uint8.itemsize
    if file_size == expected_size:
        print(f"Successfully saved 3D raw binary file ({file_size} bytes) to: {raw_filename}")
    else:
        print(f"Warning: Raw file size ({file_size}) differs from expected ({expected_size}) for {raw_filename}")

except Exception as e:
    print(f"Error saving 3D raw binary file: {e}")


# --- Save 2D slice to its own HDF5 file ---
# (Saves solid=1, void=0 as uint8)
try:
    with h5py.File(hdf5_filename_2d, 'w') as f:
        slice_to_save = slice_to_plot.astype(np.uint8)
        f.create_dataset(hdf5_dataset_name, data=slice_to_save)
    print(f"Successfully saved 2D slice HDF5 file to: {hdf5_filename_2d} (Dataset: '{hdf5_dataset_name}')")
except Exception as e:
    print(f"Error saving 2D slice HDF5 file: {e}")

print("\nScript finished.")
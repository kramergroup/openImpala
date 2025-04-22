#include "HDF5Reader.H"

#include <cstddef>
#include <sstream>
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <algorithm>
#include <map>

#include <AMReX.H>
#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::The_Pinned_Arena, amrex::ParallelFor
#include <AMReX_Utility.H>
#include <AMReX_MFIter.H>        // Needed for MFIter
#include <AMReX_BaseFab.H>      // Needed for BaseFab<T> temporary buffer
#include <AMReX_Array4.H>       // Needed for Array4 access
#include <AMReX_ParallelDescriptor.H> // Needed for MyProc in debug prints

// --- HDF5 C++ API Include ---
#include <H5Cpp.h>

// Make sure H5_USE_110_API is defined if using HDF5 1.10+ API features implicitly
// (Usually handled by HDF5 build configuration)

namespace OpenImpala {

//-----------------------------------------------------------------------
// Helper function for reading attributes
//-----------------------------------------------------------------------
namespace { // Keep helpers internal if possible
std::string readAttributeAsString(const H5::H5Object& obj, const std::string& attr_name) {
    std::string value = "";
    try {
        H5::Attribute attr = obj.openAttribute(attr_name);
        H5::DataType type = attr.getDataType();

        if (type.getClass() == H5T_STRING) {
            H5::StrType str_type = attr.getStrType();
            attr.read(type, value);
        } else {
             // Attempt to read as double/int and convert to string for non-string types
             try {
                 if (type.getSize() == sizeof(double) && type.getClass() == H5T_FLOAT) {
                    double dval;
                    attr.read(H5::PredType::NATIVE_DOUBLE, &dval);
                    value = std::to_string(dval);
                 } else if (type.getSize() == sizeof(int) && type.getClass() == H5T_INTEGER) {
                    int ival;
                    attr.read(H5::PredType::NATIVE_INT, &ival);
                    value = std::to_string(ival);
                 } // Add more types as needed
                 else {
                    value = "<Non-string/unhandled Attribute>";
                 }
             } catch (...) { // Catch potential errors during non-string read/convert
                value = "<Attribute Read Error>";
             }
        }
    } catch (H5::Exception& /*error*/) { // Catch HDF5 error specifically
         value = "<Attribute not found>"; // More specific error
    } catch (std::exception& /*std_err*/) { // Catch other potential errors
         value = "<Attribute Read Error>";
    }
    return value;
}
} // end anonymous namespace


//-----------------------------------------------------------------------
// Constructor Implementations
//-----------------------------------------------------------------------

HDF5Reader::HDF5Reader() :
    m_width(0), m_height(0), m_depth(0), m_is_read(false)
{
    // Initialize native type to something invalid/default
    m_native_type = H5::DataType(H5::PredType::NATIVE_DOUBLE); // Example default
}

// Constructor that reads metadata and throws on error
HDF5Reader::HDF5Reader(const std::string& filename, const std::string& hdf5dataset) :
    HDF5Reader() // Delegate for default member initialization
{
    if (!readFile(filename, hdf5dataset)) {
        throw std::runtime_error("HDF5Reader: Failed to read metadata for dataset '" + hdf5dataset +
                                 "' from file '" + filename + "' during construction.");
    }
}

//-----------------------------------------------------------------------
// Metadata Reading Implementation
//-----------------------------------------------------------------------

bool HDF5Reader::readFile(const std::string& filename, const std::string& hdf5dataset)
{
    m_filename = filename;
    m_hdf5dataset = hdf5dataset;
    return readMetadataInternal(); // Delegate to internal implementation
}

// Internal implementation using HDF5 C++ API to read METADATA ONLY
bool HDF5Reader::readMetadataInternal()
{
    // Reset state before reading
    m_width = 0; m_height = 0; m_depth = 0;
    m_is_read = false;
    m_native_type = H5::DataType(H5::PredType::NATIVE_DOUBLE); // Reset type

    try {
        // Open the HDF5 file read-only
        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        // Open the specified dataset
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);
        // --- Get Dataspace and Dimensions ---
        H5::DataSpace dataspace = dataset.getSpace();
        int rank = dataspace.getSimpleExtentNdims();

        if (rank != 3) {
            amrex::Print() << "Error: [HDF5Reader] Dataset '" << m_hdf5dataset << "' in file '"
                           << m_filename << "' has rank " << rank << ", expected 3.\n";
            return false;
        }
        hsize_t file_dims[3];
        // NOTE: HDF5 C API typically returns dimensions in C-order (slowest to fastest index: e.g., Z, Y, X)
        // Check H5Sget_simple_extent_dims documentation or test to confirm order if needed.
        // Assuming for now it returns [dim0, dim1, dim2] where dim0 is slowest varying.
        // Let's assume HDF5 store is (Z, Y, X) for C-order default.
        dataspace.getSimpleExtentDims(file_dims);

        // Assuming file_dims[0]=Z, file_dims[1]=Y, file_dims[2]=X for C order
        // AMReX uses X, Y, Z order for width, height, depth
        int dim_x = static_cast<int>(file_dims[2]);
        int dim_y = static_cast<int>(file_dims[1]);
        int dim_z = static_cast<int>(file_dims[0]);

        // Store dimensions as int, check for potential overflow
        if (file_dims[0] > static_cast<hsize_t>(std::numeric_limits<int>::max()) ||
            file_dims[1] > static_cast<hsize_t>(std::numeric_limits<int>::max()) ||
            file_dims[2] > static_cast<hsize_t>(std::numeric_limits<int>::max()))
        {
            amrex::Print() << "Error: [HDF5Reader] Dataset dimensions exceed integer limits.\n";
            return false;
        }

        // Store in AMReX convention (Width=X, Height=Y, Depth=Z)
        m_width  = dim_x;
        m_height = dim_y;
        m_depth  = dim_z;

        if (m_width <= 0 || m_height <= 0 || m_depth <= 0) {
             amrex::Print() << "Error: [HDF5Reader] Invalid dimensions read from dataset in " << m_filename
                            << " (X=" << m_width << ", Y=" << m_height << ", Z=" << m_depth << ")\n";
             return false;
        }

        // --- Get and Store Native Data Type ---
        m_native_type = dataset.getDataType(); // Store the type object

        if (amrex::ParallelDescriptor::IOProcessor()) {
            H5T_class_t dt_class = m_native_type.getClass();
            size_t dt_size = m_native_type.getSize();
            amrex::Print() << "DEBUG: Detected native HDF5 type Class=" << dt_class << ", Size=" << dt_size << "\n";
        }

        m_is_read = true; // Metadata read successfully
        if (amrex::ParallelDescriptor::IOProcessor()) {
             amrex::Print() << "Successfully read HDF5 Metadata: " << m_filename << " [" << m_hdf5dataset << "]"
                           << " (Dimensions: X=" << m_width << ", Y=" << m_height << ", Z=" << m_depth << ")\n";
        }
        return true;

    } catch (H5::Exception& error) {
        amrex::Print() << "HDF5 Error during HDF5Reader metadata read for file '" << m_filename
                       << "', dataset '" << m_hdf5dataset << "':\n";
        amrex::Print() << "  " << error.getDetailMsg() << "\n";
        m_width = m_height = m_depth = 0;
        m_is_read = false;
        return false;
    } catch (std::exception& std_err) {
        std::string err_msg = "Standard Exception during HDF5Reader metadata read: " + std::string(std_err.what());
        amrex::Warning(err_msg.c_str());
        m_width = m_height = m_depth = 0;
        m_is_read = false;
        return false;
    }
}

//-----------------------------------------------------------------------
// Getter Implementations
//-----------------------------------------------------------------------
int HDF5Reader::width() const { return m_width; }
int HDF5Reader::height() const { return m_height; }
int HDF5Reader::depth() const { return m_depth; }
amrex::Box HDF5Reader::box() const {
    if (!m_is_read) return amrex::Box();
    // AMReX Box uses (X, Y, Z) convention
    return amrex::Box(amrex::IntVect::TheZeroVector(),
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

//-----------------------------------------------------------------------
// Attribute Reading Implementations
//-----------------------------------------------------------------------
std::string HDF5Reader::getAttribute(const std::string& attr_name) const {
    if (!m_is_read) {
        return "<Attribute read error: Metadata not read>";
    }
    std::string value = "<Attribute Error>";
    try {
        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);
        value = readAttributeAsString(dataset, attr_name); // Use helper
    } catch (H5::Exception& /*error*/) {
        value = "<HDF5 Error reading attr '" + attr_name + "'>";
    } catch (std::exception& std_err) {
        value = "<Error reading attr '" + attr_name + "': " + std_err.what() + ">";
    }
    return value;
}

std::map<std::string, std::string> HDF5Reader::getAllAttributes() const {
     std::map<std::string, std::string> attributes;
     if (!m_is_read) {
         return attributes; // Return empty map
     }
     try {
         H5::H5File file(m_filename, H5F_ACC_RDONLY);
         H5::DataSet dataset = file.openDataSet(m_hdf5dataset);
         int num_attrs = dataset.getNumAttrs();

         for (int i = 0; i < num_attrs; ++i) {
             H5::Attribute attr = dataset.openAttribute(static_cast<unsigned int>(i));
             std::string name = attr.getName();
             attributes[name] = readAttributeAsString(dataset, name); // Reuse helper
         }
     } catch (H5::Exception& error) {
         std::string warn_msg = "HDF5 Warning in HDF5Reader::getAllAttributes for file '" + m_filename +
                                "', dataset '" + m_hdf5dataset + "':\n  " + error.getDetailMsg();
         amrex::Warning(warn_msg.c_str());
         attributes.clear();
     } catch (std::exception& std_err) {
         std::string err_msg = "Standard Exception in HDF5Reader::getAllAttributes: " + std::string(std_err.what());
         amrex::Warning(err_msg.c_str());
         attributes.clear();
     }
     return attributes;
}

//-----------------------------------------------------------------------
// Threshold Implementation (Main Refactored Logic)
//-----------------------------------------------------------------------

// Private template helper function with corrected memspace and dimension ordering
template<typename T_Native>
void HDF5Reader::readAndThresholdFab(H5::DataSet& dataset, double raw_threshold,
                                     int value_if_true, int value_if_false,
                                     const amrex::Box& box, amrex::IArrayBox& fab) const
{
    // Determine PredType
    H5::PredType native_pred_type = H5::PredType::NATIVE_DOUBLE; // Default/fallback
    if constexpr (std::is_same_v<T_Native, uint8_t>)  { native_pred_type = H5::PredType::NATIVE_UINT8; }
    else if constexpr (std::is_same_v<T_Native, int8_t>)   { native_pred_type = H5::PredType::NATIVE_INT8; }
    else if constexpr (std::is_same_v<T_Native, uint16_t>) { native_pred_type = H5::PredType::NATIVE_UINT16; }
    else if constexpr (std::is_same_v<T_Native, int16_t>)  { native_pred_type = H5::PredType::NATIVE_INT16; }
    else if constexpr (std::is_same_v<T_Native, uint32_t>) { native_pred_type = H5::PredType::NATIVE_UINT32; }
    else if constexpr (std::is_same_v<T_Native, int32_t>)  { native_pred_type = H5::PredType::NATIVE_INT32; }
    else if constexpr (std::is_same_v<T_Native, uint64_t>) { native_pred_type = H5::PredType::NATIVE_UINT64; }
    else if constexpr (std::is_same_v<T_Native, int64_t>)  { native_pred_type = H5::PredType::NATIVE_INT64; }
    else if constexpr (std::is_same_v<T_Native, float>)    { native_pred_type = H5::PredType::NATIVE_FLOAT; }
    else if constexpr (std::is_same_v<T_Native, double>)   { native_pred_type = H5::PredType::NATIVE_DOUBLE; }
    else { amrex::Abort("readAndThresholdFab: Unsupported native type T_Native"); }

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: readAndThresholdFab<T_Native> entered for box: " << box << "\n";
    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: Using HDF5 PredType corresponding to T_Native.\n";
    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: Creating temp_fab...\n";

    amrex::BaseFab<T_Native> temp_fab(box, 1, amrex::The_Pinned_Arena()); // Use pinned memory

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: temp_fab created. Getting filespace...\n";

    H5::DataSpace filespace = dataset.getSpace(); // Get file dataspace

    // Define hyperslab in file dataspace based on the box
    hsize_t offset[3], count[3];
    const amrex::IntVect& smallEnd = box.smallEnd();
    const amrex::IntVect& box_size = box.size();

    // <<< FIX: Reverse order for HDF5 (C-order: Z, Y, X) from AMReX Box (X, Y, Z) >>>
    offset[0] = static_cast<hsize_t>(smallEnd[2]); // HDF5 Dim 0 = AMReX Dim 2 (Z)
    offset[1] = static_cast<hsize_t>(smallEnd[1]); // HDF5 Dim 1 = AMReX Dim 1 (Y)
    offset[2] = static_cast<hsize_t>(smallEnd[0]); // HDF5 Dim 2 = AMReX Dim 0 (X)

    count[0] = static_cast<hsize_t>(box_size[2]); // HDF5 Dim 0 = AMReX Dim 2 (Z)
    count[1] = static_cast<hsize_t>(box_size[1]); // HDF5 Dim 1 = AMReX Dim 1 (Y)
    count[2] = static_cast<hsize_t>(box_size[0]); // HDF5 Dim 2 = AMReX Dim 0 (X)

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: Selecting hyperslab (HDF5 C-Order Z,Y,X): offset=("
                       << offset[0] << "," << offset[1] << "," << offset[2] << "), count=("
                       << count[0] << "," << count[1] << "," << count[2] << ")...\n";

    filespace.selectHyperslab(H5S_SELECT_SET, count, offset); // Select the chunk in the file

    // Define memory dataspace as 1D matching the contiguous buffer size
    hsize_t mem_dims[1] = { static_cast<hsize_t>(box.numPts()) }; // Total points in the box
    H5::DataSpace memspace(1, mem_dims);                         // Rank 1

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: Defined memory dataspace (Rank " << memspace.getSimpleExtentNdims() << " Size " << mem_dims[0] << "). Calling dataset.read()...\n";

    // *** Read data into temp_fab ***
    dataset.read(temp_fab.dataPtr(), native_pred_type, memspace, filespace);

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: dataset.read() completed. Preparing for ParallelFor...\n";

    // Apply threshold using Array4 interface
    amrex::Array4<int> const& fab_arr = fab.array(); // Target iMultiFab Array4
    amrex::Array4<const T_Native> const& temp_arr = temp_fab.const_array(); // Source temp data Array4

    // Use ParallelFor for potential OMP/GPU execution
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Convert native value to double for comparison
        double value_as_double = static_cast<double>(temp_arr(i, j, k));
        fab_arr(i, j, k) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
    });

    amrex::AllPrint() << "DEBUG [Rank " << amrex::ParallelDescriptor::MyProc() << "]: ParallelFor completed. Exiting readAndThresholdFab for box: " << box << "\n";
}


// Public threshold method - determines native type and calls template helper
void HDF5Reader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        throw std::runtime_error("[HDF5Reader::threshold] Cannot threshold, metadata not read successfully.");
    }

    try {
        if (amrex::ParallelDescriptor::IOProcessor()) {
            H5T_class_t dt_class = m_native_type.getClass();
            size_t dt_size = m_native_type.getSize();
            amrex::Print() << "DEBUG: Thresholding based on detected native HDF5 type Class=" << dt_class << ", Size=" << dt_size << "\n";
        }

        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mf, true); mfi.isValid(); ++mfi) // Use Tiling MFIter
        {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = mf[mfi]; // Get current FArrayBox

            // Print box info before dispatch
            amrex::AllPrint() << "DEBUG: Processing tile_box: " << tile_box << "\n";

            // Dispatch based on stored native type
            if (m_native_type == H5::PredType::NATIVE_UINT8) {
                 amrex::AllPrint() << "DEBUG: Dispatching as uint8_t\n";
                 readAndThresholdFab<uint8_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT8) {
                 amrex::AllPrint() << "DEBUG: Dispatching as int8_t\n";
                 readAndThresholdFab<int8_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT16) {
                 amrex::AllPrint() << "DEBUG: Dispatching as uint16_t\n";
                 readAndThresholdFab<uint16_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT16) {
                 amrex::AllPrint() << "DEBUG: Dispatching as int16_t\n";
                 readAndThresholdFab<int16_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT32) {
                 amrex::AllPrint() << "DEBUG: Dispatching as uint32_t\n";
                 readAndThresholdFab<uint32_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT32) {
                 amrex::AllPrint() << "DEBUG: Dispatching as int32_t\n";
                 readAndThresholdFab<int32_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_FLOAT) {
                 amrex::AllPrint() << "DEBUG: Dispatching as float\n";
                 readAndThresholdFab<float>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_DOUBLE) {
                 amrex::AllPrint() << "DEBUG: Dispatching as double\n";
                 readAndThresholdFab<double>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            }
            // Add cases for NATIVE_LONG, NATIVE_ULONG, INT64, UINT64 etc. if needed
            else {
                // Handle unsupported type (Corrected error reporting)
                H5T_class_t type_class_enum = m_native_type.getClass(); // Correct type
                std::string err_msg = "[HDF5Reader::threshold] Unsupported native HDF5 data type detected in file. Class enum value: "
                                    + std::to_string(static_cast<int>(type_class_enum));
                try { err_msg += ", Size (bytes): " + std::to_string(m_native_type.getSize()); } catch (...) {}
                throw std::runtime_error(err_msg);
            }
        } // End MFIter loop

    } catch (H5::Exception& error) {
        // Catch HDF5 specific exceptions during file/dataset open or read
        throw std::runtime_error("[HDF5Reader::threshold] HDF5 Error during threshold operation:\n  " + error.getDetailMsg());
    } catch (std::exception& std_err) {
        // Catch standard exceptions (e.g., from type dispatch error, ParallelFor)
        throw std::runtime_error("[HDF5Reader::threshold] Standard Exception during threshold operation: " + std::string(std_err.what()));
    }
}

// Original overload (output 1/0) - calls the flexible version
void HDF5Reader::threshold(double raw_threshold, amrex::iMultiFab& mf) const
{
    threshold(raw_threshold, 1, 0, mf); // Call the flexible version with 1/0
}


} // namespace OpenImpala

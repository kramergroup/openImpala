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
#include <AMReX_GpuContainers.H>
#include <AMReX_Utility.H>
#include <AMReX_MFIter.H>     // Needed for MFIter
#include <AMReX_BaseFab.H>   // Needed for BaseFab<T> temporary buffer
#include <AMReX_Array4.H>    // Needed for Array4 access

// --- HDF5 C++ API Include ---
#include <H5Cpp.h>

// Make sure H5_USE_110_API is defined if using HDF5 1.10+ API features implicitly
// (Usually handled by HDF5 build configuration)

namespace OpenImpala {

//-----------------------------------------------------------------------
// Helper function for reading attributes (moved from anonymous namespace)
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
             // Attempt to read as double and convert to string for non-string types
             // More robust handling might be needed for different attribute types
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
        // H5::Exception::dontPrint(); // Optional: disable HDF5 stack trace

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
        dataspace.getSimpleExtentDims(file_dims);

        // Store dimensions as int, check for potential overflow
        if (file_dims[0] > static_cast<hsize_t>(std::numeric_limits<int>::max()) ||
            file_dims[1] > static_cast<hsize_t>(std::numeric_limits<int>::max()) ||
            file_dims[2] > static_cast<hsize_t>(std::numeric_limits<int>::max()))
        {
            amrex::Print() << "Error: [HDF5Reader] Dataset dimensions exceed integer limits.\n";
            return false;
        }
        m_width  = static_cast<int>(file_dims[0]);
        m_height = static_cast<int>(file_dims[1]);
        m_depth  = static_cast<int>(file_dims[2]);

        if (m_width <= 0 || m_height <= 0 || m_depth <= 0) {
             amrex::Print() << "Error: [HDF5Reader] Invalid dimensions read from dataset in " << m_filename
                            << " (W=" << m_width << ", H=" << m_height << ", D=" << m_depth << ")\n";
             return false;
        }

        // --- Get and Store Native Data Type ---
        m_native_type = dataset.getDataType(); // Store the type object

        // Optionally print type info for debugging
        // H5::DataTypeClass dt_class = m_native_type.getClass();
        // size_t dt_size = m_native_type.getSize();
        // amrex::Print() << "DEBUG: Detected native HDF5 type Class=" << dt_class << ", Size=" << dt_size << "\n";


        // H5 Objects automatically closed by RAII

        m_is_read = true; // Metadata read successfully
        if (amrex::ParallelDescriptor::IOProcessor()) { // Print only on IO rank
             amrex::Print() << "Successfully read HDF5 Metadata: " << m_filename << " [" << m_hdf5dataset << "]"
                           << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
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
    // Assuming global domain starts at (0,0,0)
    return amrex::Box(amrex::IntVect::TheZeroVector(),
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

//-----------------------------------------------------------------------
// Attribute Reading Implementations
//-----------------------------------------------------------------------
// (These remain largely the same, but might fail if called before metadata read)

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

// Private template helper function
template<typename T_Native>
void HDF5Reader::readAndThresholdFab(H5::DataSet& dataset, double raw_threshold,
                                     int value_if_true, int value_if_false,
                                     const amrex::Box& box, amrex::IArrayBox& fab) const
{
    // Get the HDF5 PredType corresponding to T_Native
    // This requires a small helper or explicit checks
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
    else {
        // This should not happen if called correctly based on m_native_type
        amrex::Abort("readAndThresholdFab: Unsupported native type T_Native");
    }


    // Create temporary AMReX BaseFab for holding the chunk read from HDF5
    amrex::BaseFab<T_Native> temp_fab(box, 1, amrex::The_Pinned_Arena()); // Use pinned memory for potential async/GPU

    // Get file dataspace
    H5::DataSpace filespace = dataset.getSpace();

    // Define hyperslab in file dataspace based on the box
    hsize_t offset[3], count[3];
    const amrex::IntVect& smallEnd = box.smallEnd();
    const amrex::IntVect& box_size = box.size();

    // HDF5 is typically C-ordered (last index varies fastest), check convention if needed
    // AMReX Box is (x,y,z), assuming HDF5 dataset is also (x,y,z)
    offset[0] = static_cast<hsize_t>(smallEnd[0]);
    offset[1] = static_cast<hsize_t>(smallEnd[1]);
    offset[2] = static_cast<hsize_t>(smallEnd[2]);

    count[0] = static_cast<hsize_t>(box_size[0]);
    count[1] = static_cast<hsize_t>(box_size[1]);
    count[2] = static_cast<hsize_t>(box_size[2]);

    // Select the chunk in the file
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // Define memory dataspace (matches the layout of the FArrayBox/BaseFab)
    // Assuming standard C-order for AMReX BaseFab memory
    hsize_t mem_dims[3];
    mem_dims[0] = static_cast<hsize_t>(box_size[0]);
    mem_dims[1] = static_cast<hsize_t>(box_size[1]);
    mem_dims[2] = static_cast<hsize_t>(box_size[2]);
    H5::DataSpace memspace(3, mem_dims); // Rank 3, matching box dimensions

    // Read the selected hyperslab directly into the temporary BaseFab's memory
    dataset.read(temp_fab.dataPtr(), native_pred_type, memspace, filespace);

    // Now apply threshold using Array4 interface
    amrex::Array4<int> const& fab_arr = fab.array(); // Target iMultiFab Array4
    amrex::Array4<const T_Native> const& temp_arr = temp_fab.const_array(); // Source temp data Array4

    // Use ParallelFor if OMP/GPU enabled, otherwise LoopOnCpu
    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // Convert native value to double for comparison
        // Be mindful of potential precision issues for large 64-bit integers
        double value_as_double = static_cast<double>(temp_arr(i, j, k));
        fab_arr(i, j, k) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
    });
}


// Public threshold method - determines native type and calls template helper
void HDF5Reader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        throw std::runtime_error("[HDF5Reader::threshold] Cannot threshold, metadata not read successfully.");
    }

    try {
        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);

        // Check native type and call the appropriate template instantiation
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mf, true); mfi.isValid(); ++mfi) // Use Tiling MFIter
        {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = mf[mfi]; // Get current FArrayBox

            // Dispatch based on stored native type
            // NOTE: This requires comparing H5::DataType objects. Using PredType::NATIVE_* assumes
            // the file uses standard native types. Handling complex/custom types would need more work.
            if (m_native_type == H5::PredType::NATIVE_UINT8) {
                readAndThresholdFab<uint8_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT8) {
                readAndThresholdFab<int8_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT16) {
                readAndThresholdFab<uint16_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT16) {
                readAndThresholdFab<int16_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT32) {
                readAndThresholdFab<uint32_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT32) {
                readAndThresholdFab<int32_t>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_FLOAT) {
                readAndThresholdFab<float>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_DOUBLE) {
                readAndThresholdFab<double>(dataset, raw_threshold, value_if_true, value_if_false, tile_box, fab);
            }
            // Add cases for NATIVE_LONG, NATIVE_ULONG, INT64, UINT64 etc. if needed
            else {
                // Get details for error message
                 std::string type_class_str = "";
                 H5std_string class_name;
                 H5::DataTypeClass type_class = m_native_type.getClass();
                 // ... (code to convert type_class enum to string if desired) ...

                 throw std::runtime_error("[HDF5Reader::threshold] Unsupported native HDF5 data type detected in file. Class: " + std::to_string(type_class));
            }
        } // End MFIter loop

    } catch (H5::Exception& error) {
        throw std::runtime_error("[HDF5Reader::threshold] HDF5 Error during threshold operation:\n  " + error.getDetailMsg());
    } catch (std::exception& std_err) {
        throw std::runtime_error("[HDF5Reader::threshold] Standard Exception during threshold operation: " + std::string(std_err.what()));
    }
}

// Original overload (output 1/0) - calls the flexible version
void HDF5Reader::threshold(double raw_threshold, amrex::iMultiFab& mf) const
{
    threshold(raw_threshold, 1, 0, mf); // Call the flexible version with 1/0
}


} // namespace OpenImpala

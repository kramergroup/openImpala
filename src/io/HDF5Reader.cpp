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
#include <AMReX_MFIter.H>       // Needed for MFIter
#include <AMReX_BaseFab.H>      // Needed for BaseFab<T> temporary buffer
#include <AMReX_Array4.H>       // Needed for Array4 access
#include <AMReX_ParallelDescriptor.H> // Needed for IOProcessor
#include <AMReX_GpuAssert.H>    // Needed for AMREX_ASSERT (Can likely remove if not using)

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
        dataspace.getSimpleExtentDims(file_dims);

        // Assuming file_dims[0]=Z, file_dims[1]=Y, file_dims[2]=X for C order
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

        // (Removed DEBUG print for native type)

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

template<typename T_Native>
void HDF5Reader::readAndCopyFab(H5::DataSet& dataset,
                                  const amrex::Box& box, amrex::IArrayBox& fab) const
{
    // Determine PredType from T_Native
    H5::PredType native_pred_type;
    if constexpr (std::is_same_v<T_Native, uint8_t>)  { native_pred_type = H5::PredType::NATIVE_UINT8; }
    else if constexpr (std::is_same_v<T_Native, int8_t>)   { native_pred_type = H5::PredType::NATIVE_INT8; }
    else if constexpr (std::is_same_v<T_Native, uint16_t>) { native_pred_type = H5::PredType::NATIVE_UINT16; }
    else if constexpr (std::is_same_v<T_Native, int16_t>)  { native_pred_type = H5::PredType::NATIVE_INT16; }
    else if constexpr (std::is_same_v<T_Native, uint32_t>) { native_pred_type = H5::PredType::NATIVE_UINT32; }
    else if constexpr (std::is_same_v<T_Native, int32_t>)  { native_pred_type = H5::PredType::NATIVE_INT32; }
    else if constexpr (std::is_same_v<T_Native, float>)    { native_pred_type = H5::PredType::NATIVE_FLOAT; }
    else if constexpr (std::is_same_v<T_Native, double>)   { native_pred_type = H5::PredType::NATIVE_DOUBLE; }
    else { amrex::Abort("readAndCopyFab: Unsupported native type T_Native"); }

    // Create a temporary buffer in pinned memory to read the data chunk
    amrex::BaseFab<T_Native> temp_fab(box, 1, amrex::The_Pinned_Arena());

    // --- HDF5 Hyperslab Selection (same as in readAndThresholdFab) ---
    H5::DataSpace filespace = dataset.getSpace();
    hsize_t offset[3], count[3];
    const amrex::IntVect& smallEnd = box.smallEnd();
    const amrex::IntVect& box_size = box.size();
    offset[0] = static_cast<hsize_t>(smallEnd[2]); // Z
    offset[1] = static_cast<hsize_t>(smallEnd[1]); // Y
    offset[2] = static_cast<hsize_t>(smallEnd[0]); // X
    count[0] = static_cast<hsize_t>(box_size[2]);  // Z
    count[1] = static_cast<hsize_t>(box_size[1]);  // Y
    count[2] = static_cast<hsize_t>(box_size[0]);  // X
    filespace.selectHyperslab(H5S_SELECT_SET, count, offset);

    // Define memory dataspace
    hsize_t mem_dims_3d[3] = {count[0], count[1], count[2]};
    H5::DataSpace memspace(3, mem_dims_3d);

    // Read data chunk from file into the temporary buffer
    dataset.read(temp_fab.dataPtr(), native_pred_type, memspace, filespace);

    // --- Process the data from the temporary buffer into the final iMultiFab ---
    amrex::Array4<int> const& fab_arr = fab.array();
    amrex::Array4<const T_Native> const& temp_arr = temp_fab.const_array();

    amrex::ParallelFor(box, [=] AMREX_GPU_DEVICE (int i, int j, int k) noexcept
    {
        // This is the key change: direct copy instead of thresholding
        // We round to handle cases where phase IDs might be stored as floats
        fab_arr(i, j, k) = static_cast<int>(std::round(static_cast<double>(temp_arr(i, j, k))));
    });
}


void HDF5Reader::readPhaseIDs(amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        throw std::runtime_error("[HDF5Reader::readPhaseIDs] Cannot process, metadata not read successfully.");
    }
    try {
        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
        for (amrex::MFIter mfi(mf, true); mfi.isValid(); ++mfi)
        {
            const amrex::Box& tile_box = mfi.tilebox();
            amrex::IArrayBox& fab = mf[mfi];

            // Dispatch to the NEW helper based on the stored native type
            if (m_native_type == H5::PredType::NATIVE_UINT8) {
                readAndCopyFab<uint8_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT8) {
                readAndCopyFab<int8_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT16) {
                readAndCopyFab<uint16_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT16) {
                readAndCopyFab<int16_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_UINT32) {
                readAndCopyFab<uint32_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_INT32) {
                readAndCopyFab<int32_t>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_FLOAT) {
                readAndCopyFab<float>(dataset, tile_box, fab);
            } else if (m_native_type == H5::PredType::NATIVE_DOUBLE) {
                readAndCopyFab<double>(dataset, tile_box, fab);
            }
            else {
                H5T_class_t type_class_enum = m_native_type.getClass();
                std::string err_msg = "[HDF5Reader::readPhaseIDs] Unsupported native HDF5 data type. Class enum: "
                                    + std::to_string(static_cast<int>(type_class_enum));
                throw std::runtime_error(err_msg);
            }
        } // End MFIter loop

    } catch (H5::Exception& error) {
        throw std::runtime_error("[HDF5Reader::readPhaseIDs] HDF5 Error:\n  " + error.getDetailMsg());
    } catch (std::exception& std_err) {
        throw std::runtime_error("[HDF5Reader::readPhaseIDs] Standard Exception: " + std::string(std_err.what()));
    }
}
        
// Original overload (output 1/0) - calls the flexible version
void HDF5Reader::threshold(double raw_threshold, amrex::iMultiFab& mf) const
{
    threshold(raw_threshold, 1, 0, mf); // Call the flexible version with 1/0
}

} // namespace OpenImpala

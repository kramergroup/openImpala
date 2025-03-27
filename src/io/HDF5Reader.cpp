#include "HDF5Reader.H"

#include <fstream>   // Not strictly needed now, using HDF5 API
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <limits>    // For std::numeric_limits
#include <cmath>     // For std::abs (though amrex::Math::abs is better)
#include <cstdint>   // For type conversion checks if needed
#include <numeric>   // For std::accumulate if used
#include <algorithm> // For std::copy, std::transform
#include <map>       // For attribute map

#include <AMReX.H>
#include <AMReX_Print.H> // For amrex::Print, amrex::Warning
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::LoopOnCpu
#include <AMReX_Utility.H>       // Maybe needed?
// AMReX_Extension.H might be needed for AMREX_LIKELY, but we remove it instead

// --- HDF5 C++ API Include ---
#include <H5Cpp.h>

namespace OpenImpala {

//-----------------------------------------------------------------------
// Helper function for reading/converting data
//-----------------------------------------------------------------------
namespace { // Anonymous namespace for internal helper

// Reads data from dataset, converting known types to double.
// Returns true on success, false on failure (and prints error).
bool readAndConvert(H5::DataSet& dataset, const H5::DataType& dtype, std::vector<double>& target_vector)
{
    const size_t num_elements = target_vector.size();
    if (num_elements == 0) return true; // Nothing to read

    try {
        // Check common native types and read/convert
        if (dtype == H5::PredType::NATIVE_DOUBLE) {
            dataset.read(target_vector.data(), H5::PredType::NATIVE_DOUBLE);
        } else if (dtype == H5::PredType::NATIVE_FLOAT) {
            std::vector<float> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_FLOAT);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // float -> double
        } else if (dtype == H5::PredType::NATIVE_INT) { // System 'int'
            std::vector<int> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_INT);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // int -> double
        } else if (dtype == H5::PredType::NATIVE_UINT16) {
            std::vector<uint16_t> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_UINT16);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // uint16 -> double
        } else if (dtype == H5::PredType::NATIVE_INT16) {
            std::vector<int16_t> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_INT16);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // int16 -> double
        } else if (dtype == H5::PredType::NATIVE_UINT8) {
            std::vector<uint8_t> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_UINT8);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // uint8 -> double
        } else if (dtype == H5::PredType::NATIVE_INT8) {
            std::vector<int8_t> temp_raw(num_elements);
            dataset.read(temp_raw.data(), H5::PredType::NATIVE_INT8);
            std::copy(temp_raw.begin(), temp_raw.end(), target_vector.begin()); // int8 -> double
        }
        // Add other types as needed (NATIVE_LONG, NATIVE_UINT, NATIVE_UINT32, etc.)
        // Beware of potential precision loss when converting large 64-bit integers to double.
        else {
            amrex::Print() << "Error: [HDF5Reader] Unsupported HDF5 data type encountered. Class: "
                           << dtype.getClass() << "\n"; // Optionally print more details
            return false;
        }
    } catch (H5::Exception& error) {
         amrex::Print() << "HDF5 Error during data read/convert:\n";
         amrex::Print() << "  " << error.getDetailMsg() << "\n";
         return false;
    }
    return true;
}

// Helper to read attribute value as string (example)
std::string readAttributeAsString(const H5::H5Object& obj, const std::string& attr_name) {
    std::string value = "";
    try {
        H5::Attribute attr = obj.openAttribute(attr_name);
        H5::DataType type = attr.getDataType();

        // *** FIX: Check H5T_STRING class instead of specific PredTypes ***
        if (type.getClass() == H5T_STRING)
        {
             // If it's a string type, read it directly into std::string
             // Note: Might need adjustment if dealing with fixed-size strings, but variable is common
             H5::StrType str_type = attr.getStrType(); // Can use this to get more details if needed
             attr.read(type, value);
        } else {
            // Handle non-string types (simplified - robust solution needs type checking)
            value = "<Non-string Attribute>";
        }
    } catch (H5::Exception& error) {
        // Attribute might not exist, treat as non-fatal here
        // std::string warn_msg = "HDF5 Warning: Could not read attribute '" + attr_name + "': " + error.getDetailMsg();
        // amrex::Warning(warn_msg);
        value = "<Attribute not found or read error>";
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
    // Default constructor: initializes members to default state.
}

// Constructor that reads file and throws on error
HDF5Reader::HDF5Reader(const std::string& filename, const std::string& hdf5dataset) :
    HDF5Reader() // Delegate for default member initialization
{
    // Try reading the file; rely on readFile to set members and print errors.
    if (!readFile(filename, hdf5dataset)) {
        // Construct a meaningful error message
        throw std::runtime_error("HDF5Reader: Failed to read dataset '" + hdf5dataset +
                                 "' from file '" + filename + "' during construction.");
    }
}

//-----------------------------------------------------------------------
// File Reading Implementation
//-----------------------------------------------------------------------

bool HDF5Reader::readFile(const std::string& filename, const std::string& hdf5dataset)
{
    m_filename = filename;
    m_hdf5dataset = hdf5dataset;
    return readHDF5FileInternal(); // Delegate to internal implementation
}

// Internal implementation using HDF5 C++ API
bool HDF5Reader::readHDF5FileInternal()
{
    // Reset state before reading
    m_raw.clear();
    m_width = 0;
    m_height = 0;
    m_depth = 0;
    m_is_read = false;

    try {
        // Disable automatic HDF5 error printing stack if desired
        // H5::Exception::dontPrint();

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
            return false; // H5::* objects auto-closed by RAII here and on exit
        }

        hsize_t file_dims[3];
        dataspace.getSimpleExtentDims(file_dims);

        // Store dimensions as int, check for potential overflow from hsize_t (usually 64-bit unsigned)
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

        amrex::Long num_elements = static_cast<amrex::Long>(m_width) * m_height * m_depth;

        // --- Get Data Type and Read/Convert Data ---
        H5::DataType dtype = dataset.getDataType();

        // Resize internal storage vector (double)
        try {
             if (num_elements > static_cast<amrex::Long>(std::numeric_limits<size_t>::max())) {
                 throw std::overflow_error("Number of elements exceeds vector size limit.");
             }
             m_raw.resize(static_cast<size_t>(num_elements));
        } catch (const std::exception& e) {
             amrex::Print() << "Error: [HDF5Reader] Failed to allocate memory for " << num_elements
                            << " elements: " << e.what() << "\n";
             return false;
        }

        // Read data, converting supported types to double
        if (!readAndConvert(dataset, dtype, m_raw)) {
            // readAndConvert should print details
            m_raw.clear(); // Clear potentially partial data
            return false;
        }

        // H5::DataSet, H5::DataSpace, H5::DataType, H5::H5File automatically closed by RAII

        m_is_read = true;
        if (m_width > 0) { // Only print success if dimensions seem valid
             amrex::Print() << "Successfully read HDF5 Dataset: " << m_filename << " [" << m_hdf5dataset << "]"
                            << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
        } else {
             amrex::Warning("HDF5Reader: Read operation completed but resulted in zero dimensions.");
             m_is_read = false; // Treat as not successfully read if dimensions are zero
             return false;
        }
        return true;

    } catch (H5::Exception& error) {
        // Catch HDF5 C++ API exceptions
        amrex::Print() << "HDF5 Error in HDF5Reader processing file '" << m_filename
                       << "', dataset '" << m_hdf5dataset << "':\n";
        amrex::Print() << "  " << error.getDetailMsg() << "\n";
        m_raw.clear(); // Ensure data is cleared on error
        m_width = m_height = m_depth = 0;
        m_is_read = false;
        return false;
    } catch (std::exception& std_err) {
        // Catch potential standard exceptions (e.g., from memory allocation)
        std::string err_msg = "Standard Exception in HDF5Reader processing: " + std::string(std_err.what());
        amrex::Warning(err_msg); // Corrected amrex::Warning call
        m_raw.clear();
        m_width = m_height = m_depth = 0;
        m_is_read = false;
        return false;
    }
}

//-----------------------------------------------------------------------
// Getter Implementations
//-----------------------------------------------------------------------

int HDF5Reader::width() const {
    return m_width;
}

int HDF5Reader::height() const {
    return m_height;
}

int HDF5Reader::depth() const {
    return m_depth;
}

amrex::Box HDF5Reader::box() const {
    if (!m_is_read) {
        return amrex::Box(); // Return default (empty) box
    }
    // Box is cell-centered, index from 0 to dim-1
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

const std::vector<HDF5Reader::InternalDataType>& HDF5Reader::getRawData() const {
    return m_raw;
}

HDF5Reader::InternalDataType HDF5Reader::getRawValue(int i, int j, int k) const {
    if (!m_is_read) {
        throw std::out_of_range("[HDF5Reader::getRawValue] Data not read yet.");
    }
    // Check indices against stored dimensions
    if (i < 0 || i >= m_width || j < 0 || j >= m_height || k < 0 || k >= m_depth) {
        throw std::out_of_range("[HDF5Reader::getRawValue] Index ("
                                 + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)
                                 + ") out of bounds (W:" + std::to_string(m_width)
                                 + ", H:" + std::to_string(m_height) + ", D:" + std::to_string(m_depth) + ").");
    }

    // Calculate 1D index (assuming XYZ layout where Z varies slowest)
    amrex::Long idx = static_cast<amrex::Long>(k) * m_width * m_height +
                       static_cast<amrex::Long>(j) * m_width +
                       static_cast<amrex::Long>(i);

    // Bounds check for calculated index (safety)
    if (idx >= static_cast<amrex::Long>(m_raw.size())) {
         throw std::out_of_range("[HDF5Reader::getRawValue] Calculated index (" + std::to_string(idx)
                                + ") exceeds raw data vector size (" + std::to_string(m_raw.size()) + ").");
    }

    return m_raw[static_cast<size_t>(idx)];
}

//-----------------------------------------------------------------------
// Attribute Reading Implementations (Examples)
//-----------------------------------------------------------------------

std::string HDF5Reader::getAttribute(const std::string& attr_name) const {
    if (!m_is_read) {
        throw std::runtime_error("[HDF5Reader::getAttribute] Cannot read attribute, file/dataset not successfully read.");
    }
    std::string value = "<Attribute Error>"; // Default error value
    try {
        H5::H5File file(m_filename, H5F_ACC_RDONLY);
        H5::DataSet dataset = file.openDataSet(m_hdf5dataset);
        value = readAttributeAsString(dataset, attr_name); // Use helper
    } catch (H5::Exception& error) {
         // Make slightly less noisy - return error string instead of printing/throwing again
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
        amrex::Warning(warn_msg); // Corrected amrex::Warning call
        attributes.clear();
    } catch (std::exception& std_err) {
        std::string err_msg = "Standard Exception in HDF5Reader::getAllAttributes: " + std::string(std_err.what());
        amrex::Warning(err_msg); // Corrected amrex::Warning call
        attributes.clear();
    }
    return attributes;
}

//-----------------------------------------------------------------------
// Threshold Implementation
//-----------------------------------------------------------------------

// Overload with customizable true/false values
void HDF5Reader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        amrex::Abort("[HDF5Reader::threshold] Cannot threshold, data not read successfully.");
    }

    // Get raw pointer for potentially faster access inside loop
    const InternalDataType* const AMREX_RESTRICT data_ptr = m_raw.data();
    const amrex::Long raw_data_size = static_cast<amrex::Long>(m_raw.size());

    // Capture necessary members by value/reference for the lambda
    const int current_width = m_width;
    const int current_height = m_height;
    const int current_depth = m_depth;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox(); // Get the valid box for this FAB
        amrex::IArrayBox& fab = mf[mfi];        // Get the FAB to write to

        // Use amrex::LoopOnCpu for efficient iteration over the box
        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            // Perform bounds check against the original image dimensions
            // *** FIX 2 Applied: Remove AMREX_LIKELY ***
            if (i >= 0 && i < current_width && j >= 0 && j < current_height && k >= 0 && k < current_depth)
            {
                // Calculate 1D index (assuming XYZ layout where Z varies slowest)
                amrex::Long idx = static_cast<amrex::Long>(k) * current_width * current_height +
                                  static_cast<amrex::Long>(j) * current_width +
                                  static_cast<amrex::Long>(i);

                // Check calculated index against actual data size (safety check)
                // *** FIX 2 Applied: Remove AMREX_LIKELY *** (Implicitly, was likely here too)
                if (idx >= 0 && idx < raw_data_size)
                {
                    // Apply threshold condition (>) on the internal double data
                    // *** FIX 3 Applied: Use amrex::IntVect for fab access ***
                    fab(amrex::IntVect(i, j, k), 0) = (data_ptr[idx] > raw_threshold) ? value_if_true : value_if_false;
                } else {
                    // Should not happen if dimensions/logic are correct, but handle defensively
                    // *** FIX 3 Applied: Use amrex::IntVect for fab access ***
                    fab(amrex::IntVect(i, j, k), 0) = value_if_false;
                }
            } else {
                 // If the iMultiFab's valid box extends beyond the HDF5 image dimensions,
                 // fill the outside region with the 'false' value.
                 // *** FIX 3 Applied: Use amrex::IntVect for fab access ***
                 fab(amrex::IntVect(i, j, k), 0) = value_if_false;
            }
        });
    }
}

// Original overload (output 1/0) - calls the flexible version
void HDF5Reader::threshold(double raw_threshold, amrex::iMultiFab& mf) const
{
    threshold(raw_threshold, 1, 0, mf); // Call the flexible version with 1/0
}


} // namespace OpenImpala

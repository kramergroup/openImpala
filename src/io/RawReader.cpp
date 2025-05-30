#include "RawReader.H"

#include <fstream>   // For std::ifstream
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error
#include <cmath>     // For std::isnan, std::isinf (not currently used)
#include <limits>    // For std::numeric_limits // Ensure this is included
#include <algorithm> // For std::reverse
#include <cstring>   // For std::memcpy
#include <utility>   // For std::move

#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H> // Needed for the fix
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::LoopOnCpu
#include <AMReX_Extension.H>     // For AMREX_ALWAYS_ASSERT_WITH_MESSAGE
#include <AMReX_GpuQualifiers.H> // For AMREX_FORCE_INLINE
#include <AMReX_ParallelDescriptor.H> // For IOProcessor

namespace OpenImpala {

//-----------------------------------------------------------------------
// Helper Function & Static Variable for Host Endianness
//-----------------------------------------------------------------------

namespace { // Anonymous namespace for internal linkage

// Function to determine host endianness (called once)
bool determineHostEndianness() {
    std::uint16_t test_value = 1;
    unsigned char first_byte;
    std::memcpy(&first_byte, &test_value, 1);
    return (first_byte == 1);
}

// Cache the host endianness result once at startup
const static bool host_is_little_endian = determineHostEndianness();

// Helper function to reconstruct value from bytes, handling endianness
template <typename T>
AMREX_FORCE_INLINE T reconstructValue(const RawReader::ByteType* src_ptr, const size_t bytes_to_read, const bool needs_swap)
{
    static_assert(sizeof(T) <= 8, "reconstructValue: Type size mismatch or too large");
    RawReader::ByteType temp_buffer[8]; // Sufficient for up to 64-bit types

    if (needs_swap) {
         for (size_t b = 0; b < bytes_to_read; ++b) {
             temp_buffer[b] = src_ptr[bytes_to_read - 1 - b];
         }
    } else {
        std::memcpy(temp_buffer, src_ptr, bytes_to_read);
    }

    T val;
    // Ensure memcpy reads exactly sizeof(T) bytes, requires bytes_to_read == sizeof(T)
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes_to_read == sizeof(T), "reconstructValue: bytes_to_read does not match sizeof(T)");
    std::memcpy(&val, temp_buffer, sizeof(T));
    return val;
}


} // namespace

// Public helper method now just returns the cached value
bool RawReader::isHostLittleEndian() const {
    return host_is_little_endian;
}


//-----------------------------------------------------------------------
// Constructor Implementations
//-----------------------------------------------------------------------

RawReader::RawReader() :
    m_width(0), m_height(0), m_depth(0),
    m_data_type(RawDataType::UNKNOWN), m_is_read(false)
{
    // Default constructor: initialize members
}

// Constructor that reads file and throws on error
RawReader::RawReader(const std::string& filename,
                     int width, int height, int depth,
                     RawDataType data_type) :
    RawReader() // Delegate for default member initialization
{
    if (!readFile(filename, width, height, depth, data_type)) {
        // Error messages are printed within readFile/readRawFileInternal
        throw std::runtime_error("RawReader: Failed to read raw file '" + filename +
                                 "' during construction.");
    }
}

//-----------------------------------------------------------------------
// File Reading Implementation
//-----------------------------------------------------------------------

bool RawReader::readFile(const std::string& filename,
                         int width, int height, int depth,
                         RawDataType data_type)
{
    // --- Reset State ---
    m_raw_bytes.clear(); // Ensure vector is cleared even if readRawFileInternal fails early
    m_is_read = false;
    m_filename = filename;
    m_width = width;
    m_height = height;
    m_depth = depth;
    m_data_type = data_type;

    // --- Validate Inputs ---
    if (m_filename.empty()) {
        amrex::Print() << "Error: [RawReader] No filename provided.\n";
        return false;
    }
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0) {
        amrex::Print() << "Error: [RawReader] Invalid dimensions provided (W=" << m_width
                       << ", H=" << m_height << ", D=" << m_depth << "). Must be positive.\n";
        return false;
    }
    if (m_data_type == RawDataType::UNKNOWN) {
         amrex::Print() << "Error: [RawReader] Data type cannot be UNKNOWN.\n";
         return false;
    }

    // --- Perform Read ---
    // readRawFileInternal will print specific errors on failure
    bool success = readRawFileInternal();
    m_is_read = success;

    if (success) {
        // Only print success message on IOProcessor to reduce noise
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Successfully read Raw File: " << m_filename
                           << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth
                           << ", Type: " << static_cast<int>(m_data_type) // Simple type print
                           << ", HostLE: " << isHostLittleEndian() << ")\n";
        }
    } else {
        // Ensure state reflects failure if readRawFileInternal failed
        // Error message should have been printed by readRawFileInternal
        m_raw_bytes.clear(); // Clear potentially partially allocated/read data
        m_width = m_height = m_depth = 0; // Reset dimensions
        m_data_type = RawDataType::UNKNOWN; // Reset type
    }

    return success;
}


// <<< START OF CORRECTED readRawFileInternal (DEBUG prints removed) >>>
bool RawReader::readRawFileInternal()
{
    // --- Calculate Expected Size ---
    const size_t bytes_per_voxel = getBytesPerVoxel();
    if (bytes_per_voxel == 0) {
        // This case should ideally be caught by the UNKNOWN check in readFile
        amrex::Print() << "Internal Error: [RawReader] Invalid data type resulted in zero bytes per voxel.\n";
        return false;
    }

    // Use long long for intermediate calculation to avoid overflow before size_t check
    long long total_voxels = static_cast<long long>(m_width) * m_height * m_depth;
    long long expected_bytes_ll = total_voxels * bytes_per_voxel;

    if (expected_bytes_ll <= 0) {
        // This could happen if dimensions are large enough to overflow 'long long'
        amrex::Print() << "Error: [RawReader] Calculated data size (" << expected_bytes_ll << ") is zero or negative.\n";
        return false;
    }

    // --- Check against size_t max ---
    // Revised check: Compare expected_bytes_ll (known positive) as unsigned with size_t::max
    if (static_cast<unsigned long long>(expected_bytes_ll) > std::numeric_limits<size_t>::max())
    {
        amrex::Print() << "Error: [RawReader] Calculated data size (" << expected_bytes_ll
                       << ") exceeds maximum value representable by size_t ("
                       << std::numeric_limits<size_t>::max() << ").\n"; // Corrected error message
        return false;
    }
    // If the check passes, it's safe to cast to size_t.
    const size_t expected_bytes = static_cast<size_t>(expected_bytes_ll);

    // --- Open File ---
    std::ifstream file(m_filename, std::ios::binary | std::ios::ate); // Open at end to get size
    if (!file.is_open()) {
        amrex::Print() << "Error: [RawReader] Could not open file: " << m_filename << "\n";
        return false;
    }

    // --- Check File Size ---
    std::streamsize file_size = file.tellg();
    if (file_size < 0) {
        amrex::Print() << "Error: [RawReader] Could not determine file size or file size is negative for: " << m_filename << "\n";
        file.close();
        return false;
    }
    // Check if file size is smaller than expected (unsigned comparison needed)
    if (static_cast<unsigned long long>(file_size) < expected_bytes) {
        amrex::Print() << "Error: [RawReader] File size (" << file_size
                       << ") is smaller than expected data size (" << expected_bytes << ") for: "
                       << m_filename << "\n";
        file.close();
        return false;
    }
     // Check if file size is different (and issue warning if larger)
     if (static_cast<size_t>(file_size) != expected_bytes) {
        // Warning only on IOProcessor to avoid clutter
        if (amrex::ParallelDescriptor::IOProcessor()) {
            amrex::Print() << "Warning: [RawReader] File size (" << file_size
                           << ") does not exactly match expected data size (" << expected_bytes
                           << ") for: " << m_filename << ". Reading only expected bytes.\n";
        }
     }

    // --- Allocate Memory ---
    try {
        m_raw_bytes.resize(expected_bytes);
    } catch (const std::exception& e) { // Catch bad_alloc and length_error
        amrex::Print() << "Error: [RawReader] Failed to allocate " << expected_bytes
                       << " bytes for raw data: " << e.what() << "\n";
        file.close();
        return false;
    }

    // --- Read Data ---
    file.seekg(0, std::ios::beg); // Go back to the beginning to read
    if (!file.read(reinterpret_cast<char*>(m_raw_bytes.data()), expected_bytes)) {
        // Read failed, report stream state
        amrex::Print() << "Error: [RawReader] Failed to read " << expected_bytes << " bytes from file: " << m_filename << "\n";
        amrex::Print() << "  Stream state: good=" << file.good() << " eof=" << file.eof()
                       << " fail=" << file.fail() << " bad=" << file.bad() << "\n";
        file.close();
        m_raw_bytes.clear(); // Clear potentially partially read data
        return false;
    }

    file.close();

    // Endian handling is done during value reconstruction (getValue / threshold)

    return true; // Success!
}
// <<< END OF CORRECTED readRawFileInternal >>>


//-----------------------------------------------------------------------
// Helper Method Implementations
//-----------------------------------------------------------------------

size_t RawReader::getBytesPerVoxel() const
{
    // Assumes m_data_type is valid (checked in readFile)
    switch (m_data_type) {
        case RawDataType::UINT8:    case RawDataType::INT8:    return 1;
        case RawDataType::INT16_LE: case RawDataType::INT16_BE:
        case RawDataType::UINT16_LE:case RawDataType::UINT16_BE: return 2;
        case RawDataType::INT32_LE: case RawDataType::INT32_BE:
        case RawDataType::UINT32_LE:case RawDataType::UINT32_BE:
        case RawDataType::FLOAT32_LE:case RawDataType::FLOAT32_BE: return 4;
        case RawDataType::FLOAT64_LE:case RawDataType::FLOAT64_BE: return 8;
        case RawDataType::UNKNOWN: default: return 0; // Should not happen if called after validation
    }
}

//-----------------------------------------------------------------------
// Getter Implementations
//-----------------------------------------------------------------------
bool RawReader::isRead() const { return m_is_read; }
int RawReader::width() const { return m_width; }
int RawReader::height() const { return m_height; }
int RawReader::depth() const { return m_depth; }
RawDataType RawReader::getDataType() const { return m_data_type; }

amrex::Box RawReader::box() const {
    if (!m_is_read) return amrex::Box();
    // AMReX Box uses inclusive indices: [low_corner, high_corner]
    return amrex::Box(amrex::IntVect(0, 0, 0),
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}


//-----------------------------------------------------------------------
// Data Access Implementation (`getValue`)
//-----------------------------------------------------------------------

double RawReader::getValue(int i, int j, int k) const
{
    // Ensure data has been successfully read before accessing
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_read, "[RawReader::getValue] Data not read yet.");

    // --- Bounds Check ---
    // Use AMREX_ALWAYS_ASSERT for conditions that MUST be true for correctness
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        i >= 0 && i < m_width && j >= 0 && j < m_height && k >= 0 && k < m_depth,
        "[RawReader::getValue] Index (" + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)
        + ") out of bounds (W:" + std::to_string(m_width) + ", H:" + std::to_string(m_height)
        + ", D:" + std::to_string(m_depth) + ")."
    );

    // --- Calculate Byte Offset ---
    const size_t bytes_per_voxel = getBytesPerVoxel();
    // Ensure dimensions used for index calculation are positive
    AMREX_ALWAYS_ASSERT(m_width > 0 && m_height > 0 && m_depth > 0 && bytes_per_voxel > 0);

    // Use size_t for index calculation to match vector size type and prevent potential overflow
    // Assuming XYZ layout, Z varies slowest (k index)
    size_t plane_size = static_cast<size_t>(m_width) * static_cast<size_t>(m_height);
    size_t row_offset = static_cast<size_t>(j) * static_cast<size_t>(m_width);
    size_t idx_1d = static_cast<size_t>(k) * plane_size + row_offset + static_cast<size_t>(i);
    size_t offset = idx_1d * bytes_per_voxel;

    // --- Check Offset vs Buffer Size ---
    // Ensure the read won't go past the end of the allocated buffer
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        (offset + bytes_per_voxel) <= m_raw_bytes.size(), // Use <= because offset is 0-based
        "[RawReader::getValue] Calculated byte offset (" + std::to_string(offset)
        + ") + size (" + std::to_string(bytes_per_voxel)
        + ") exceeds raw data vector size (" + std::to_string(m_raw_bytes.size()) + ")."
    );

    // --- Determine Endianness and Need for Swap ---
    bool data_is_le;
    switch(m_data_type) {
        case RawDataType::INT16_LE: case RawDataType::UINT16_LE: case RawDataType::INT32_LE:
        case RawDataType::UINT32_LE: case RawDataType::FLOAT32_LE: case RawDataType::FLOAT64_LE:
            data_is_le = true; break;
        case RawDataType::INT16_BE: case RawDataType::UINT16_BE: case RawDataType::INT32_BE:
        case RawDataType::UINT32_BE: case RawDataType::FLOAT32_BE: case RawDataType::FLOAT64_BE:
            data_is_le = false; break;
        default: // UINT8, INT8, UNKNOWN (although UNKNOWN shouldn't happen here)
            // For single-byte types, endianness doesn't matter for swapping logic
            data_is_le = host_is_little_endian; // Assign something for consistency
            break;
    }
    // Need swap only if bytes > 1 AND data endianness differs from host
    const bool needs_swap = (bytes_per_voxel > 1) && (data_is_le != host_is_little_endian);

    // --- Read Bytes (Potentially Swap) and Reconstruct ---
    double result = 0.0;
    // Pointer to the start of the relevant bytes in the raw data vector
    const ByteType* src_ptr = m_raw_bytes.data() + offset;

    // Use the helper function to reconstruct the value based on type
    switch (m_data_type) {
        case RawDataType::UINT8: {
            result = static_cast<double>(reconstructValue<uint8_t>(src_ptr, 1, needs_swap)); break; }
        case RawDataType::INT8: {
            result = static_cast<double>(reconstructValue<int8_t>(src_ptr, 1, needs_swap)); break; }
        case RawDataType::UINT16_LE: case RawDataType::UINT16_BE: {
            result = static_cast<double>(reconstructValue<uint16_t>(src_ptr, 2, needs_swap)); break; }
        case RawDataType::INT16_LE: case RawDataType::INT16_BE: {
            result = static_cast<double>(reconstructValue<int16_t>(src_ptr, 2, needs_swap)); break; }
        case RawDataType::UINT32_LE: case RawDataType::UINT32_BE: {
            result = static_cast<double>(reconstructValue<uint32_t>(src_ptr, 4, needs_swap)); break; }
        case RawDataType::INT32_LE: case RawDataType::INT32_BE: {
            result = static_cast<double>(reconstructValue<int32_t>(src_ptr, 4, needs_swap)); break; }
        case RawDataType::FLOAT32_LE: case RawDataType::FLOAT32_BE: {
            static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "Require IEEE 754 32-bit float");
            result = static_cast<double>(reconstructValue<float>(src_ptr, 4, needs_swap)); break; }
        case RawDataType::FLOAT64_LE: case RawDataType::FLOAT64_BE: {
            static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8, "Require IEEE 754 64-bit double");
            result = reconstructValue<double>(src_ptr, 8, needs_swap); break; } // Direct assignment for double
        case RawDataType::UNKNOWN: default:
            // Should be caught by validation in readFile, but include for safety
            amrex::Abort("[RawReader::getValue] Unknown or unsupported data type encountered.");
    }
    return result;
}


//-----------------------------------------------------------------------
// Threshold Implementation (Corrected)
//-----------------------------------------------------------------------

// Overload with customizable true/false values
void RawReader::threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_read, "[RawReader::threshold] Cannot threshold, data not read successfully.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.nComp() == 1, "[RawReader::threshold] Output iMultiFab must have 1 component.");

    // Capture members needed by the lambda by value or const reference/pointer for safety and potential performance
    const int current_width = m_width;
    const int current_height = m_height;
    const int current_depth = m_depth;
    const RawDataType current_data_type = m_data_type;
    const size_t bytes_per_voxel = getBytesPerVoxel();
    // Ensure bytes_per_voxel is valid before proceeding (should be checked in readFile)
    AMREX_ALWAYS_ASSERT(bytes_per_voxel > 0);

    const size_t raw_bytes_size = m_raw_bytes.size();
    const ByteType* const raw_bytes_ptr = m_raw_bytes.data(); // Get raw data pointer once

    // Determine if data needs swapping based on type and cached host endianness
    bool data_is_le;
    switch(current_data_type) { /* Same switch logic as getValue */
        case RawDataType::INT16_LE: case RawDataType::UINT16_LE: case RawDataType::INT32_LE:
        case RawDataType::UINT32_LE: case RawDataType::FLOAT32_LE: case RawDataType::FLOAT64_LE:
            data_is_le = true; break;
        case RawDataType::INT16_BE: case RawDataType::UINT16_BE: case RawDataType::INT32_BE:
        case RawDataType::UINT32_BE: case RawDataType::FLOAT32_BE: case RawDataType::FLOAT64_BE:
            data_is_le = false; break;
        default:
            data_is_le = host_is_little_endian; break;
    }
    const bool needs_swap = (bytes_per_voxel > 1) && (data_is_le != host_is_little_endian);

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    // Iterate over the iMultiFab using MFIter
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox(); // Get the valid box for this FAB
        amrex::IArrayBox& fab = mf[mfi];       // Get the FArrayBox data (non-const ref)

        // Loop over the cells in the box using amrex::LoopOnCpu (since m_raw_bytes is host memory)
        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            // --- Bounds Check for Raw Data Access ---
            // Check if the current index (i,j,k) is within the bounds of the loaded raw data dimensions
            if (i >= 0 && i < current_width && j >= 0 && j < current_height && k >= 0 && k < current_depth)
            {
                // --- Calculate Offset ---
                // Use size_t consistently for indexing calculations to prevent overflow
                // Ensure dimensions are positive (checked in readFile)
                size_t plane_size = static_cast<size_t>(current_width) * static_cast<size_t>(current_height);
                size_t row_offset = static_cast<size_t>(j) * static_cast<size_t>(current_width);
                size_t idx_1d = static_cast<size_t>(k) * plane_size + row_offset + static_cast<size_t>(i);
                size_t offset = idx_1d * bytes_per_voxel;

                // --- Check Calculated Offset against raw buffer size ---
                if ((offset + bytes_per_voxel) <= raw_bytes_size) {

                    // --- Inline Reconstruction & Comparison ---
                    // Point to the start of the bytes for this voxel
                    const ByteType* src_ptr = raw_bytes_ptr + offset;
                    bool comparison_result = false;

                    // Switch on type to reconstruct value and compare efficiently
                    // This avoids calling the full getValue function (with its checks) for every voxel
                    switch (current_data_type) {
                        case RawDataType::UINT8: {
                            uint8_t val = reconstructValue<uint8_t>(src_ptr, 1, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::INT8: {
                            int8_t val = reconstructValue<int8_t>(src_ptr, 1, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::UINT16_LE: case RawDataType::UINT16_BE: {
                            uint16_t val = reconstructValue<uint16_t>(src_ptr, 2, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::INT16_LE: case RawDataType::INT16_BE: {
                            int16_t val = reconstructValue<int16_t>(src_ptr, 2, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::UINT32_LE: case RawDataType::UINT32_BE: {
                            uint32_t val = reconstructValue<uint32_t>(src_ptr, 4, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::INT32_LE: case RawDataType::INT32_BE: {
                            int32_t val = reconstructValue<int32_t>(src_ptr, 4, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::FLOAT32_LE: case RawDataType::FLOAT32_BE: {
                            static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "");
                            float val = reconstructValue<float>(src_ptr, 4, needs_swap);
                            comparison_result = (static_cast<double>(val) > threshold_value); break; }
                        case RawDataType::FLOAT64_LE: case RawDataType::FLOAT64_BE: {
                            static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8, "");
                            double val = reconstructValue<double>(src_ptr, 8, needs_swap);
                            comparison_result = (val > threshold_value); break; }
                        case RawDataType::UNKNOWN: default:
                            // This state should ideally be impossible if m_is_read is true
                            amrex::Abort("[RawReader::threshold] Unsupported data type encountered in loop.");
                    }
                    // Assign result to the iMultiFab using IntVect for indexing
                    fab(amrex::IntVect(i, j, k), 0) = comparison_result ? value_if_true : value_if_false;

                } else {
                    // Offset check failed - This indicates a logic error (e.g., in index calculation) or data corruption.
                    // Abort is safer than potentially reading garbage or setting incorrect values silently.
                    amrex::Abort("[RawReader::threshold] Internal error: Calculated offset exceeds bounds.");
                    // If choosing to continue instead: fab(amrex::IntVect(i, j, k), 0) = value_if_false;
                }
            } else {
                 // Index (i,j,k) from the iMultiFab box is outside the raw data bounds.
                 // Set to the 'false' value as per the function description.
                 fab(amrex::IntVect(i, j, k), 0) = value_if_false;
            }
        }); // End of amrex::LoopOnCpu lambda
    } // End of MFIter loop
} // End of threshold function


// Original overload (output 1/0) - calls the flexible version
void RawReader::threshold(double threshold_value, amrex::iMultiFab& mf) const
{
    threshold(threshold_value, 1, 0, mf); // Call the flexible version with 1/0
}


} // namespace OpenImpala

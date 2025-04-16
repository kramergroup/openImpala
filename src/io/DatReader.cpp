#include "DatReader.H"
#include <AMReX_Utility.H> // <-- MOVE HERE
#include <cstddef>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <limits>    // For std::numeric_limits
#include <cstdint>   // For std::int32_t, std::uint16_t (via DataType)

#include <AMReX.H>
#include <AMReX_Print.H> // For amrex::Print, amrex::Warning
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::LoopOnCpu

namespace OpenImpala {

// --- Constructor Implementations ---

DatReader::DatReader() :
    m_width(0), m_height(0), m_depth(0), m_is_read(false)
{
    // Default constructor: initializes members to default state.
}

DatReader::DatReader(const std::string& filename) :
    DatReader() // Delegate to default constructor for initialization
{
    // Attempt to read the file. readFile prints errors on failure.
    if (!readFile(filename)) {
        // Construct the error message *after* readFile attempted and potentially printed details.
        throw std::runtime_error("DatReader: Failed to read file: " + filename);
    }
}

// --- File Reading Implementation ---

bool DatReader::readFile(const std::string& filename)
{
    // Reset state before reading
    m_filename = filename;
    m_raw.clear();
    m_width = 0;
    m_height = 0;
    m_depth = 0;
    m_is_read = false;

    std::ifstream ifs(m_filename, std::ios::binary | std::ios::ate); // Open in binary, start at end

    if (!ifs.is_open()) {
        amrex::Print() << "Error: [DatReader] Could not open file: " << m_filename << "\n";
        return false;
    }

    // Get file size
    std::streamsize file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg); // Go back to beginning

    // --- Read Header (ASSUMPTION: 3 x int32_t for W, H, D) ---
    constexpr std::streamsize header_size = 3 * sizeof(std::int32_t);
    if (file_size < header_size) {
        amrex::Print() << "Error: [DatReader] File too small to contain header: " << m_filename
                       << " (Size: " << file_size << " bytes)\n";
        return false;
    }

    std::int32_t dims[3] = {0, 0, 0}; // W, H, D
    ifs.read(reinterpret_cast<char*>(dims), header_size);

    if (!ifs.good()) {
        amrex::Print() << "Error: [DatReader] Failed reading header from file: " << m_filename << "\n";
        return false;
    }

    // --- Handle Endianness for Header (ASSUMPTION: File is Little Endian) ---
    if (amrex::isBigEndian()) {
        amrex::SwapBytes(dims[0]);
        amrex::SwapBytes(dims[1]);
        amrex::SwapBytes(dims[2]);
    }

    // Check dimensions
    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
         amrex::Print() << "Error: [DatReader] Invalid dimensions read from header in " << m_filename
                        << " (W=" << dims[0] << ", H=" << dims[1] << ", D=" << dims[2] << ")\n";
         return false;
    }

    m_width = static_cast<int>(dims[0]);
    m_height = static_cast<int>(dims[1]);
    m_depth = static_cast<int>(dims[2]);

    // --- Read Voxel Data ---
    amrex::Long num_voxels = static_cast<amrex::Long>(m_width) * m_height * m_depth; // Use Long
    constexpr std::streamsize element_size = sizeof(DataType);
    std::streamsize expected_data_size = num_voxels * element_size;
    std::streamsize actual_data_size = file_size - header_size;

    if (actual_data_size < expected_data_size) {
         amrex::Print() << "Error: [DatReader] File size mismatch in " << m_filename
                        << ". Expected data size: " << expected_data_size
                        << " bytes, Available: " << actual_data_size << " bytes.\n";
         return false;
    }
    // Optionally warn if actual_data_size > expected_data_size (extra data ignored)
    if (actual_data_size > expected_data_size) {
         amrex::Warning("Warning: [DatReader] File contains more data than expected based on header dimensions. Ignoring extra data.");
    }


    try {
        // Check for potential overflow before resize (unlikely if it fits in memory)
        if (num_voxels > std::numeric_limits<size_t>::max()) {
             throw std::overflow_error("DatReader: Calculated number of voxels exceeds vector size limit.");
        }
        m_raw.resize(static_cast<size_t>(num_voxels));
    } catch (const std::exception& e) {
        amrex::Print() << "Error: [DatReader] Failed to allocate memory for " << num_voxels
                       << " voxels: " << e.what() << "\n";
        return false;
    }

    ifs.read(reinterpret_cast<char*>(m_raw.data()), expected_data_size);

    if (!ifs.good() && ifs.gcount() != expected_data_size) {
         amrex::Print() << "Error: [DatReader] Failed reading voxel data from file: " << m_filename
                        << ". Read " << ifs.gcount() << " bytes, expected " << expected_data_size << ".\n";
         m_raw.clear(); // Clear potentially partial data
         return false;
    }

    // --- Handle Endianness for Data (ASSUMPTION: File is Little Endian) ---
    if (amrex::isBigEndian()) {
        for (DataType& val : m_raw) {
            amrex::SwapBytes(val);
        }
    }

    m_is_read = true;
    amrex::Print() << "Successfully read DAT file: " << m_filename
                   << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}


// --- Getter Implementations ---

int DatReader::width() const {
    return m_width;
}

int DatReader::height() const {
    return m_height;
}

int DatReader::depth() const {
    return m_depth;
}

amrex::Box DatReader::box() const {
    if (!m_is_read) {
        return amrex::Box(); // Return empty box if not read
    }
    // Box is cell-centered, index from 0 to dim-1
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

const std::vector<DatReader::DataType>& DatReader::getRawData() const {
    return m_raw;
}

DatReader::DataType DatReader::getRawValue(int i, int j, int k) const {
    if (!m_is_read) {
        throw std::out_of_range("[DatReader::getRawValue] Data not read yet.");
    }
    if (i < 0 || i >= m_width || j < 0 || j >= m_height || k < 0 || k >= m_depth) {
        // Consider using amrex::Abort for fatal errors in parallel runs?
        throw std::out_of_range("[DatReader::getRawValue] Index ("
                                + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)
                                + ") out of bounds (W:" + std::to_string(m_width)
                                + ", H:" + std::to_string(m_height) + ", D:" + std::to_string(m_depth) + ").");
    }

    // Calculate 1D index (assuming XYZ layout where Z varies slowest)
    // Use amrex::Long for intermediate calculations to avoid overflow
    amrex::Long idx = static_cast<amrex::Long>(k) * m_width * m_height +
                      static_cast<amrex::Long>(j) * m_width +
                      static_cast<amrex::Long>(i);

    // This check should ideally not be needed if dimensions*sizeof(DataType) fits memory,
    // but provides extra safety against calculation errors or unexpected vector states.
    if (idx >= static_cast<amrex::Long>(m_raw.size())) {
         throw std::out_of_range("[DatReader::getRawValue] Calculated index exceeds raw data vector size.");
    }

    return m_raw[static_cast<size_t>(idx)];
}

// --- Threshold Implementation ---

// Overload with customizable true/false values
void DatReader::threshold(DataType raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        amrex::Abort("[DatReader::threshold] Cannot threshold, data not read successfully.");
        return; // Or throw? Abort is safer in parallel AMReX context
    }

    // Get raw pointer for potentially faster access inside loop
    // Ensure vector isn't empty (already checked by m_is_read indirectly)
    const DataType* const AMREX_RESTRICT data_ptr = m_raw.data();

    // Pre-calculate for index bounds checking inside loop
    const amrex::Long raw_data_size = static_cast<amrex::Long>(m_raw.size());
    const int current_width = m_width;   // Copy to local const for lambda capture
    const int current_height = m_height;
    const int current_depth = m_depth;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        amrex::IArrayBox& fab = mf[mfi];

        // Use amrex::LoopOnCpu for efficient iteration
        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            // Perform bounds check against the original image dimensions
            if (i >= 0 && i < current_width && j >= 0 && j < current_height && k >= 0 && k < current_depth)
            {
                // Calculate 1D index (assuming XYZ layout where Z varies slowest)
                amrex::Long idx = static_cast<amrex::Long>(k) * current_width * current_height +
                                  static_cast<amrex::Long>(j) * current_width +
                                  static_cast<amrex::Long>(i);

                // Check calculated index against actual data size
                if (idx >= 0 && idx < raw_data_size) // idx should always be >= 0 here
                {
                    // Apply threshold condition (>)
                    fab(amrex::IntVect(i, j, k), 0) = (data_ptr[idx] > raw_threshold) ? value_if_true : value_if_false;
                } else {
                    // Voxel (i,j,k) is within dimensions but calculated index is bad (should not happen if logic is correct)
                    // OR voxel is outside original dimensions covered by mf Box (handled below)
                    fab(amrex::IntVect(i, j, k), 0) = value_if_false; // Or some error value? Defaulting to false seems reasonable.
                    // Consider adding a warning here if idx check fails unexpectedly
                    // amrex::Warning("Index calculation error in threshold");
                }
            } else {
                 // The box associated with this fab extends beyond the original image dimensions
                 fab(amrex::IntVect(i, j, k), 0) = value_if_false; // Assign 'false' value to regions outside image
            }
        });
    }
}

// Original overload (output 1/0) - calls the flexible version
void DatReader::threshold(DataType raw_threshold, amrex::iMultiFab& mf) const
{
    threshold(raw_threshold, 1, 0, mf);
}


} // namespace OpenImpala

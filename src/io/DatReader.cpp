#include "DatReader.H"
#include <AMReX_Utility.H> // Keep moved or remove if not needed elsewhere
#include <cstddef>
#include <fstream>
#include <vector>
#include <string>
#include <stdexcept> // For std::runtime_error, std::out_of_range
#include <limits>    // For std::numeric_limits
#include <cstdint>   // For std::int32_t, std::uint16_t (via DataType)
// Include for compiler endianness macros (often implicitly available with GCC)
#include <endian.h>  // Or rely on compiler built-ins

#include <AMReX.H>
#include <AMReX_Print.H> // For amrex::Print, amrex::Warning
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::LoopOnCpu


namespace OpenImpala {

// --- Helper functions for byte swapping using GCC builtins ---
namespace { // Anonymous namespace

inline std::int32_t swap_bytes_int32(std::int32_t val) {
    return __builtin_bswap32(val);
}

inline std::uint16_t swap_bytes_uint16(std::uint16_t val) {
    // Assuming DataType is uint16_t based on DatReader.H
    // Use appropriate bswap if DataType changes!
    static_assert(sizeof(DatReader::DataType) == sizeof(std::uint16_t), "Swap logic assumes uint16_t");
    return __builtin_bswap16(val);
}

// Simple compile-time endian check using standard macros
constexpr bool is_system_big_endian() {
#if defined(__BYTE_ORDER__) && defined(__ORDER_BIG_ENDIAN__)
    return __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__;
#else
    // Fallback or error if macros not defined - might need runtime check instead
    // Forcing little-endian assumption here if macros missing, add warning.
    #warning "Could not determine endianness via macros, assuming little-endian."
    return false; // Assume little-endian if unsure
#endif
}

} // end anonymous namespace


// --- Constructor Implementations ---

DatReader::DatReader() :
    m_width(0), m_height(0), m_depth(0), m_is_read(false)
{
    // Default constructor: initializes members to default state.
}

DatReader::DatReader(const std::string& filename) :
    DatReader() // Delegate to default constructor for initialization
{
    if (!readFile(filename)) {
        throw std::runtime_error("DatReader: Failed to read file: " + filename);
    }
}

// --- File Reading Implementation ---

bool OpenImpala::DatReader::isRead() const {
    return m_is_read; // Return the state of your internal flag
}

bool DatReader::readFile(const std::string& filename)
{
    m_filename = filename;
    m_raw.clear();
    m_width = 0; m_height = 0; m_depth = 0;
    m_is_read = false;

    std::ifstream ifs(m_filename, std::ios::binary | std::ios::ate);
    if (!ifs.is_open()) {
        amrex::Print() << "Error: [DatReader] Could not open file: " << m_filename << "\n";
        return false;
    }

    std::streamsize file_size = ifs.tellg();
    ifs.seekg(0, std::ios::beg);

    constexpr std::streamsize header_size = 3 * sizeof(std::int32_t);
    if (file_size < header_size) {
        amrex::Print() << "Error: [DatReader] File too small for header: " << m_filename << "\n";
        return false;
    }

    std::int32_t dims[3] = {0, 0, 0};
    ifs.read(reinterpret_cast<char*>(dims), header_size);
    if (!ifs.good()) {
        amrex::Print() << "Error: [DatReader] Failed reading header: " << m_filename << "\n";
        return false;
    }

    // --- Handle Endianness for Header (ASSUMPTION: File is Little Endian) ---
    // <<< Use workaround instead of amrex::isBigEndian / SwapBytes >>>
    if (is_system_big_endian()) { // Check system endianness
        // If system is big-endian, swap bytes read from little-endian file
        dims[0] = swap_bytes_int32(dims[0]);
        dims[1] = swap_bytes_int32(dims[1]);
        dims[2] = swap_bytes_int32(dims[2]);
    }

    if (dims[0] <= 0 || dims[1] <= 0 || dims[2] <= 0) {
         amrex::Print() << "Error: [DatReader] Invalid dimensions in header: " << m_filename
                        << " (W=" << dims[0] << ", H=" << dims[1] << ", D=" << dims[2] << ")\n";
         return false;
    }
    m_width = static_cast<int>(dims[0]);
    m_height = static_cast<int>(dims[1]);
    m_depth = static_cast<int>(dims[2]);

    // --- Read Voxel Data ---
    amrex::Long num_voxels = static_cast<amrex::Long>(m_width) * m_height * m_depth;
    constexpr std::streamsize element_size = sizeof(DataType);
    std::streamsize expected_data_size = num_voxels * element_size;
    std::streamsize actual_data_size = file_size - header_size;

    if (actual_data_size < expected_data_size) {
         amrex::Print() << "Error: [DatReader] File size mismatch: " << m_filename
                        << ". Expected data: " << expected_data_size
                        << " bytes, Available: " << actual_data_size << " bytes.\n";
         return false;
    }
    if (actual_data_size > expected_data_size) {
         std::string msg = "Warning: [DatReader] File contains more data than expected. Ignoring extra data.";
         amrex::Warning(msg.c_str()); // <<< FIX: Use .c_str() for safety >>>
    }

    try {
        if (static_cast<size_t>(num_voxels) > std::numeric_limits<size_t>::max()) {
             throw std::overflow_error("DatReader: Voxel count exceeds vector size limit.");
        }
        m_raw.resize(static_cast<size_t>(num_voxels));
    } catch (const std::exception& e) {
        amrex::Print() << "Error: [DatReader] Failed memory allocation for " << num_voxels
                       << " voxels: " << e.what() << "\n";
        return false;
    }

    ifs.read(reinterpret_cast<char*>(m_raw.data()), expected_data_size);
    if (!ifs.good() && ifs.gcount() != expected_data_size) {
         amrex::Print() << "Error: [DatReader] Failed reading voxel data: " << m_filename
                        << ". Read " << ifs.gcount() << "/" << expected_data_size << " bytes.\n";
         m_raw.clear();
         return false;
    }

    // --- Handle Endianness for Data (ASSUMPTION: File is Little Endian) ---
    // <<< Use workaround instead of amrex::isBigEndian / SwapBytes >>>
    if (is_system_big_endian()) { // Check system endianness
        // If system is big-endian, swap bytes read from little-endian file
        for (DataType& val : m_raw) {
            // Assuming DataType is uint16_t here based on header!
            val = swap_bytes_uint16(val);
        }
    }

    m_is_read = true;
    amrex::Print() << "Successfully read DAT file: " << m_filename
                   << " (Dimensions: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}


// --- Getter Implementations ---
int DatReader::width() const { return m_width; }
int DatReader::height() const { return m_height; }
int DatReader::depth() const { return m_depth; }
amrex::Box DatReader::box() const {
    if (!m_is_read) return amrex::Box();
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}
const std::vector<DatReader::DataType>& DatReader::getRawData() const { return m_raw; }

DatReader::DataType DatReader::getRawValue(int i, int j, int k) const {
    if (!m_is_read) {
        throw std::out_of_range("[DatReader::getRawValue] Data not read yet.");
    }
    if (i < 0 || i >= m_width || j < 0 || j >= m_height || k < 0 || k >= m_depth) {
        throw std::out_of_range("[DatReader::getRawValue] Index ("
                                + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)
                                + ") out of bounds (W:" + std::to_string(m_width)
                                + ", H:" + std::to_string(m_height) + ", D:" + std::to_string(m_depth) + ").");
    }
    amrex::Long idx = static_cast<amrex::Long>(k) * m_width * m_height +
                      static_cast<amrex::Long>(j) * m_width +
                      static_cast<amrex::Long>(i);
    if (idx >= static_cast<amrex::Long>(m_raw.size())) {
         throw std::out_of_range("[DatReader::getRawValue] Calculated index exceeds raw data vector size.");
    }
    return m_raw[static_cast<size_t>(idx)];
}

// --- Threshold Implementation ---
void DatReader::threshold(DataType raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    if (!m_is_read) {
        amrex::Abort("[DatReader::threshold] Cannot threshold, data not read successfully.");
    }
    const DataType* const AMREX_RESTRICT data_ptr = m_raw.data();
    const amrex::Long raw_data_size = static_cast<amrex::Long>(m_raw.size());
    const int current_width = m_width;
    const int current_height = m_height;
    const int current_depth = m_depth;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        amrex::IArrayBox& fab = mf[mfi];
        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            if (i >= 0 && i < current_width && j >= 0 && j < current_height && k >= 0 && k < current_depth)
            {
                amrex::Long idx = static_cast<amrex::Long>(k) * current_width * current_height +
                                  static_cast<amrex::Long>(j) * current_width +
                                  static_cast<amrex::Long>(i);
                if (idx >= 0 && idx < raw_data_size)
                {
                    fab(amrex::IntVect(i, j, k), 0) = (data_ptr[idx] > raw_threshold) ? value_if_true : value_if_false;
                } else {
                    fab(amrex::IntVect(i, j, k), 0) = value_if_false;
                }
            } else {
                 fab(amrex::IntVect(i, j, k), 0) = value_if_false;
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

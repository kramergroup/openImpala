#include "TiffReader.H" // Ensure this header is updated (needs m_fill_order member)!

#include <tiffio.h>  // libtiff C API Header
#include <memory>    // For std::unique_ptr
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <sstream>   // Needed for std::ostringstream
#include <cstring>   // For std::memcpy (in helper)
#include <algorithm> // For std::min/max
#include <iomanip>   // For std::setw, std::setfill
#include <map>       // For attributes (if keeping)

#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H> // For amrex::The_Pinned_Arena? Optional.
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Utility.H>       // Included for AMREX_ALWAYS_ASSERT
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Bcast
#include <AMReX_MFIter.H>       // Needed for MFIter
#include <AMReX_Array4.H>       // Needed for Array4 access
#include <AMReX_Loop.H>         // For LoopOnCpu / amrex::ParallelFor


namespace OpenImpala {

// Anonymous namespace for internal helpers
namespace {

// RAII wrapper for TIFF* handle (same as before)
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// Helper to interpret raw bytes based on TIFF metadata (for BPS >= 8)
// Returns value cast to double for threshold comparison
// IMPORTANT: Assumes byte_ptr points to the START of the pixel/sample data
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double interpretBytesAsDouble(const unsigned char* byte_ptr,
                                uint16_t bits_per_sample,
                                uint16_t sample_format)
{
    size_t bytes_per_sample = bits_per_sample / 8; // Assumes BPS >= 8 here
    double value = 0.0;
    if (bytes_per_sample == 0) return 0.0; // Should not happen if called correctly

    switch (sample_format) {
        case SAMPLEFORMAT_UINT:
            if (bytes_per_sample == 1) {
                uint8_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 2) {
                uint16_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 4) {
                uint32_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 8) {
                uint64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            }
            break;
        case SAMPLEFORMAT_INT:
            if (bytes_per_sample == 1) {
                int8_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 2) {
                int16_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 4) {
                int32_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 8) {
                int64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            }
            break;
        case SAMPLEFORMAT_IEEEFP:
            if (bytes_per_sample == 4) {
                float val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 8) {
                double val; std::memcpy(&val, byte_ptr, sizeof(val)); value = val;
            }
            break;
        default:
            // Consider adding a warning or error for unsupported sample formats if necessary
            value = 0.0;
            break;
    }
    return value;
}

// Helper to generate filename for sequences
std::string generateFilename(const std::string& base, int index, int digits, const std::string& suffix) {
     std::ostringstream ss;
     ss << base << std::setw(digits) << std::setfill('0') << index << suffix;
     return ss.str();
}


} // namespace


// --- Constructors ---
TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0),
    m_bits_per_sample(0), m_sample_format(0), m_samples_per_pixel(0),
    m_fill_order(FILLORDER_MSB2LSB), // Initialize fill order default
    m_is_read(false), m_is_sequence(false), m_start_index(0), m_digits(1)
{}

TiffReader::TiffReader(const std::string& filename) : TiffReader()
{
    m_filename = filename;
    if (!readFile(filename)) {
         throw std::runtime_error("TiffReader: Failed to read metadata from file: " + filename);
    }
}

TiffReader::TiffReader(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix) : TiffReader()
{
    m_base_pattern = base_pattern;
    m_start_index = start_index;
    m_digits = digits;
    m_suffix = suffix;
    if (!readFileSequence(base_pattern, num_files, start_index, digits, suffix)) {
         throw std::runtime_error("TiffReader: Failed to read metadata for sequence: " + base_pattern);
    }
}

// --- Metadata Getters ---
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) { return amrex::Box(); }
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

// --- readFile (Metadata Only) ---
// (Implementation as provided - assumes Abort calls within are sufficient for metadata stage)
bool TiffReader::readFile(const std::string& filename)
{
    m_filename = filename;
    m_is_sequence = false;
    m_base_pattern = "";

    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    uint16_t fill_order_r0 = FILLORDER_MSB2LSB; // Default fill order

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (filename.empty()) { amrex::Abort("[TiffReader::readFile] Filename cannot be empty."); }
        TiffPtr tif(TIFFOpen(filename.c_str(), "r"), TiffCloser());
        if (!tif) { amrex::Abort("[TiffReader::readFile] Failed to open TIFF file: " + filename); }
        uint32_t w32 = 0, h32 = 0; uint16_t planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { amrex::Abort("[TiffReader::readFile] Failed to get image dimensions from: " + filename); }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
        // *** Read FillOrder ***
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);

        width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32);
        bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) { amrex::Abort("[TiffReader::readFile] Invalid or unsupported TIFF format in: " + filename + " (Check dimensions, BPS, PlanarConfig)"); }
        depth_r0 = 0;
        // Check if setting directory 0 works before looping
        if (!TIFFSetDirectory(tif.get(), 0)) { amrex::Abort("[TiffReader::readFile] Failed to set initial directory (0) in: " + filename); }
        do { depth_r0++; } while (TIFFReadDirectory(tif.get()));
        if (depth_r0 == 0) { amrex::Abort("[TiffReader::readFile] Could not read any directories (depth is zero) in: " + filename); }
    }

    // Broadcast integer/flag data (Add fill_order_r0)
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                               static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                               static_cast<int>(m_is_sequence), static_cast<int>(fill_order_r0)}; // Added fill_order
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // Set members from broadcast
    m_width               = idata[0]; m_height              = idata[1]; m_depth               = idata[2];
    m_bits_per_sample     = static_cast<uint16_t>(idata[3]); m_sample_format       = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]); m_is_sequence         = static_cast<bool>(idata[6]);
    m_fill_order          = static_cast<uint16_t>(idata[7]); // Set fill_order

    // Broadcast filename manually
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_filename.length());
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root);
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        m_filename.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root);
    }

    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0) {
        amrex::Abort("TiffReader::readFile: Invalid metadata received after broadcast.");
    }
    m_is_read = true;
    return true;
}

// --- readFileSequence (Metadata Only) ---
// (Implementation as provided - assumes Abort calls within are sufficient for metadata stage)
bool TiffReader::readFileSequence(
    const std::string& base_pattern, int num_files, int start_index, int digits, const std::string& suffix)
{
    m_base_pattern = base_pattern; m_start_index = start_index; m_digits = digits; m_suffix = suffix;
    m_is_sequence = true; m_filename = "";

    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    uint16_t fill_order_r0 = FILLORDER_MSB2LSB; // Default fill order
    std::string first_filename_r0 = "";

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (num_files <= 0 || digits <= 0 || base_pattern.empty()) { amrex::Abort("[TiffReader::readFileSequence] Invalid sequence parameters (num_files, digits, base_pattern)."); }
         depth_r0 = num_files;
         first_filename_r0 = generateFilename(base_pattern, start_index, digits, suffix);
         TiffPtr tif(TIFFOpen(first_filename_r0.c_str(), "r"), TiffCloser());
         if (!tif) { amrex::Abort("[TiffReader::readFileSequence] Failed to open first sequence file: " + first_filename_r0); }
         uint32_t w32 = 0, h32 = 0; uint16_t planar = PLANARCONFIG_CONTIG;
         if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { amrex::Abort("[TiffReader::readFileSequence] Failed to get image dimensions from: " + first_filename_r0); }
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
         // *** Read FillOrder ***
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);
         width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32);
         bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
         if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) { amrex::Abort("[TiffReader::readFileSequence] Invalid or unsupported TIFF format in: " + first_filename_r0 + " (Check dimensions, BPS, PlanarConfig)"); }
    }

    // Broadcast integer/flag data (Add fill_order_r0)
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                               static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                               static_cast<int>(m_is_sequence), m_start_index, m_digits,
                               static_cast<int>(fill_order_r0)}; // Added fill_order
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // Set members from broadcast
    m_width               = idata[0]; m_height              = idata[1]; m_depth               = idata[2];
    m_bits_per_sample     = static_cast<uint16_t>(idata[3]); m_sample_format       = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]); m_is_sequence         = static_cast<bool>(idata[6]);
    m_start_index         = idata[7]; m_digits              = idata[8];
    m_fill_order          = static_cast<uint16_t>(idata[9]); // Set fill_order

    // Broadcast string parameters manually
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_base_pattern.length());
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root);
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        m_base_pattern.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root);
    }
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_suffix.length());
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root);
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        m_suffix.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root);
    }

    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || (m_is_sequence && m_base_pattern.empty())) {
        amrex::Abort("TiffReader::readFileSequence: Invalid metadata received after broadcast.");
    }
    m_is_read = true;
    return true;
}


// --- readDistributedIntoFab Method ---
void TiffReader::readDistributedIntoFab(
    amrex::iMultiFab& dest_mf,
    int value_if_true,
    int value_if_false,
    double raw_threshold
) const
{
    if (!m_is_read) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Cannot read, metadata not processed successfully.");
    }

    const int bits_per_sample_val = m_bits_per_sample;
    const size_t bytes_per_pixel = (bits_per_sample_val < 8) ?
                                   (bits_per_sample_val * m_samples_per_pixel + 7) / 8
                                   : (bits_per_sample_val / 8) * m_samples_per_pixel;
    if (bits_per_sample_val == 0 || bytes_per_pixel == 0) { // Added bytes_per_pixel check
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bits per sample or calculated bytes per pixel is zero!");
    }

    // Open the single stack file handle once outside the parallel region if needed.
    TiffPtr shared_tif_stack_handle = nullptr;
    if (!m_is_sequence) {
        shared_tif_stack_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
        if (!shared_tif_stack_handle) {
            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open shared TIFF file: " + m_filename);
        }
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        // Each thread gets its own buffer and potentially its own file handle for sequences
        std::vector<unsigned char> temp_buffer;
        TiffPtr sequence_tif_handle = nullptr; // Per-thread handle for sequence files

        for (amrex::MFIter mfi(dest_mf, true); mfi.isValid(); ++mfi)
        {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi);
            const amrex::Box& tile_box = mfi.tilebox();

            const int k_min = tile_box.smallEnd(2);
            const int k_max = tile_box.bigEnd(2);

            for (int k = k_min; k <= k_max; ++k) {
                TIFF* current_tif_raw_ptr = nullptr;
                std::string current_file_id_for_error = m_filename; // Default to stack filename

                // --- Get the correct TIFF handle for this slice ---
                if (m_is_sequence) {
                    std::string current_filename = generateFilename(m_base_pattern, m_start_index + k, m_digits, m_suffix);
                    current_file_id_for_error = current_filename; // Update for error messages
                    sequence_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!sequence_tif_handle) {
                        std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Failed to open sequence file: ";
                        error_msg += current_filename + " for slice " + std::to_string(k);
                        amrex::Abort(error_msg.c_str());
                    }
                    current_tif_raw_ptr = sequence_tif_handle.get();
                } else {
                    if (!shared_tif_stack_handle) {
                        std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Shared TIFF handle is null for slice " + std::to_string(k);
                        amrex::Abort(error_msg.c_str());
                    }
                    if (!TIFFSetDirectory(shared_tif_stack_handle.get(), static_cast<tdir_t>(k))) {
                        std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Failed to set directory ";
                        error_msg += std::to_string(k);
                        error_msg += " in file: ";
                        error_msg += m_filename;
                        error_msg += ". File may be corrupt.";
                        amrex::Abort(error_msg.c_str());
                    }
                    current_tif_raw_ptr = shared_tif_stack_handle.get();
                }

                if (!current_tif_raw_ptr) {
                     std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Null TIFF pointer encountered unexpectedly for slice " + std::to_string(k);
                     amrex::Abort(error_msg.c_str());
                }

                // --- Read data for the current slice (k) ---
                if (TIFFIsTiled(current_tif_raw_ptr)) {
                    // --- Tiled Reading Logic ---
                    // (Keep previous robust checks here)
                    // ... (rest of tiled logic as in previous version) ...
                    // *** Ensure robust checks inside tiled loop remain ***
                    uint32_t tile_width=0, tile_height=0;
                    TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width);
                    TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILELENGTH, &tile_height);
                    if (tile_width == 0 || tile_height == 0) {
                        std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Invalid tile dimensions (0) for slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                        amrex::Abort(error_msg.c_str());
                    }
                    const int chunk_width = static_cast<int>(tile_width);

                    tsize_t tile_buffer_size = TIFFTileSize(current_tif_raw_ptr);
                    if (tile_buffer_size <= 0) {
                         std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Invalid tile buffer size (" + std::to_string(tile_buffer_size) + ") for slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                         amrex::Abort(error_msg.c_str());
                    }
                    if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) { temp_buffer.resize(tile_buffer_size); }

                    int tx_min = tile_box.smallEnd(0) / tile_width; int tx_max = tile_box.bigEnd(0) / tile_width;
                    int ty_min = tile_box.smallEnd(1) / tile_height; int ty_max = tile_box.bigEnd(1) / tile_height;

                    for (int ty = ty_min; ty <= ty_max; ++ty) {
                        for (int tx = tx_min; tx <= tx_max; ++tx) {
                            int tile_origin_x = tx * tile_width; int tile_origin_y = ty * tile_height;
                            const int chunk_origin_x = tile_origin_x; const int chunk_origin_y = tile_origin_y;
                            ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, tile_origin_x, tile_origin_y, 0, 0);

                            tsize_t bytes_read = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);

                            if (bytes_read < 0) {
                                std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Error reading tile index ";
                                error_msg += std::to_string(tile_index);
                                error_msg += " (coords " + std::to_string(tx) + "," + std::to_string(ty) + ")";
                                error_msg += " for slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                                amrex::Abort(error_msg.c_str());
                            }

                            amrex::Box tile_abs_box(amrex::IntVect(tile_origin_x, tile_origin_y, k),
                                                    amrex::IntVect(tile_origin_x + tile_width - 1, tile_origin_y + tile_height - 1, k));
                            amrex::Box intersection = tile_box & tile_abs_box;

                            if (intersection.ok()) {
                                amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                    // ... (pixel interpretation logic remains the same) ...
                                    double value_as_double = 0.0;
                                    if (bits_per_sample_val == 1) { // Special handling for 1-bit data
                                        int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                        size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                        size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                        int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                        if (byte_index_in_buffer >= static_cast<size_t>(bytes_read)) { return; }
                                        unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                        int bit_value = 0;
                                        if (m_fill_order == FILLORDER_MSB2LSB) { bit_value = (packed_byte >> (7 - bit_index_in_byte)) & 1; }
                                        else { bit_value = (packed_byte >> bit_index_in_byte) & 1; }
                                        value_as_double = static_cast<double>(bit_value);
                                    } else { // Handling for BPS >= 8
                                        const size_t bytes_per_sample = bits_per_sample_val / 8;
                                        int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                        size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                        if (offset_in_buffer + bytes_per_sample > static_cast<size_t>(bytes_read)) { return; }
                                        const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                        value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                    }
                                    fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                });
                            }
                        }
                    }
                } else {
                    // --- Striped Reading Logic ---
                    uint32_t rows_per_strip = 0; uint32_t current_height32 = static_cast<uint32_t>(m_height);
                    uint16_t compression = COMPRESSION_NONE; // Default to check against
                    uint64_t* strip_byte_counts_ptr = nullptr; // Pointer for TIFFTAG_STRIPBYTECOUNTS
                    uint32_t strip_byte_counts_count = 0;      // Count for the array

                    TIFFGetFieldDefaulted(current_tif_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                    TIFFGetFieldDefaulted(current_tif_raw_ptr, TIFFTAG_COMPRESSION, &compression);

                    // *** Check if StripByteCounts tag exists ***
                    bool has_strip_byte_counts = TIFFGetField(current_tif_raw_ptr, TIFFTAG_STRIPBYTECOUNTS, &strip_byte_counts_count, &strip_byte_counts_ptr);

                    // Clamp rows_per_strip if it's invalid or larger than image height
                    if (rows_per_strip == 0 || rows_per_strip > current_height32) { rows_per_strip = current_height32; }
                    const int chunk_width = m_width;

                    // Calculate expected size IF uncompressed and StripByteCounts is missing
                    tsize_t calculated_strip_size = -1; // Use -1 to indicate not calculated/needed
                    bool use_raw_read = false;
                    if (!has_strip_byte_counts) {
                        if (compression == COMPRESSION_NONE) {
                            // Calculate size: width * rows_in_this_strip * bytes_per_pixel
                            // Note: This assumes bytes_per_pixel is accurate and BPS >= 8 for simplicity here
                            // A more robust calculation might be needed for BPS < 8
                            if (bits_per_sample_val >= 8) {
                                calculated_strip_size = static_cast<tsize_t>(m_width) * rows_per_strip * bytes_per_pixel;
                                use_raw_read = true; // We will use TIFFReadRawStrip
                                // Optional: Add a warning that we are proceeding without StripByteCounts
                                // amrex::Warning("[TiffReader] Missing StripByteCounts for slice " + std::to_string(k) + ". Assuming uncompressed and calculating size.");
                            } else {
                                // Abort if BPS < 8 and StripByteCounts is missing, as size calculation is complex
                                std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Missing StripByteCounts and BPS < 8 for slice " + std::to_string(k) + " in file: " + current_file_id_for_error + ". Cannot reliably calculate strip size.";
                                amrex::Abort(error_msg.c_str());
                            }
                        } else {
                            // Abort if compressed and StripByteCounts is missing
                            std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Missing StripByteCounts for compressed slice " + std::to_string(k) + " in file: " + current_file_id_for_error + ". Cannot read.";
                            amrex::Abort(error_msg.c_str());
                        }
                    }

                    // Determine buffer size needed
                    tsize_t buffer_size_needed = use_raw_read ? calculated_strip_size : TIFFStripSize(current_tif_raw_ptr);
                     if (buffer_size_needed <= 0) {
                         std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Invalid strip buffer size (" + std::to_string(buffer_size_needed) + ") determined for slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                         amrex::Abort(error_msg.c_str());
                     }
                    if (temp_buffer.size() < static_cast<size_t>(buffer_size_needed)) { temp_buffer.resize(buffer_size_needed); }

                    // Determine the range of strips needed for the current MFIter box
                    int strip_y_min = tile_box.smallEnd(1); int strip_y_max = tile_box.bigEnd(1);
                    tstrip_t first_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_min, 0);
                    tstrip_t last_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_max, 0);

                    // Loop through required strips for this slice
                    for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                        tsize_t bytes_read = -1; // Initialize to error state

                        if (use_raw_read) {
                            // *** Use TIFFReadRawStrip ***
                            // We need the offset for this strip
                            uint64_t* strip_offsets_ptr = nullptr;
                            uint32_t strip_offsets_count = 0;
                            if (!TIFFGetField(current_tif_raw_ptr, TIFFTAG_STRIPOFFSETS, &strip_offsets_count, &strip_offsets_ptr) || strip >= strip_offsets_count || !strip_offsets_ptr) {
                                std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Missing or invalid StripOffsets tag when trying raw read for strip " + std::to_string(strip) + " slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                                amrex::Abort(error_msg.c_str());
                            }
                            // Read raw data using the offset and calculated size
                            bytes_read = TIFFReadRawStrip(current_tif_raw_ptr, strip, temp_buffer.data(), calculated_strip_size);
                        } else {
                            // *** Use TIFFReadEncodedStrip (standard path) ***
                            bytes_read = TIFFReadEncodedStrip(current_tif_raw_ptr, strip, temp_buffer.data(), buffer_size_needed); // Use buffer_size_needed as hint
                        }


                        if (bytes_read < 0) { // Check specifically for error (-1)
                            // *** ABORT if reading strip fails ***
                            std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Error reading strip ";
                            error_msg += std::to_string(strip);
                            error_msg += " for slice " + std::to_string(k) + " in file: " + current_file_id_for_error;
                            error_msg += (use_raw_read ? " (using raw read)" : " (using encoded read)");
                            amrex::Abort(error_msg.c_str());
                        }
                        // Check if zero bytes were read when expecting data (might indicate issue even if no error code)
                        if (bytes_read == 0 && buffer_size_needed > 0) {
                             // Optional: Add warning or abort? Zero bytes read might be valid for empty strips sometimes.
                             // For now, let's add a warning.
                             amrex::Warning("[TiffReader] Read 0 bytes for strip " + std::to_string(strip) + " slice " + std::to_string(k) + " when buffer size was " + std::to_string(buffer_size_needed));
                        }


                        // Calculate the absolute Y range covered by this strip
                        uint32_t strip_origin_y = strip * rows_per_strip;
                        uint32_t strip_rows_this = std::min(rows_per_strip, current_height32 - strip_origin_y);
                        uint32_t strip_end_y = strip_origin_y + strip_rows_this - 1;
                        const int chunk_origin_x = 0; const int chunk_origin_y = static_cast<int>(strip_origin_y);

                        // Calculate intersection between the current strip and the MFIter box
                        amrex::Box strip_abs_box(amrex::IntVect(0, static_cast<int>(strip_origin_y), k),
                                                 amrex::IntVect(m_width - 1, static_cast<int>(strip_end_y), k));
                        amrex::Box intersection = tile_box & strip_abs_box;

                        // Process the intersection if it's valid
                        if (intersection.ok()) {
                            // Use AMReX's loop for CPU execution (adjust if using GPU)
                            amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                // ... (pixel interpretation logic remains the same) ...
                                double value_as_double = 0.0;
                                if (bits_per_sample_val == 1) { // Special handling for 1-bit data
                                    int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                    size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                    size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                    int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                    if (byte_index_in_buffer >= static_cast<size_t>(bytes_read)) { return; }
                                    unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                    int bit_value = 0;
                                    if (m_fill_order == FILLORDER_MSB2LSB) { bit_value = (packed_byte >> (7 - bit_index_in_byte)) & 1; }
                                    else { bit_value = (packed_byte >> bit_index_in_byte) & 1; }
                                    value_as_double = static_cast<double>(bit_value);
                                } else { // Handling for BPS >= 8
                                    const size_t bytes_per_sample = bits_per_sample_val / 8;
                                    int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                    size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                     if (offset_in_buffer + bytes_per_sample > static_cast<size_t>(bytes_read)) { return; }
                                    const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                    value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                }
                                fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                            });
                        } // end if intersection ok
                    } // end loop over strips
                } // end else (striped reading)

                // Close the per-thread sequence handle if it was used for this slice
                if (m_is_sequence) { sequence_tif_handle.reset(); }

            } // End loop k (Z-slices)
        } // End MFIter loop
    } // End OMP parallel region

    // Ensure all ranks have finished before proceeding
    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");
}

// Public threshold method with custom values - calls private helper
void TiffReader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    readDistributedIntoFab(mf, value_if_true, value_if_false, raw_threshold);
}

// Overload for 1/0 threshold output
void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    threshold(raw_threshold, 1, 0, mf);
}

} // namespace OpenImpala

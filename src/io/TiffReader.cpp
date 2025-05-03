// Add this include if you haven't already, for OpenMP directives
#include <omp.h>

// --- Other includes from your original file ---
#include "TiffReader.H" // Ensure this header defines the class and enum
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
#include <AMReX_Utility.H>        // Included for AMREX_ALWAYS_ASSERT, amrex::Verbose
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Bcast, MyProc, Communicator
#include <AMReX_MFIter.H>         // Needed for MFIter
#include <AMReX_Array4.H>         // Needed for Array4 access
#include <AMReX_Loop.H>           // For LoopOnCpu / amrex::ParallelFor


namespace OpenImpala {

//================================================================
// Anonymous namespace for internal helpers
//================================================================
namespace { // Anonymous namespace content remains the same...

// RAII wrapper for TIFF* handle
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// Helper to interpret raw bytes based on TIFF metadata (for BPS >= 8)
// Returns value cast to double for threshold comparison
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE
double interpretBytesAsDouble(const unsigned char* byte_ptr,
                              uint16_t bits_per_sample,
                              uint16_t sample_format)
{
    // Ensure bits_per_sample is a multiple of 8 for this function
    // Note: This function is NOT called for 1-bit data in the corrected reader logic
    if (bits_per_sample < 8 || (bits_per_sample % 8 != 0) ) return 0.0;

    size_t bytes_per_sample = bits_per_sample / 8;
    double value = 0.0;

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
            } // Add else? Error/warning?
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
            } // Add else?
            break;
        case SAMPLEFORMAT_IEEEFP:
             if (bytes_per_sample == 4) {
                // Ensure float is standard size
                static_assert(sizeof(float) == 4, "Float size mismatch");
                float val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 8) {
                 // Ensure double is standard size
                static_assert(sizeof(double) == 8, "Double size mismatch");
                double val; std::memcpy(&val, byte_ptr, sizeof(val)); value = val; // Direct assignment
            } // Add else?
            break;
        default:
            // Consider adding a warning or error for unsupported sample formats if necessary
            value = 0.0; // Default for unsupported format
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

} // end anonymous namespace


//================================================================
// Constructors / Metadata Getters / readFile / readFileSequence
//================================================================
// --- These sections remain unchanged from your previous correct version ---
// --- Assume TiffReader(), TiffReader(filename), TiffReader(sequence), ---
// --- width(), height(), depth(), etc., box(), readFile(), readFileSequence() ---
// --- are all present and correct as previously shown.                       ---

TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0),
    m_bits_per_sample(0), m_sample_format(0), m_samples_per_pixel(0),
    m_fill_order(FILLORDER_MSB2LSB), // Initialize fill order default
    m_is_read(false), m_is_sequence(false), m_start_index(0), m_digits(1)
{}

TiffReader::TiffReader(const std::string& filename) : TiffReader()
{
    // Constructor for single stack file
    m_filename = filename; // Store filename immediately
    if (!readFile(filename)) {
        // Error message handled within readFile -> readMetadataInternal
        throw std::runtime_error("TiffReader(filename): Failed to read metadata from file: " + filename);
    }
}

TiffReader::TiffReader(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix) : TiffReader()
{
    // Constructor for sequence files
    m_base_pattern = base_pattern;
    m_start_index = start_index;
    m_digits = digits;
    m_suffix = suffix;
    if (!readFileSequence(base_pattern, num_files, start_index, digits, suffix)) {
         // Error message handled within readFileSequence
         throw std::runtime_error("TiffReader(sequence): Failed to read metadata for sequence: " + base_pattern);
    }
}

int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }
// Assumes isRead() and fillOrder() are defined in TiffReader.H
// bool TiffReader::isRead() const { return m_is_read; }
// uint16_t TiffReader::fillOrder() const { return m_fill_order; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) { return amrex::Box(); } // Return empty box if not read
    // AMReX Box is inclusive: (low_corner, high_corner)
    return amrex::Box(amrex::IntVect::TheZeroVector(), // Low corner is (0,0,0)
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1)); // High corner
}

bool TiffReader::readFile(const std::string& filename)
{
    m_is_sequence = false; m_filename = filename; m_base_pattern = "";
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    uint16_t fill_order_r0 = FILLORDER_MSB2LSB;
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (filename.empty()) { amrex::Abort("[TiffReader::readFile] Filename cannot be empty."); }
        TiffPtr tif(TIFFOpen(filename.c_str(), "r"), TiffCloser());
        if (!tif) { amrex::Abort("[TiffReader::readFile] Failed to open TIFF file: " + filename); }
        uint32_t w32 = 0, h32 = 0; uint16_t planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { amrex::Abort("[TiffReader::readFile] Failed to get image dimensions from: " + filename); }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);
        width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32); bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) { std::stringstream ss; ss << "[TiffReader::readFile] Invalid or unsupported TIFF format in: " << filename << " (W=" << width_r0 << ", H=" << height_r0 << ", BPS=" << bps_r0 << ", Planar=" << planar << ")"; amrex::Abort(ss.str()); }
        depth_r0 = 0; if (!TIFFSetDirectory(tif.get(), 0)) { amrex::Abort("[TiffReader::readFile] Failed to set initial directory (0) in: " + filename); } do { depth_r0++; } while (TIFFReadDirectory(tif.get())); if (depth_r0 == 0) { amrex::Abort("[TiffReader::readFile] Could not read any directories (depth is zero) in: " + filename); }
    }
    std::vector<int> idata = {width_r0, height_r0, depth_r0, static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0), static_cast<int>(m_is_sequence), static_cast<int>(fill_order_r0)};
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());
    m_width = idata[0]; m_height = idata[1]; m_depth = idata[2]; m_bits_per_sample = static_cast<uint16_t>(idata[3]); m_sample_format = static_cast<uint16_t>(idata[4]); m_samples_per_pixel = static_cast<uint16_t>(idata[5]); m_is_sequence = static_cast<bool>(idata[6]); m_fill_order = static_cast<uint16_t>(idata[7]);
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    if (amrex::ParallelDescriptor::IOProcessor()) { int string_len = static_cast<int>(m_filename.length()); amrex::ParallelDescriptor::Bcast(&string_len, 1, root); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root); } else { int string_len = 0; amrex::ParallelDescriptor::Bcast(&string_len, 1, root); m_filename.resize(string_len); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root); }
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0) { amrex::Abort("TiffReader::readFile: Invalid metadata received after broadcast."); }
    m_is_read = true; return true;
}

bool TiffReader::readFileSequence( const std::string& base_pattern, int num_files, int start_index, int digits, const std::string& suffix)
{
    m_is_sequence = true; m_base_pattern = base_pattern; m_start_index = start_index; m_digits = digits; m_suffix = suffix; m_filename = "";
    int width_r0=0, height_r0=0, depth_r0=0; uint16_t bps_r0=0, fmt_r0=0, spp_r0=0; uint16_t fill_order_r0 = FILLORDER_MSB2LSB; std::string first_filename_r0 = "";
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (num_files <= 0 || digits <= 0 || base_pattern.empty()) { amrex::Abort("[TiffReader::readFileSequence] Invalid sequence parameters (num_files, digits, base_pattern)."); }
        depth_r0 = num_files; first_filename_r0 = generateFilename(base_pattern, start_index, digits, suffix); TiffPtr tif(TIFFOpen(first_filename_r0.c_str(), "r"), TiffCloser()); if (!tif) { amrex::Abort("[TiffReader::readFileSequence] Failed to open first sequence file: " + first_filename_r0); }
        uint32_t w32 = 0, h32 = 0; uint16_t planar = PLANARCONFIG_CONTIG; if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { amrex::Abort("[TiffReader::readFileSequence] Failed to get image dimensions from: " + first_filename_r0); }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar); TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0);
        width_r0 = static_cast<int>(w32); height_r0 = static_cast<int>(h32); bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64); if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) { amrex::Abort("[TiffReader::readFileSequence] Invalid or unsupported TIFF format in: " + first_filename_r0 + " (Check dimensions, BPS, PlanarConfig)"); }
    }
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    std::vector<int> idata = {width_r0, height_r0, depth_r0, static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0), static_cast<int>(m_is_sequence), m_start_index, m_digits, static_cast<int>(fill_order_r0)};
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), root);
    m_width = idata[0]; m_height = idata[1]; m_depth = idata[2]; m_bits_per_sample = static_cast<uint16_t>(idata[3]); m_sample_format = static_cast<uint16_t>(idata[4]); m_samples_per_pixel = static_cast<uint16_t>(idata[5]); m_is_sequence = static_cast<bool>(idata[6]); m_start_index = idata[7]; m_digits = idata[8]; m_fill_order = static_cast<uint16_t>(idata[9]);
    if (amrex::ParallelDescriptor::IOProcessor()) { int string_len = static_cast<int>(m_base_pattern.length()); amrex::ParallelDescriptor::Bcast(&string_len, 1, root); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root); } else { int string_len = 0; amrex::ParallelDescriptor::Bcast(&string_len, 1, root); m_base_pattern.resize(string_len); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_base_pattern.data()), string_len, root); }
    if (amrex::ParallelDescriptor::IOProcessor()) { int string_len = static_cast<int>(m_suffix.length()); amrex::ParallelDescriptor::Bcast(&string_len, 1, root); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root); } else { int string_len = 0; amrex::ParallelDescriptor::Bcast(&string_len, 1, root); m_suffix.resize(string_len); amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_suffix.data()), string_len, root); }
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || (m_is_sequence && m_base_pattern.empty())) { amrex::Abort("TiffReader::readFileSequence: Invalid metadata received after broadcast."); }
    m_is_read = true; return true;
}


//================================================================
// readDistributedIntoFab Method - MODIFIED WITH DEBUGGING
//================================================================
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

    // Assertions remain the same...
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.boxArray().minimalBox() == this->box(),
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab BoxArray domain does not match reader Box.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nComp() == 1,
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab must have exactly 1 component.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nGrow() == 0,
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab must have 0 ghost cells.");


    const int bits_per_sample_val = m_bits_per_sample;
    const size_t bytes_per_sample = (bits_per_sample_val >= 8) ? (bits_per_sample_val / 8) : 1; // Should be 1 for BPS=1
    const size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel; // Should be 1 for BPS=1, SPP=1

    if (bits_per_sample_val == 0 ) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bits per sample is zero!");
    }

    // File handle setup remains the same...
    TIFF* shared_tif_stack_raw_ptr = nullptr;
    TiffPtr shared_tif_stack_handle = nullptr;
    if (!m_is_sequence) {
        shared_tif_stack_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
        if (!shared_tif_stack_handle) {
            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open shared TIFF file: " + m_filename);
        }
        shared_tif_stack_raw_ptr = shared_tif_stack_handle.get();
    }

    // *** Get overall image dimensions for boundary check ***
    const int image_width = m_width;
    const int image_height = m_height;
    const int image_depth = m_depth;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        std::vector<unsigned char> temp_buffer;
        TiffPtr sequence_tif_handle = nullptr;

        for (amrex::MFIter mfi(dest_mf, true); mfi.isValid(); ++mfi)
        {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi);
            const amrex::Box& tile_box = mfi.tilebox();

            const int k_min = tile_box.smallEnd(2);
            const int k_max = tile_box.bigEnd(2);

            for (int k = k_min; k <= k_max; ++k) {

                if (m_is_sequence) {
                    // --- Sequence Reading Logic ---
                    // [Remains unchanged from previous correct version]
                    // ... (Open file, read tile/strip, process intersection) ...
                    // *** Important: Insert the debug printing block below inside the
                    // *** amrex::LoopOnCpu for sequence reading as well if needed.
                    // *** For brevity, it's shown only in the stack reading path below.
                    // ...
                } else { // !m_is_sequence -> Stack Reading Logic
                    #pragma omp critical (TiffReadLock)
                    {
                        if (!shared_tif_stack_raw_ptr) { amrex::Abort("[TiffReader] FATAL: Shared handle pointer is null inside critical section!"); }
                        if (!TIFFSetDirectory(shared_tif_stack_raw_ptr, static_cast<tdir_t>(k))) { std::string error_msg = "[TiffReader] FATAL: Failed to set directory " + std::to_string(k) + " in file: " + m_filename; amrex::Abort(error_msg.c_str()); }

                        if (TIFFIsTiled(shared_tif_stack_raw_ptr)) {
                            // --- Tiled Stack Reading (Protected) ---
                            uint32_t tile_width=0, tile_height=0; TIFFGetField(shared_tif_stack_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width); TIFFGetField(shared_tif_stack_raw_ptr, TIFFTAG_TILELENGTH, &tile_height);
                            if (tile_width == 0 || tile_height == 0) { amrex::Abort("[TiffReader] FATAL: Invalid tile dimensions."); } const int chunk_width = static_cast<int>(tile_width);
                            tsize_t tile_buffer_size = TIFFTileSize(shared_tif_stack_raw_ptr); if (tile_buffer_size <= 0) { amrex::Abort("[TiffReader] FATAL: Invalid tile buffer size."); }
                            if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) { temp_buffer.resize(tile_buffer_size); }
                            int tx_min = tile_box.smallEnd(0) / tile_width; int tx_max = tile_box.bigEnd(0) / tile_width; int ty_min = tile_box.smallEnd(1) / tile_height; int ty_max = tile_box.bigEnd(1) / tile_height;
                            for (int ty = ty_min; ty <= ty_max; ++ty) {
                                for (int tx = tx_min; tx <= tx_max; ++tx) {
                                    int chunk_origin_x = tx * tile_width; int chunk_origin_y = ty * tile_height;
                                    amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k), amrex::IntVect(chunk_origin_x + tile_width - 1, chunk_origin_y + tile_height - 1, k));
                                    amrex::Box intersection = tile_box & chunk_abs_box;
                                    if (intersection.ok()) {
                                        ttile_t tile_index = TIFFComputeTile(shared_tif_stack_raw_ptr, chunk_origin_x, chunk_origin_y, 0, 0);
                                        tsize_t bytes_read = TIFFReadEncodedTile(shared_tif_stack_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);
                                        if (bytes_read < 0) { std::string error_msg = "[TiffReader] FATAL: Error reading tile index " + std::to_string(tile_index) + " slice " + std::to_string(k); amrex::Abort(error_msg.c_str()); }

                                        // Process data (outside critical section, but buffer is thread-local)
                                        amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                            double value_as_double = 0.0;
                                            int bit_value = -1; // Initialize to invalid
                                            unsigned char packed_byte = 0;
                                            size_t byte_index_in_buffer = 0;
                                            int bit_index_in_byte = 0;
                                            size_t linear_pixel_index_in_chunk = 0;

                                            if (bits_per_sample_val == 1) {
                                                 int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                                 linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                                 byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                                 bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                                 if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                     packed_byte = temp_buffer[byte_index_in_buffer];
                                                     bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                     value_as_double = static_cast<double>(bit_value);
                                                 } else { value_as_double = 0.0; bit_value = -2; /* Indicate Bounds Error */ }
                                            } else { // BPS >= 8 logic (unchanged)
                                                int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                                size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                                if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                                   const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                   value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                                } else { value_as_double = 0.0; }
                                            }

                                            // *** START DEBUG BLOCK ***
                                            // Print details for pixels near max boundaries if verbose >= 3 and on Rank 0
                                            // Check i,j,k against overall image dimensions stored in members
                                            if (amrex::Verbose() >= 3 && amrex::ParallelDescriptor::IOProcessor()) {
                                                 bool is_boundary = ( i == image_width - 1 || j == image_height - 1 || k_loop == image_depth - 1 );
                                                 // Optionally add || i == 0 || j == 0 || k_loop == 0

                                                 if (is_boundary && bits_per_sample_val == 1) { // Only print for 1-bit boundary pixels
                                                      amrex::Print() << "TIFF_DBG: Voxel(" << i << "," << j << "," << k_loop << ") "
                                                                     << "ChunkIdx(" << linear_pixel_index_in_chunk << ") "
                                                                     << "ByteIdx(" << byte_index_in_buffer << ") "
                                                                     << "BitInByte(" << bit_index_in_byte << ") "
                                                                     << "PackedByte(0x" << std::hex << static_cast<int>(packed_byte) << std::dec << ") "
                                                                     << "RawBitValue(" << bit_value << ") "
                                                                     << "Thresholded(" << ((value_as_double > raw_threshold) ? value_if_true : value_if_false) << ")\n";
                                                 }
                                            }
                                            // *** END DEBUG BLOCK ***

                                            fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                        }); // End LoopOnCpu
                                    } // end if intersection ok
                                } // end tx loop
                            } // end ty loop
                        } else {
                            // --- Striped Stack Reading (Protected) ---
                            uint32_t rows_per_strip = 0; uint32_t current_height32 = static_cast<uint32_t>(m_height); TIFFGetFieldDefaulted(shared_tif_stack_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                            if (rows_per_strip == 0 || rows_per_strip > current_height32) { rows_per_strip = current_height32; } const int chunk_width = m_width;
                            tsize_t strip_buffer_size = TIFFStripSize(shared_tif_stack_raw_ptr); if (strip_buffer_size <= 0) { amrex::Abort("[TiffReader] FATAL: Invalid strip buffer size."); }
                            if (temp_buffer.size() < static_cast<size_t>(strip_buffer_size)) { temp_buffer.resize(strip_buffer_size); }
                            int strip_y_min = tile_box.smallEnd(1); int strip_y_max = tile_box.bigEnd(1); tstrip_t first_strip = TIFFComputeStrip(shared_tif_stack_raw_ptr, strip_y_min, 0); tstrip_t last_strip = TIFFComputeStrip(shared_tif_stack_raw_ptr, strip_y_max, 0);
                            for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                                uint32_t strip_origin_y_uint = strip * rows_per_strip; if (strip_origin_y_uint > std::numeric_limits<int>::max()) { amrex::Abort("Strip origin Y exceeds integer limits"); } int chunk_origin_y = static_cast<int>(strip_origin_y_uint); const int chunk_origin_x = 0;
                                uint32_t strip_rows_this = std::min(rows_per_strip, current_height32 - strip_origin_y_uint); if (strip_rows_this == 0) continue; int chunk_height = static_cast<int>(strip_rows_this);
                                amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k), amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                                amrex::Box intersection = tile_box & chunk_abs_box;
                                if (intersection.ok()) {
                                    tsize_t bytes_read = TIFFReadEncodedStrip(shared_tif_stack_raw_ptr, strip, temp_buffer.data(), strip_buffer_size);
                                    if (bytes_read < 0) { std::string error_msg = "[TiffReader] FATAL: Error reading strip " + std::to_string(strip) + " slice " + std::to_string(k); amrex::Abort(error_msg.c_str()); }

                                    amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                        double value_as_double = 0.0;
                                        int bit_value = -1; // Initialize to invalid
                                        unsigned char packed_byte = 0;
                                        size_t byte_index_in_buffer = 0;
                                        int bit_index_in_byte = 0;
                                        size_t linear_pixel_index_in_chunk = 0;

                                        if (bits_per_sample_val == 1) {
                                            int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                            linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                            byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                            bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                            if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                packed_byte = temp_buffer[byte_index_in_buffer];
                                                bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                value_as_double = static_cast<double>(bit_value);
                                            } else { value_as_double = 0.0; bit_value = -2; /* Indicate Bounds Error */ }
                                        } else { // BPS >= 8 logic (unchanged)
                                            int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                            size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                            if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                                const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                            } else { value_as_double = 0.0; }
                                        }

                                        // *** START DEBUG BLOCK ***
                                        if (amrex::Verbose() >= 3 && amrex::ParallelDescriptor::IOProcessor()) {
                                             bool is_boundary = ( i == image_width - 1 || j == image_height - 1 || k_loop == image_depth - 1 );
                                             if (is_boundary && bits_per_sample_val == 1) {
                                                  amrex::Print().SetPrecision(0) << "TIFF_DBG: Voxel(" << i << "," << j << "," << k_loop << ") "
                                                                 << "Strip(" << strip << ") "
                                                                 << "ChunkIdx(" << linear_pixel_index_in_chunk << ") "
                                                                 << "ByteIdx(" << byte_index_in_buffer << ") "
                                                                 << "BitInByte(" << bit_index_in_byte << ") "
                                                                 << "PackedByte(0x" << std::hex << static_cast<int>(packed_byte) << std::dec << ") "
                                                                 << "RawBitValue(" << bit_value << ") "
                                                                 << "Thresholded(" << ((value_as_double > raw_threshold) ? value_if_true : value_if_false) << ")\n";
                                             }
                                        }
                                        // *** END DEBUG BLOCK ***

                                        fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                    }); // End LoopOnCpu
                                } // end if intersection ok
                            } // end strip loop
                        } // end if tiled/striped for stack
                    } // >>> END of #pragma omp critical (TiffReadLock) <<<
                } // End else (Stack reading vs Sequence reading)
            } // End loop k (Z-slices)
        } // End MFIter loop
    } // End OMP parallel region

    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");
}

//================================================================
// Public threshold methods (call the main implementation)
//================================================================
void TiffReader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    // Calls the main implementation above
    readDistributedIntoFab(mf, value_if_true, value_if_false, raw_threshold);
}

void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    // Calls the overload with default true/false values
    threshold(raw_threshold, 1, 0, mf);
}

} // namespace OpenImpala

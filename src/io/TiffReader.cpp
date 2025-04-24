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
#include <AMReX_Utility.H>       // Included for AMREX_ALWAYS_ASSERT
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Bcast, MyProc, Communicator
#include <AMReX_MFIter.H>       // Needed for MFIter
#include <AMReX_Array4.H>       // Needed for Array4 access
#include <AMReX_Loop.H>         // For LoopOnCpu / amrex::ParallelFor


namespace OpenImpala {

//================================================================
// Anonymous namespace for internal helpers
//================================================================
namespace {

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
// Constructors
//================================================================
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

//================================================================
// Metadata Getters
//================================================================
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }
// ** NOTE: isRead() definition is removed from here, assumed to be in TiffReader.H **
// bool TiffReader::isRead() const { return m_is_read; } // <-- REMOVED

amrex::Box TiffReader::box() const {
    if (!m_is_read) { return amrex::Box(); } // Return empty box if not read
    // AMReX Box is inclusive: (low_corner, high_corner)
    return amrex::Box(amrex::IntVect::TheZeroVector(), // Low corner is (0,0,0)
                      amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1)); // High corner
}

//================================================================
// readFile (Metadata Only) - For Single Stack File
//================================================================
bool TiffReader::readFile(const std::string& filename)
{
    // Ensure single file mode is set, store filename
    m_is_sequence = false;
    m_filename = filename;
    m_base_pattern = ""; // Clear sequence pattern

    // --- Metadata Reading (Rank 0 only) ---
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    uint16_t fill_order_r0 = FILLORDER_MSB2LSB; // Initialize to TIFF default

    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (filename.empty()) {
             amrex::Abort("[TiffReader::readFile] Filename cannot be empty.");
        }
        // Open file using RAII handle
        TiffPtr tif(TIFFOpen(filename.c_str(), "r"), TiffCloser());
        if (!tif) {
             amrex::Abort("[TiffReader::readFile] Failed to open TIFF file: " + filename);
        }

        // Read mandatory dimension tags
        uint32_t w32 = 0, h32 = 0;
        uint16_t planar = PLANARCONFIG_CONTIG; // Check for contiguous planar config
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
            !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32))
        {
             amrex::Abort("[TiffReader::readFile] Failed to get image dimensions from: " + filename);
        }

        // Read optional tags with defaults
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0); // Read fill order

        // Validate read values
        width_r0 = static_cast<int>(w32);
        height_r0 = static_cast<int>(h32);
        bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) {
             std::stringstream ss;
             ss << "[TiffReader::readFile] Invalid or unsupported TIFF format in: " << filename
                << " (W=" << width_r0 << ", H=" << height_r0 << ", BPS=" << bps_r0
                << ", Planar=" << planar << ")";
             amrex::Abort(ss.str());
        }

        // Count directories (Z-depth)
        depth_r0 = 0;
        if (!TIFFSetDirectory(tif.get(), 0)) { // Check if first directory is valid
             amrex::Abort("[TiffReader::readFile] Failed to set initial directory (0) in: " + filename);
        }
        do {
             depth_r0++;
        } while (TIFFReadDirectory(tif.get())); // Loop until no more directories

        if (depth_r0 == 0) { // Should be at least 1 if initial set worked
             amrex::Abort("[TiffReader::readFile] Could not read any directories (depth is zero) in: " + filename);
        }
    } // End IOProcessor block

    // --- Broadcast Metadata to All Ranks ---
    // Pack data into a vector for Bcast (add fill_order_r0)
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence), static_cast<int>(fill_order_r0)}; // Added fill_order
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // Unpack data on all ranks
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2];
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]); // Should be false here
    m_fill_order        = static_cast<uint16_t>(idata[7]); // Unpack fill_order

    // ** CORRECTED: Manually Broadcast filename string **
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_filename.length());
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root); // Send length
        // Use const_cast carefully if necessary, Bcast takes void*
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root); // Send data
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root); // Receive length
        m_filename.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root); // Receive data
    }

    // Final validation on all ranks after broadcast
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0) {
        amrex::Abort("TiffReader::readFile: Invalid metadata received after broadcast.");
    }

    m_is_read = true; // Mark metadata as successfully read
    return true;
}

//================================================================
// readFileSequence (Metadata Only) - For Sequence of Files
//================================================================
bool TiffReader::readFileSequence(
    const std::string& base_pattern, int num_files, int start_index, int digits, const std::string& suffix)
{
    // Set sequence mode and parameters
    m_is_sequence = true;
    m_base_pattern = base_pattern;
    m_start_index = start_index;
    m_digits = digits;
    m_suffix = suffix;
    m_filename = ""; // Clear single filename

    // --- Read Metadata from First File (Rank 0 only) ---
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    uint16_t fill_order_r0 = FILLORDER_MSB2LSB; // Default fill order
    std::string first_filename_r0 = "";

    if (amrex::ParallelDescriptor::IOProcessor()) {
         if (num_files <= 0 || digits <= 0 || base_pattern.empty()) {
              amrex::Abort("[TiffReader::readFileSequence] Invalid sequence parameters (num_files, digits, base_pattern).");
         }
         depth_r0 = num_files; // Depth is number of files
         first_filename_r0 = generateFilename(base_pattern, start_index, digits, suffix);

         TiffPtr tif(TIFFOpen(first_filename_r0.c_str(), "r"), TiffCloser());
         if (!tif) {
              amrex::Abort("[TiffReader::readFileSequence] Failed to open first sequence file: " + first_filename_r0);
         }

         // Read metadata from the first file (same logic as readFile)
         uint32_t w32 = 0, h32 = 0;
         uint16_t planar = PLANARCONFIG_CONTIG;
         if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) || !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
              amrex::Abort("[TiffReader::readFileSequence] Failed to get image dimensions from: " + first_filename_r0);
         }
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_FILLORDER, &fill_order_r0); // Read fill order

         width_r0 = static_cast<int>(w32);
         height_r0 = static_cast<int>(h32);
         bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
         if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps || planar != PLANARCONFIG_CONTIG) {
              amrex::Abort("[TiffReader::readFileSequence] Invalid or unsupported TIFF format in: " + first_filename_r0 + " (Check dimensions, BPS, PlanarConfig)");
         }
         // No need to count directories here
    } // End IOProcessor block

    // --- Broadcast Metadata ---
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    // Add fill_order_r0 to broadcast data
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence), m_start_index, m_digits,
                              static_cast<int>(fill_order_r0)}; // Added fill_order
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), root);

    // Unpack data on all ranks
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2]; // Set from num_files
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]); // Should be true here
    m_start_index       = idata[7];
    m_digits            = idata[8];
    m_fill_order        = static_cast<uint16_t>(idata[9]); // Unpack fill_order

    // ** CORRECTED: Manually Broadcast string parameters **
    // Bcast m_base_pattern
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
    // Bcast m_suffix
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


    // Final validation on all ranks
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || (m_is_sequence && m_base_pattern.empty())) {
        amrex::Abort("TiffReader::readFileSequence: Invalid metadata received after broadcast.");
    }

    m_is_read = true; // Mark metadata read
    return true;
}


//================================================================
// readDistributedIntoFab Method - Corrected with OMP Critical Section
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

    // Ensure destination MultiFab matches geometry
    // ** CORRECTED ASSERTION **
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.boxArray().minimalBox() == this->box(),
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab BoxArray domain does not match reader Box.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nComp() == 1,
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab must have exactly 1 component.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(dest_mf.nGrow() == 0,
                                     "TiffReader::readDistributedIntoFab: Destination MultiFab must have 0 ghost cells.");


    const int bits_per_sample_val = m_bits_per_sample;
    // Calculate bytes per pixel based on BPS and SamplesPerPixel (for >= 8 BPS)
    // For 1-bit, pixel interpretation happens differently, but having a valid bytes_per_sample helps offset calculations
    const size_t bytes_per_sample = (bits_per_sample_val >= 8) ? (bits_per_sample_val / 8) : 1;
    const size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel; // Assumes PLANARCONFIG_CONTIG

    if (bits_per_sample_val == 0 ) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bits per sample is zero!");
    }

    // Open the single stack file handle once OUTSIDE the parallel region.
    // Store the raw pointer for use inside the OMP region (unique_ptr not easily captured)
    TIFF* shared_tif_stack_raw_ptr = nullptr;
    TiffPtr shared_tif_stack_handle = nullptr; // Keep unique_ptr for lifetime management
    if (!m_is_sequence) {
        // Use TIFFOpen instead of C++ fstream for libtiff handle
        shared_tif_stack_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
        if (!shared_tif_stack_handle) {
            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open shared TIFF file: " + m_filename);
        }
        shared_tif_stack_raw_ptr = shared_tif_stack_handle.get(); // Get raw pointer
    }

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion()) // Parallel region remains
#endif
    {
        // Each thread gets its own buffer.
        std::vector<unsigned char> temp_buffer;
        // For sequences, each thread gets its own file handle per slice.
        TiffPtr sequence_tif_handle = nullptr; // Per-thread handle for sequence files

        for (amrex::MFIter mfi(dest_mf, true); mfi.isValid(); ++mfi)
        {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi);
            const amrex::Box& tile_box = mfi.tilebox(); // Use tiling box

            const int k_min = tile_box.smallEnd(2);
            const int k_max = tile_box.bigEnd(2);

            // Loop over Z-slices needed for this MFIter tile
            for (int k = k_min; k <= k_max; ++k) {

                if (m_is_sequence) {
                    // --- Sequence Reading Logic (Thread-Safe per slice) ---
                    // [This block remains the same as the previous correct version]
                    std::string current_filename = generateFilename(m_base_pattern, m_start_index + k, m_digits, m_suffix);
                    std::string current_file_id_for_error = current_filename;
                    // Open handle locally for this slice
                    sequence_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!sequence_tif_handle) {
                         amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Failed to open sequence file: " + current_filename + " for slice " + std::to_string(k));
                    }
                    TIFF* current_tif_raw_ptr = sequence_tif_handle.get();

                    // Read and process data using the thread-local handle
                    if (TIFFIsTiled(current_tif_raw_ptr)) {
                         uint32_t tile_width=0, tile_height=0;
                         TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width);
                         TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILELENGTH, &tile_height);
                         if (tile_width == 0 || tile_height == 0) { amrex::Abort("..."); }
                         const int chunk_width = static_cast<int>(tile_width);

                         tsize_t tile_buffer_size = TIFFTileSize(current_tif_raw_ptr);
                         if (tile_buffer_size <= 0) { amrex::Abort("..."); }
                         if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) { temp_buffer.resize(tile_buffer_size); }

                         int tx_min = tile_box.smallEnd(0) / tile_width; int tx_max = tile_box.bigEnd(0) / tile_width;
                         int ty_min = tile_box.smallEnd(1) / tile_height; int ty_max = tile_box.bigEnd(1) / tile_height;

                         for (int ty = ty_min; ty <= ty_max; ++ty) {
                             for (int tx = tx_min; tx <= tx_max; ++tx) {
                                 int chunk_origin_x = tx * tile_width; int chunk_origin_y = ty * tile_height;
                                 amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                          amrex::IntVect(chunk_origin_x + tile_width - 1, chunk_origin_y + tile_height - 1, k));
                                 amrex::Box intersection = tile_box & chunk_abs_box;

                                 if (intersection.ok()) {
                                     ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, chunk_origin_x, chunk_origin_y, 0, 0);
                                     tsize_t bytes_read = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);
                                     if (bytes_read < 0) { amrex::Abort("..."); }

                                     // Process sequence tile data
                                     amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                         double value_as_double = 0.0;
                                         if (bits_per_sample_val == 1) { /* 1-bit logic */
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                              size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                              size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                              int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                              if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                   unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                                   int bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                   value_as_double = static_cast<double>(bit_value);
                                              }
                                         } else { /* BPS >= 8 logic */
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                              size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                              if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                                 const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                 value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                              }
                                         }
                                         fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                     });
                                 } // end if intersection ok
                             } // end tx loop
                         } // end ty loop
                    } else {
                         // --- Striped Reading Logic for Sequences ---
                         uint32_t rows_per_strip = 0; uint32_t current_height32 = static_cast<uint32_t>(m_height);
                         TIFFGetFieldDefaulted(current_tif_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                         if (rows_per_strip == 0 || rows_per_strip > current_height32) { rows_per_strip = current_height32; }
                         const int chunk_width = m_width;

                         tsize_t strip_buffer_size = TIFFStripSize(current_tif_raw_ptr);
                         if (strip_buffer_size <= 0) { amrex::Abort("..."); }
                         if (temp_buffer.size() < static_cast<size_t>(strip_buffer_size)) { temp_buffer.resize(strip_buffer_size); }

                         int strip_y_min = tile_box.smallEnd(1); int strip_y_max = tile_box.bigEnd(1);
                         tstrip_t first_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_min, 0);
                         tstrip_t last_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_max, 0);

                         for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                              uint32_t strip_origin_y_uint = strip * rows_per_strip;
                              if (strip_origin_y_uint > std::numeric_limits<int>::max()) { amrex::Abort("..."); }
                              int chunk_origin_y = static_cast<int>(strip_origin_y_uint);
                              const int chunk_origin_x = 0;

                              uint32_t strip_rows_this = std::min(rows_per_strip, current_height32 - strip_origin_y_uint);
                              if (strip_rows_this == 0) continue;
                              int chunk_height = static_cast<int>(strip_rows_this);

                              amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                  amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                              amrex::Box intersection = tile_box & chunk_abs_box;

                              if (intersection.ok()) {
                                  tsize_t bytes_read = TIFFReadEncodedStrip(current_tif_raw_ptr, strip, temp_buffer.data(), strip_buffer_size);
                                  if (bytes_read < 0) { amrex::Abort("..."); }

                                  amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                       double value_as_double = 0.0;
                                       if (bits_per_sample_val == 1) { /* 1-bit logic */
                                            int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                            size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                            size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                            int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                            if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                 unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                                 int bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                 value_as_double = static_cast<double>(bit_value);
                                            }
                                       } else { /* BPS >= 8 logic */
                                            int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                            size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                            if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                               const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                               value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                            }
                                       }
                                       fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                  }); // End LoopOnCpu
                              } // end if intersection ok
                         } // end strip loop
                    } // end if tiled/striped for sequence
                    sequence_tif_handle.reset(); // Close handle for this sequence file

                } else { // !m_is_sequence -> Stack Reading Logic
                    // --- Stack Reading Logic (Protected by Critical Section) ---

                    #pragma omp critical (TiffReadLock) // Protect all libtiff interactions for the stack file handle
                    {
                        // Use the raw pointer to the shared handle captured outside OMP region
                        if (!shared_tif_stack_raw_ptr) {
                            // This should not happen if file opening succeeded outside
                            amrex::Abort("[TiffReader::readDistributedIntoFab] FATAL: Shared TIFF handle pointer is null inside critical section!");
                        }

                        // Set Directory for slice k
                        if (!TIFFSetDirectory(shared_tif_stack_raw_ptr, static_cast<tdir_t>(k))) {
                             std::string error_msg = "[TiffReader::readDistributedIntoFab] FATAL: Failed to set directory " + std::to_string(k) + " in file: " + m_filename;
                             amrex::Abort(error_msg.c_str());
                        }

                        // Read and process data for this MFIter tile intersecting slice k
                        // We do the read AND the processing inside the critical section
                        if (TIFFIsTiled(shared_tif_stack_raw_ptr)) {
                             uint32_t tile_width=0, tile_height=0;
                             TIFFGetField(shared_tif_stack_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width);
                             TIFFGetField(shared_tif_stack_raw_ptr, TIFFTAG_TILELENGTH, &tile_height);
                             if (tile_width == 0 || tile_height == 0) { amrex::Abort("[TiffReader] FATAL: Invalid tile dimensions."); }
                             const int chunk_width = static_cast<int>(tile_width);

                             tsize_t tile_buffer_size = TIFFTileSize(shared_tif_stack_raw_ptr);
                             if (tile_buffer_size <= 0) { amrex::Abort("[TiffReader] FATAL: Invalid tile buffer size."); }
                             // Ensure thread-local buffer is large enough
                             if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) { temp_buffer.resize(tile_buffer_size); }

                             // Calculate range of tiles needed for this MFIter box
                             int tx_min = tile_box.smallEnd(0) / tile_width; int tx_max = tile_box.bigEnd(0) / tile_width;
                             int ty_min = tile_box.smallEnd(1) / tile_height; int ty_max = tile_box.bigEnd(1) / tile_height;

                             for (int ty = ty_min; ty <= ty_max; ++ty) {
                                 for (int tx = tx_min; tx <= tx_max; ++tx) {
                                     int chunk_origin_x = tx * tile_width; int chunk_origin_y = ty * tile_height;
                                     amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                              amrex::IntVect(chunk_origin_x + tile_width - 1, chunk_origin_y + tile_height - 1, k));
                                     amrex::Box intersection = tile_box & chunk_abs_box;

                                     if (intersection.ok()) { // Process only if there is overlap
                                         ttile_t tile_index = TIFFComputeTile(shared_tif_stack_raw_ptr, chunk_origin_x, chunk_origin_y, 0, 0);
                                         // Read the tile data into the thread-local buffer
                                         tsize_t bytes_read = TIFFReadEncodedTile(shared_tif_stack_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);
                                         if (bytes_read < 0) {
                                             std::string error_msg = "[TiffReader] FATAL: Error reading tile index " + std::to_string(tile_index) + " slice " + std::to_string(k);
                                             amrex::Abort(error_msg.c_str());
                                         }

                                         // Process data IMMEDIATELY after reading (still inside critical section)
                                         amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                             double value_as_double = 0.0;
                                             if (bits_per_sample_val == 1) {
                                                  int i_in_chunk = i - chunk_origin_x;
                                                  int j_in_chunk = j - chunk_origin_y;
                                                  // Use chunk_width (tile_width) for linear index calculation
                                                  size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                                  size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                                  int bit_index_in_byte = linear_pixel_index_in_chunk % 8;

                                                  // Check bounds using bytes_read from TIFFReadEncodedTile
                                                  if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                       unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                                       int bit_value = 0;
                                                       // Apply FillOrder
                                                       if (m_fill_order == FILLORDER_MSB2LSB) {
                                                           bit_value = (packed_byte >> (7 - bit_index_in_byte)) & 1;
                                                       } else { // Assume FILLORDER_LSB2MSB
                                                           bit_value = (packed_byte >> bit_index_in_byte) & 1;
                                                       }
                                                       value_as_double = static_cast<double>(bit_value);
                                                  } else {
                                                       // Index out of bounds - handle error or assign default? Assign default for now.
                                                       value_as_double = 0.0; // Or some other indicator?
                                                  }
                                             }
                                             // Handle BPS >= 8 data
                                             else {
                                                  // Use correct bytes_per_pixel calculated earlier
                                                  int i_in_chunk = i - chunk_origin_x;
                                                  int j_in_chunk = j - chunk_origin_y;
                                                  size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;

                                                  // Check bounds using bytes_read and bytes_per_sample
                                                  if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                                      const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                      value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                                  } else {
                                                       // Index out of bounds
                                                       value_as_double = 0.0; // Or handle error
                                                  }
                                             }
                                             // Apply threshold and assign to destination fab
                                             fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                         }); // End LoopOnCpu
                                     } // end if intersection ok
                                 } // end tx loop
                             } // end ty loop
                        } else {
                            // --- Striped Reading Logic (Protected) ---
                             uint32_t rows_per_strip = 0; uint32_t current_height32 = static_cast<uint32_t>(m_height);
                             TIFFGetFieldDefaulted(shared_tif_stack_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                             if (rows_per_strip == 0 || rows_per_strip > current_height32) { rows_per_strip = current_height32; }
                             const int chunk_width = m_width; // Strip width is image width

                             tsize_t strip_buffer_size = TIFFStripSize(shared_tif_stack_raw_ptr);
                             // Add checks for missing StripByteCounts / Compression if necessary
                             if (strip_buffer_size <= 0) { amrex::Abort("[TiffReader] FATAL: Invalid strip buffer size."); }
                             if (temp_buffer.size() < static_cast<size_t>(strip_buffer_size)) { temp_buffer.resize(strip_buffer_size); }

                             int strip_y_min = tile_box.smallEnd(1); int strip_y_max = tile_box.bigEnd(1);
                             tstrip_t first_strip = TIFFComputeStrip(shared_tif_stack_raw_ptr, strip_y_min, 0);
                             tstrip_t last_strip = TIFFComputeStrip(shared_tif_stack_raw_ptr, strip_y_max, 0);

                             for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                                 uint32_t strip_origin_y_uint = strip * rows_per_strip;
                                 if (strip_origin_y_uint > std::numeric_limits<int>::max()) { amrex::Abort("Strip origin Y exceeds integer limits"); }
                                 int chunk_origin_y = static_cast<int>(strip_origin_y_uint);
                                 const int chunk_origin_x = 0;

                                 uint32_t strip_rows_this = std::min(rows_per_strip, current_height32 - strip_origin_y_uint);
                                 if (strip_rows_this == 0) continue;
                                 int chunk_height = static_cast<int>(strip_rows_this);

                                 amrex::Box chunk_abs_box(amrex::IntVect(chunk_origin_x, chunk_origin_y, k),
                                                          amrex::IntVect(chunk_origin_x + chunk_width - 1, chunk_origin_y + chunk_height - 1, k));
                                 amrex::Box intersection = tile_box & chunk_abs_box;

                                 if (intersection.ok()) {
                                     // Read strip data into thread-local buffer
                                     tsize_t bytes_read = TIFFReadEncodedStrip(shared_tif_stack_raw_ptr, strip, temp_buffer.data(), strip_buffer_size);
                                     if (bytes_read < 0) {
                                         std::string error_msg = "[TiffReader] FATAL: Error reading strip " + std::to_string(strip) + " slice " + std::to_string(k);
                                         amrex::Abort(error_msg.c_str());
                                     }

                                     // Process striped data immediately
                                     amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                         double value_as_double = 0.0;
                                         if (bits_per_sample_val == 1) { /* 1-bit logic */
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y; // j relative to strip start
                                              size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                              size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                              int bit_index_in_byte = linear_pixel_index_in_chunk % 8;
                                              if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                                   unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                                   int bit_value = (m_fill_order == FILLORDER_MSB2LSB) ? (packed_byte >> (7 - bit_index_in_byte)) & 1 : (packed_byte >> bit_index_in_byte) & 1;
                                                   value_as_double = static_cast<double>(bit_value);
                                              }
                                         } else { /* BPS >= 8 logic */
                                              int i_in_chunk = i - chunk_origin_x; int j_in_chunk = j - chunk_origin_y;
                                              size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;
                                              if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                                 const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                                 value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                              }
                                         }
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

    // Ensure all ranks have finished reading and processing before proceeding
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

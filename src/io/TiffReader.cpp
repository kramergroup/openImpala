#include "TiffReader.H" // Ensure this header is updated!

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
#include <AMReX_Utility.H>        // Included for AMREX_ALWAYS_ASSERT
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Bcast
#include <AMReX_MFIter.H>         // Needed for MFIter
#include <AMReX_Array4.H>         // Needed for Array4 access
#include <AMReX_Loop.H>           // For LoopOnCpu / amrex::ParallelFor


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
    // Note: Using memcpy is generally safer than direct casting for type punning
    // Note: AMREX_GPU_HOST_DEVICE allows potential use in ParallelFor later if needed
    // Note: This only handles the first sample if spp > 1

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
                uint64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); // Potential precision loss
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
                int64_t val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val); // Potential precision loss
            }
            break;
        case SAMPLEFORMAT_IEEEFP:
            if (bytes_per_sample == 4) {
                float val; std::memcpy(&val, byte_ptr, sizeof(val)); value = static_cast<double>(val);
            } else if (bytes_per_sample == 8) {
                double val; std::memcpy(&val, byte_ptr, sizeof(val)); value = val; // Already double
            }
            // Add float16/bfloat16 support here if needed
            break;
        // SAMPLEFORMAT_VOID, SAMPLEFORMAT_COMPLEXINT, SAMPLEFORMAT_COMPLEXIEEEFP typically not used for images
        default:
            // Unsupported format, return 0 or handle as error
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
// Default constructor initializes members
TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0),
    m_bits_per_sample(0), m_sample_format(0), m_samples_per_pixel(0),
    m_is_read(false), m_is_sequence(false), m_start_index(0), m_digits(1)
{}

// Constructor for single file - calls readFile to handle metadata
TiffReader::TiffReader(const std::string& filename) : TiffReader()
{
    // Store filename now, used by readFileInternal
    m_filename = filename;
    if (!readFile(filename)) { // readFile now only handles metadata
         throw std::runtime_error("TiffReader: Failed to read metadata from file: " + filename);
    }
}

// Constructor for sequence - calls readFileSequence to handle metadata
TiffReader::TiffReader(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix) : TiffReader()
{
    // Store sequence parameters now
    m_base_pattern = base_pattern;
    // num_files used for depth is set within readFileSequence
    m_start_index = start_index;
    m_digits = digits;
    m_suffix = suffix;

    if (!readFileSequence(base_pattern, num_files, start_index, digits, suffix)) {
         throw std::runtime_error("TiffReader: Failed to read metadata for sequence: " + base_pattern);
    }
}

// --- Metadata Getters (Mostly unchanged) ---
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) {
        return amrex::Box(); // Return empty box if metadata not read
    }
    // Box is cell-centered, index from 0 to dim-1
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

// --- readFile (Single Multi-Dir File - METADATA ONLY) ---
bool TiffReader::readFile(const std::string& filename)
{
    // Store filename, mark as not sequence
    m_filename = filename;
    m_is_sequence = false;
    m_base_pattern = ""; // Clear sequence info

    // Temporary variables for Rank 0 reading
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;

    // --- Rank 0 reads metadata ---
    if (amrex::ParallelDescriptor::IOProcessor()) {
        if (filename.empty()) {
            amrex::Print() << "Error: [TiffReader::readFile] No filename provided.\n";
            amrex::Abort("Aborting due to missing filename on IOProcessor");
        }

        TiffPtr tif(TIFFOpen(filename.c_str(), "r"), TiffCloser());
        if (!tif) {
            amrex::Print() << "Error: [TiffReader::readFile] Failed to open TIFF file: " << filename << "\n";
            amrex::Abort("Aborting due to file open failure on IOProcessor");
        }

        // Read metadata from the first directory
        uint32_t w32 = 0, h32 = 0;
        uint16_t planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
            !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
            amrex::Print() << "Error reading initial Width/Height from " << filename << "\n";
             amrex::Abort("Aborting due to Width/Height read failure on IOProcessor");
        }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

        width_r0 = static_cast<int>(w32);
        height_r0 = static_cast<int>(h32);

        // --- Modified Validation ---
        // Allow BPS=1, check common byte-aligned BPS, reject others for now
        bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
        if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps ||
            planar != PLANARCONFIG_CONTIG) // Add checks for supported fmt/spp if needed
        {
            amrex::Print() << "Error: [TiffReader::readFile] Invalid or unsupported format in first directory of file: " << filename
                           << " (Width=" << width_r0 << ", Height=" << height_r0 << ", BPS=" << bps_r0 << ", Planar=" << planar << ")\n";
            amrex::Abort("Aborting due to invalid format on IOProcessor");
        }
        // We accept tiled or stripped here, check happens during read

        // Count directories for depth
        depth_r0 = 0;
        TIFFSetDirectory(tif.get(), 0); // Ensure we start counting from the first directory
        do { depth_r0++; } while (TIFFReadDirectory(tif.get()));

        if (depth_r0 == 0) {
             amrex::Print() << "Error: No directories found in TIFF: " << filename << "\n";
             amrex::Abort("Aborting due to no directories found on IOProcessor");
        }
        // tif handle closed automatically by TiffPtr destructor
    } // End IOProcessor Block

    // --- Broadcast metadata from Rank 0 to all ranks ---
    // Broadcast integer/flag data
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence)}; // Include is_sequence flag
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // All ranks set their member variables from broadcasted integer data
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2];
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]);

    // Broadcast filename manually (size first, then data) using Bcast (lowercase)
    int root = amrex::ParallelDescriptor::IOProcessorNumber();
    if (amrex::ParallelDescriptor::IOProcessor()) {
        int string_len = static_cast<int>(m_filename.length()); // Get length on root
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root);
    } else {
        int string_len = 0;
        amrex::ParallelDescriptor::Bcast(&string_len, 1, root);
        m_filename.resize(string_len);
        amrex::ParallelDescriptor::Bcast(const_cast<char*>(m_filename.data()), string_len, root);
    }

    // Basic check after broadcast
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0) {
        amrex::Abort("TiffReader::readFile: Invalid metadata received after broadcast.");
    }

    m_is_read = true; // Metadata is now read and distributed
    return true;
}

// --- readFileSequence (METADATA ONLY) ---
bool TiffReader::readFileSequence(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix)
{
    // Store sequence parameters, mark as sequence
    m_base_pattern = base_pattern;
    m_start_index = start_index;
    m_digits = digits;
    m_suffix = suffix;
    m_is_sequence = true;
    m_filename = ""; // Clear single filename

    // Temporary variables for Rank 0 reading
    int width_r0=0, height_r0=0, depth_r0=0;
    uint16_t bps_r0=0, fmt_r0=0, spp_r0=0;
    std::string first_filename_r0 = "";

    // --- Rank 0 reads metadata from the FIRST file in the sequence ---
    if (amrex::ParallelDescriptor::IOProcessor()) {
         if (num_files <= 0 || digits <= 0 || base_pattern.empty()) {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid sequence parameters.\n";
            amrex::Abort("Aborting due to invalid sequence parameters on IOProcessor");
         }
         depth_r0 = num_files; // Depth is number of files

         first_filename_r0 = generateFilename(base_pattern, start_index, digits, suffix);

         TiffPtr tif(TIFFOpen(first_filename_r0.c_str(), "r"), TiffCloser());
         if (!tif) {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Failed to open FIRST file: " << first_filename_r0 << "\n";
            amrex::Abort("Aborting due to first file open failure on IOProcessor");
         }

         // Read metadata from the first file
         uint32_t w32 = 0, h32 = 0;
         uint16_t planar = PLANARCONFIG_CONTIG;
         if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
             !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
             amrex::Print() << "Error reading Width/Height from " << first_filename_r0 << "\n";
              amrex::Abort("Aborting due to Width/Height read failure on IOProcessor");
         }
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp_r0);
         TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

         width_r0 = static_cast<int>(w32);
         height_r0 = static_cast<int>(h32);

         // --- Modified Validation ---
         // Allow BPS=1, check common byte-aligned BPS, reject others for now
         bool valid_bps = (bps_r0 == 1 || bps_r0 == 8 || bps_r0 == 16 || bps_r0 == 32 || bps_r0 == 64);
         if (width_r0 <= 0 || height_r0 <= 0 || !valid_bps ||
             planar != PLANARCONFIG_CONTIG)
         {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid or unsupported format in first file: " << first_filename_r0
                           << " (Width=" << width_r0 << ", Height=" << height_r0 << ", BPS=" << bps_r0 << ", Planar=" << planar << ")\n";
            amrex::Abort("Aborting due to invalid format on IOProcessor");
         }
         // tif closed by RAII
    } // End IOProcessor Block

    // --- Broadcast metadata ---
    // Broadcast integer/flag data
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence), m_start_index, m_digits}; // Include sequence params
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // All ranks set members from broadcast integer data
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2];
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]);
    m_start_index       = idata[7];
    m_digits            = idata[8];

    // Broadcast string parameters manually (size first, then data) using Bcast (lowercase)
    int root = amrex::ParallelDescriptor::IOProcessorNumber();

    // Broadcast m_base_pattern
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

    // Broadcast m_suffix
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


    // Basic check after broadcast
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || (m_is_sequence && m_base_pattern.empty())) {
        amrex::Abort("TiffReader::readFileSequence: Invalid metadata received after broadcast.");
    }

    m_is_read = true; // Metadata is read and distributed
    return true;
}


// --- readDistributedIntoFab Method ---
// This is the private helper called by the public threshold methods
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

    // Store original BPS, calculate bytes needed for storage/pixel (rounds up for bit-based)
    const int bits_per_sample_val = m_bits_per_sample; // Keep original bps
    // Calculate bytes per pixel needed for storage/offset calculations.
    // Note: This assumes SAMPLESPERPIXEL (spp) = 1 for BPS=1 data. A more robust implementation
    // would handle combinations like SPP=3, BPS=1 if needed (e.g. 3 bits per pixel packed).
    // For now, assume BPS=1 implies SPP=1.
    const size_t bytes_per_pixel = (bits_per_sample_val < 8) ?
                                   (bits_per_sample_val * m_samples_per_pixel + 7) / 8
                                   : (bits_per_sample_val / 8) * m_samples_per_pixel;


    // Check original bits_per_sample value
    if (bits_per_sample_val == 0) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bits per sample is zero!");
    }

    // Shared handle only for single-stack files, opened outside parallel region
    TiffPtr shared_tif_stack_handle = nullptr;
    if (!m_is_sequence) {
        shared_tif_stack_handle = TiffPtr(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
        if (!shared_tif_stack_handle) {
            amrex::Abort("[TiffReader::readDistributedIntoFab] Failed to open shared TIFF file: " + m_filename);
        }
    }

// Use OpenMP parallelism if available and not in a GPU launch region
#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    {
        // Per-thread temporary buffer for reading strips or tiles
        std::vector<unsigned char> temp_buffer;
        // Per-thread TIFF handle for sequences to avoid race conditions
        TiffPtr sequence_tif_handle = nullptr;

        // Loop over boxes owned by this MPI rank (and potentially thread)
        for (amrex::MFIter mfi(dest_mf, true); mfi.isValid(); ++mfi) // Use tiling MFIter
        {
            amrex::Array4<int> fab_arr = dest_mf.array(mfi); // Get Array4 view for writing
            const amrex::Box& tile_box = mfi.tilebox(); // Get the box for this tile/patch

            // Determine the Z-range needed for this box
            const int k_min = tile_box.smallEnd(2);
            const int k_max = tile_box.bigEnd(2);

            // Loop through the necessary Z-slices (directories or files) for this box
            for (int k = k_min; k <= k_max; ++k) {
                TIFF* current_tif_raw_ptr = nullptr; // Raw pointer to use for LibTIFF calls

                // --- Get the TIFF handle for the current slice k ---
                if (m_is_sequence) {
                    std::string current_filename = generateFilename(m_base_pattern, m_start_index + k, m_digits, m_suffix);
                    sequence_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!sequence_tif_handle) {
                        amrex::Warning("[TiffReader] Failed to open sequence file: " + current_filename + ". Skipping slice " + std::to_string(k));
                        continue;
                    }
                    current_tif_raw_ptr = sequence_tif_handle.get();
                } else {
                    if (!shared_tif_stack_handle) {
                         amrex::Abort("[TiffReader] Shared stack handle is null inside parallel region!");
                    }
                    if (!TIFFSetDirectory(shared_tif_stack_handle.get(), static_cast<tdir_t>(k))) {
                        amrex::Warning("[TiffReader] Failed to set directory " + std::to_string(k) + " in " + m_filename + ". Skipping slice.");
                        continue;
                    }
                    current_tif_raw_ptr = shared_tif_stack_handle.get();
                }
                 if (!current_tif_raw_ptr) continue;

                // --- Read data for slice k into the fab for the intersecting region ---
                if (TIFFIsTiled(current_tif_raw_ptr)) {
                    // --- Tiled Reading Logic ---
                    uint32_t tile_width=0, tile_height=0;
                    TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILEWIDTH, &tile_width);
                    TIFFGetField(current_tif_raw_ptr, TIFFTAG_TILELENGTH, &tile_height);
                    if (tile_width == 0 || tile_height == 0) {
                        amrex::Warning("[TiffReader] Invalid tile dimensions for slice " + std::to_string(k) + ". Skipping.");
                        continue;
                    }
                    const int chunk_width = static_cast<int>(tile_width); // Define chunk width for loop

                    tsize_t tile_buffer_size = TIFFTileSize(current_tif_raw_ptr);
                    if (tile_buffer_size <= 0) {
                        amrex::Warning("[TiffReader] Invalid tile buffer size for slice " + std::to_string(k) + ". Skipping.");
                        continue;
                    }
                    if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) {
                        temp_buffer.resize(tile_buffer_size);
                    }

                    int tx_min = tile_box.smallEnd(0) / tile_width;
                    int tx_max = tile_box.bigEnd(0) / tile_width;
                    int ty_min = tile_box.smallEnd(1) / tile_height;
                    int ty_max = tile_box.bigEnd(1) / tile_height;

                    for (int ty = ty_min; ty <= ty_max; ++ty) {
                        for (int tx = tx_min; tx <= tx_max; ++tx) {
                            int tile_origin_x = tx * tile_width;
                            int tile_origin_y = ty * tile_height;
                            // Define origins needed in loop scope
                            const int chunk_origin_x = tile_origin_x;
                            const int chunk_origin_y = tile_origin_y;

                            ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, tile_origin_x, tile_origin_y, k, 0); // Z=k dummy

                            tsize_t bytes_read = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);

                            if (bytes_read <= 0) { // Check for error or empty read
                                if (bytes_read == -1) amrex::Warning("[TiffReader] Error reading tile index " + std::to_string(tile_index) + " for slice " + std::to_string(k));
                                continue;
                            }

                            amrex::Box tile_abs_box(amrex::IntVect(tile_origin_x, tile_origin_y, k),
                                                    amrex::IntVect(tile_origin_x + tile_width - 1, tile_origin_y + tile_height - 1, k));
                            amrex::Box intersection = tile_box & tile_abs_box;

                            if (intersection.ok()) {
                                amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                    double value_as_double = 0.0;
                                    if (bits_per_sample_val == 1) {
                                        // Handle 1-bit data
                                        int i_in_chunk = i - chunk_origin_x;
                                        int j_in_chunk = j - chunk_origin_y;
                                        size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                        size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                        int bit_index_in_byte = linear_pixel_index_in_chunk % 8;

                                        if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                             unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                             int bit_value = (packed_byte >> (7 - bit_index_in_byte)) & 1; // Assume MSB fill order
                                             value_as_double = static_cast<double>(bit_value);
                                        } // Else remains 0.0
                                    } else {
                                        // Handle byte-aligned data (BPS >= 8)
                                        const size_t bytes_per_sample = bits_per_sample_val / 8;
                                        int i_in_chunk = i - chunk_origin_x;
                                        int j_in_chunk = j - chunk_origin_y;
                                        size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;

                                        if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                             const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                             value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                        } // Else remains 0.0
                                    }
                                    fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                });
                            } // end if intersection ok
                        } // end loop tx
                    } // end loop ty

                } else {
                    // --- Striped Reading Logic ---
                    uint32_t rows_per_strip = 0;
                    uint32_t current_height32 = static_cast<uint32_t>(m_height);
                    TIFFGetFieldDefaulted(current_tif_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                    if (rows_per_strip == 0 || rows_per_strip > current_height32) {
                        rows_per_strip = current_height32;
                    }
                     // Define chunk width for loop
                    const int chunk_width = m_width;

                    tsize_t strip_buffer_size = TIFFStripSize(current_tif_raw_ptr);
                     if (strip_buffer_size <= 0) {
                        amrex::Warning("[TiffReader] Invalid strip buffer size for slice " + std::to_string(k) + ". Skipping.");
                        continue;
                    }
                    if (temp_buffer.size() < static_cast<size_t>(strip_buffer_size)) {
                        temp_buffer.resize(strip_buffer_size);
                    }

                    int strip_y_min = tile_box.smallEnd(1);
                    int strip_y_max = tile_box.bigEnd(1);
                    tstrip_t first_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_min, 0);
                    tstrip_t last_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_max, 0);

                    for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                        tsize_t bytes_read = TIFFReadEncodedStrip(current_tif_raw_ptr, strip, temp_buffer.data(), strip_buffer_size);

                        if (bytes_read <= 0) { // Check for error or empty read
                             if (bytes_read == -1) amrex::Warning("[TiffReader] Error reading strip " + std::to_string(strip) + " for slice " + std::to_string(k));
                             continue;
                         }

                        uint32_t strip_origin_y = strip * rows_per_strip;
                        uint32_t strip_rows_this = std::min(rows_per_strip, current_height32 - strip_origin_y);
                        uint32_t strip_end_y = strip_origin_y + strip_rows_this - 1;
                        // Define origin needed in loop scope
                        const int chunk_origin_x = 0;
                        const int chunk_origin_y = static_cast<int>(strip_origin_y);

                        amrex::Box strip_abs_box(amrex::IntVect(0, static_cast<int>(strip_origin_y), k),
                                                 amrex::IntVect(m_width - 1, static_cast<int>(strip_end_y), k));
                        amrex::Box intersection = tile_box & strip_abs_box;

                        if (intersection.ok()) {
                            amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop ) {
                                double value_as_double = 0.0; // Initialize
                                if (bits_per_sample_val == 1) {
                                    // Handle 1-bit data
                                    int i_in_chunk = i - chunk_origin_x; // i itself for strips
                                    int j_in_chunk = j - chunk_origin_y; // Y relative to strip start
                                    size_t linear_pixel_index_in_chunk = static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk;
                                    size_t byte_index_in_buffer = linear_pixel_index_in_chunk / 8;
                                    int bit_index_in_byte = linear_pixel_index_in_chunk % 8;

                                    if (byte_index_in_buffer < static_cast<size_t>(bytes_read)) {
                                         unsigned char packed_byte = temp_buffer[byte_index_in_buffer];
                                         int bit_value = (packed_byte >> (7 - bit_index_in_byte)) & 1; // Assume MSB fill order
                                         value_as_double = static_cast<double>(bit_value);
                                    } // Else remains 0.0
                                } else {
                                    // Handle byte-aligned data (BPS >= 8)
                                    const size_t bytes_per_sample = bits_per_sample_val / 8;
                                    int i_in_chunk = i - chunk_origin_x;
                                    int j_in_chunk = j - chunk_origin_y;
                                    size_t offset_in_buffer = (static_cast<size_t>(j_in_chunk) * chunk_width + i_in_chunk) * bytes_per_pixel;

                                    if (offset_in_buffer + bytes_per_sample <= static_cast<size_t>(bytes_read)) {
                                         const unsigned char* src_ptr = temp_buffer.data() + offset_in_buffer;
                                         value_as_double = interpretBytesAsDouble(src_ptr, bits_per_sample_val, m_sample_format);
                                    } // Else remains 0.0
                                }
                                fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                            });
                        } // end if intersection ok
                    } // end loop over strips
                } // end else (striped reading)

                // Close sequence file handle via RAII if looping
                if (m_is_sequence) {
                    sequence_tif_handle.reset();
                }

            } // End loop k (Z-slices)
        } // End MFIter loop
    } // End OMP parallel region

    // Close shared stack file handle via RAII (happens when shared_tif_stack_handle goes out of scope)

    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");
}

// Public threshold method with custom values - calls private helper
void TiffReader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    // Calls the private implementation function
    readDistributedIntoFab(mf, value_if_true, value_if_false, raw_threshold);
}


// Overload for 1/0 threshold output
void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    // Calls the 3-argument public version, which then calls the private helper.
    threshold(raw_threshold, 1, 0, mf);
}


} // namespace OpenImpala

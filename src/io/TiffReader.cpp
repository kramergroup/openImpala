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
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier, Broadcast
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

// Helper to interpret raw bytes based on TIFF metadata
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

    size_t bytes_per_sample = bits_per_sample / 8;
    double value = 0.0;

    // Ensure memcpy reads the correct number of bytes.
    // byte_ptr must be valid for at least bytes_per_sample bytes.

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
            // No easy way to signal failure to broadcast here, abort is safest
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

        // Basic validation of format read from file
        if (width_r0 <= 0 || height_r0 <= 0 || bps_r0 == 0 ||
            planar != PLANARCONFIG_CONTIG) // Add checks for supported bps/fmt/spp if needed
        {
            amrex::Print() << "Error: [TiffReader::readFile] Invalid or unsupported format in first directory of file: " << filename << "\n";
            amrex::Abort("Aborting due to invalid format on IOProcessor");
        }
        // We accept tiled or stripped here, check happens during read

        // Count directories for depth
        depth_r0 = 0;
        // Reset directory to beginning before counting
        TIFFSetDirectory(tif.get(), 0);
        do { depth_r0++; } while (TIFFReadDirectory(tif.get()));

        if (depth_r0 == 0) {
             amrex::Print() << "Error: No directories found in TIFF: " << filename << "\n";
             amrex::Abort("Aborting due to no directories found on IOProcessor");
        }
        // tif handle closed automatically by TiffPtr destructor
    } // End IOProcessor Block

    // --- Broadcast metadata from Rank 0 to all ranks ---
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence)}; // Include is_sequence flag
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // All ranks set their member variables from broadcasted data
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2];
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]);

    // Also broadcast the filename (now using correct string Broadcast)
    amrex::ParallelDescriptor::Broadcast(m_filename, amrex::ParallelDescriptor::IOProcessorNumber());


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

         // Basic validation
        if (width_r0 <= 0 || height_r0 <= 0 || bps_r0 == 0 ||
            planar != PLANARCONFIG_CONTIG)
        {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid or unsupported format in first file: " << first_filename_r0 << "\n";
            amrex::Abort("Aborting due to invalid format on IOProcessor");
        }
        // tif closed by RAII
    } // End IOProcessor Block

    // --- Broadcast metadata ---
    std::vector<int> idata = {width_r0, height_r0, depth_r0,
                              static_cast<int>(bps_r0), static_cast<int>(fmt_r0), static_cast<int>(spp_r0),
                              static_cast<int>(m_is_sequence), m_start_index, m_digits}; // Include sequence params
    amrex::ParallelDescriptor::Bcast(idata.data(), idata.size(), amrex::ParallelDescriptor::IOProcessorNumber());

    // Broadcast string parameters (now using correct string Broadcast)
    amrex::ParallelDescriptor::Broadcast(m_base_pattern, amrex::ParallelDescriptor::IOProcessorNumber());
    amrex::ParallelDescriptor::Broadcast(m_suffix, amrex::ParallelDescriptor::IOProcessorNumber());


    // All ranks set members from broadcast
    m_width             = idata[0];
    m_height            = idata[1];
    m_depth             = idata[2];
    m_bits_per_sample   = static_cast<uint16_t>(idata[3]);
    m_sample_format     = static_cast<uint16_t>(idata[4]);
    m_samples_per_pixel = static_cast<uint16_t>(idata[5]);
    m_is_sequence       = static_cast<bool>(idata[6]);
    m_start_index       = idata[7];
    m_digits            = idata[8];

    // Basic check after broadcast
    if (m_width <= 0 || m_height <= 0 || m_depth <= 0 || m_bits_per_sample == 0 || m_base_pattern.empty()) {
        amrex::Abort("TiffReader::readFileSequence: Invalid metadata received after broadcast.");
    }

    m_is_read = true; // Metadata is read and distributed
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

    // Calculate derived info needed by all ranks
    const size_t bytes_per_sample = m_bits_per_sample / 8;
    const size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
    if (bytes_per_sample == 0) {
        amrex::Abort("[TiffReader::readDistributedIntoFab] Bytes per sample is zero!");
    }

    // Shared handle only for single-stack files, opened outside parallel region
    // Use a raw pointer within the parallel region if needed.
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
                    // Open file for this slice - use the thread-local handle
                    sequence_tif_handle = TiffPtr(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
                    if (!sequence_tif_handle) {
                        amrex::Warning("[TiffReader] Failed to open sequence file: " + current_filename + ". Skipping slice " + std::to_string(k));
                        continue; // Skip this slice
                    }
                    current_tif_raw_ptr = sequence_tif_handle.get();
                    // TODO: Optional: Verify metadata consistency here if paranoid
                } else {
                    // Use the shared handle for the stack file
                    if (!shared_tif_stack_handle) { // Should have been opened outside parallel region
                         amrex::Abort("[TiffReader] Shared stack handle is null inside parallel region!");
                    }
                    if (!TIFFSetDirectory(shared_tif_stack_handle.get(), static_cast<tdir_t>(k))) {
                        amrex::Warning("[TiffReader] Failed to set directory " + std::to_string(k) + " in " + m_filename + ". Skipping slice.");
                        continue; // Skip this slice
                    }
                    current_tif_raw_ptr = shared_tif_stack_handle.get();
                }
                 // Ensure we have a valid pointer
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

                    tsize_t tile_buffer_size = TIFFTileSize(current_tif_raw_ptr);
                    if (tile_buffer_size <= 0) {
                        amrex::Warning("[TiffReader] Invalid tile buffer size for slice " + std::to_string(k) + ". Skipping.");
                        continue;
                    }
                    // Resize the thread-local buffer if necessary
                    if (temp_buffer.size() < static_cast<size_t>(tile_buffer_size)) {
                        temp_buffer.resize(tile_buffer_size);
                    }

                    // Calculate the range of tiles needed for tile_box
                    int tx_min = tile_box.smallEnd(0) / tile_width;
                    int tx_max = tile_box.bigEnd(0) / tile_width;
                    int ty_min = tile_box.smallEnd(1) / tile_height;
                    int ty_max = tile_box.bigEnd(1) / tile_height;

                    // Loop over required tiles in XY for the current slice k
                    for (int ty = ty_min; ty <= ty_max; ++ty) {
                        for (int tx = tx_min; tx <= tx_max; ++tx) {
                            // Calculate absolute tile origin coordinates
                            int tile_origin_x = tx * tile_width;
                            int tile_origin_y = ty * tile_height;

                            // Read the tile corresponding to (tx, ty) coordinates for slice k
                            ttile_t tile_index = TIFFComputeTile(current_tif_raw_ptr, tile_origin_x, tile_origin_y, k, 0); // Z, Sample=0 assumed

                            tsize_t bytes_read = TIFFReadEncodedTile(current_tif_raw_ptr, tile_index, temp_buffer.data(), tile_buffer_size);

                            if (bytes_read == -1) {
                                amrex::Warning("[TiffReader] Error reading tile index " + std::to_string(tile_index) + " for slice " + std::to_string(k));
                                continue; // Skip this tile
                            }
                             if (bytes_read == 0) {
                                amrex::Warning("[TiffReader] Read 0 bytes for tile index " + std::to_string(tile_index) + " slice " + std::to_string(k));
                                continue; // Skip empty tile
                            }

                            // Define the box covered by this specific tile
                            amrex::Box tile_abs_box(amrex::IntVect(tile_origin_x, tile_origin_y, k),
                                                    amrex::IntVect(tile_origin_x + tile_width - 1, tile_origin_y + tile_height - 1, k));

                            // Find the intersection of the current MFIter tile_box and the tile we just read
                            amrex::Box intersection = tile_box & tile_abs_box;

                            if (intersection.ok()) {
                                // Loop over the intersection region and copy/threshold data
                                amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop /*same as k*/) {
                                    // Calculate offset within the temporary tile buffer
                                    int i_in_tile = i - tile_origin_x;
                                    int j_in_tile = j - tile_origin_y;
                                    size_t offset_in_buffer = (static_cast<size_t>(j_in_tile) * tile_width + i_in_tile) * bytes_per_pixel;

                                    // Bounds check (optional but recommended)
                                    if (offset_in_buffer + bytes_per_sample > static_cast<size_t>(bytes_read)) return;

                                    // Interpret bytes, threshold, and store in destination Fab using Array4
                                    double value_as_double = interpretBytesAsDouble(temp_buffer.data() + offset_in_buffer,
                                                                                    m_bits_per_sample, m_sample_format);
                                    // Use corrected Array4 access syntax
                                    fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                                });
                            } // end if intersection ok
                        } // end loop tx
                    } // end loop ty

                } else {
                    // --- Striped Reading Logic ---
                    uint32_t rows_per_strip = 0;
                    uint32_t current_height32 = static_cast<uint32_t>(m_height); // Use current height for strip calcs
                    TIFFGetFieldDefaulted(current_tif_raw_ptr, TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
                    if (rows_per_strip == 0 || rows_per_strip > current_height32) {
                        rows_per_strip = current_height32;
                    }

                    tsize_t strip_buffer_size = TIFFStripSize(current_tif_raw_ptr);
                     if (strip_buffer_size <= 0) {
                        amrex::Warning("[TiffReader] Invalid strip buffer size for slice " + std::to_string(k) + ". Skipping.");
                        continue;
                    }
                    // Resize the thread-local buffer if necessary
                    if (temp_buffer.size() < static_cast<size_t>(strip_buffer_size)) {
                        temp_buffer.resize(strip_buffer_size);
                    }

                    tstrip_t num_strips = TIFFNumberOfStrips(current_tif_raw_ptr);

                    // Calculate the range of strips needed for tile_box
                    int strip_y_min = tile_box.smallEnd(1);
                    int strip_y_max = tile_box.bigEnd(1);
                    tstrip_t first_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_min, 0);
                    tstrip_t last_strip = TIFFComputeStrip(current_tif_raw_ptr, strip_y_max, 0);

                    // Loop over required strips for the current slice k
                    for (tstrip_t strip = first_strip; strip <= last_strip; ++strip) {
                        tsize_t bytes_read = TIFFReadEncodedStrip(current_tif_raw_ptr, strip, temp_buffer.data(), strip_buffer_size);

                        if (bytes_read == -1) {
                            amrex::Warning("[TiffReader] Error reading strip " + std::to_string(strip) + " for slice " + std::to_string(k));
                            continue; // Skip this strip
                        }
                         if (bytes_read == 0) {
                             amrex::Warning("[TiffReader] Read 0 bytes for strip " + std::to_string(strip) + " slice " + std::to_string(k));
                             continue; // Skip empty strip
                         }

                        // Define the box covered by this specific strip
                        uint32_t strip_origin_y = strip * rows_per_strip;
                        uint32_t strip_end_y = std::min(strip_origin_y + rows_per_strip, current_height32) -1;
                         amrex::Box strip_abs_box(amrex::IntVect(0, static_cast<int>(strip_origin_y), k),
                                                 amrex::IntVect(m_width - 1, static_cast<int>(strip_end_y), k));

                        // Find the intersection of the current MFIter tile_box and the strip we just read
                        amrex::Box intersection = tile_box & strip_abs_box;

                        if (intersection.ok()) {
                            // Loop over the intersection region and copy/threshold data
                            amrex::LoopOnCpu(intersection, [&](int i, int j, int k_loop /*same as k*/) {
                                // Calculate offset within the temporary strip buffer
                                int j_in_strip = j - static_cast<int>(strip_origin_y); // Y relative to strip start
                                size_t offset_in_buffer = (static_cast<size_t>(j_in_strip) * m_width + i) * bytes_per_pixel;

                                // Bounds check (optional but recommended)
                                if (offset_in_buffer + bytes_per_sample > static_cast<size_t>(bytes_read)) return;

                                // Interpret bytes, threshold, and store in destination Fab using Array4
                                double value_as_double = interpretBytesAsDouble(temp_buffer.data() + offset_in_buffer,
                                                                                m_bits_per_sample, m_sample_format);
                                // Use corrected Array4 access syntax
                                fab_arr(i, j, k_loop) = (value_as_double > raw_threshold) ? value_if_true : value_if_false;
                            });
                        } // end if intersection ok
                    } // end loop over strips
                } // end else (striped reading)

                // Close sequence file handle via RAII if looping
                if (m_is_sequence) {
                    sequence_tif_handle.reset(); // Explicitly close here if needed, or rely on loop destruction
                }

            } // End loop k (Z-slices)
        } // End MFIter loop
    } // End OMP parallel region

    // Close shared stack file handle via RAII (happens when shared_tif_stack_handle goes out of scope)

    // Barrier ensures all ranks finish reading before proceeding
    amrex::ParallelDescriptor::Barrier("TiffReader::readDistributedIntoFab");

    // Data is now in dest_mf. The reader class itself might not need to be marked "read" anymore,
    // as its purpose is fulfilled by filling the MultiFab. Keep m_is_read for metadata status.
}

// Overload for 1/0 threshold output
void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    // Calls the main function with default values for true/false
    readDistributedIntoFab(mf, 1, 0, raw_threshold);
}


} // namespace OpenImpala

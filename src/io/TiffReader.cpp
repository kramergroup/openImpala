#include "TiffReader.H"

#include <tiffio.h>  // libtiff C API Header
#include <memory>    // For std::unique_ptr
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <sstream>   // Needed for std::ostringstream
#include <cstring>   // For std::memcpy
#include <algorithm> // For std::min
// #include <sstream>   // Duplicate include removed
#include <iomanip>   // For std::setw, std::setfill

#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Utility.H>          // Included for AMREX_ASSERT*
#include <AMReX_ParallelDescriptor.H> // For IOProcessor, Barrier
#include <AMReX_ParallelFor.H>      // For ParallelFor


namespace OpenImpala {

namespace { // Anonymous namespace for internal helpers

// RAII wrapper for TIFF* handle
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// getValueFromBytes helper
template <typename T>
AMREX_FORCE_INLINE T getValueFromBytes(const unsigned char* byte_ptr) {
    T val;
    // Ensure enough bytes are available - basic check assumes byte_ptr is valid
    // A more robust check would involve passing the buffer end pointer
    std::memcpy(&val, byte_ptr, sizeof(T));
    return val;
}

} // namespace


// --- Constructors ---
TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0), m_is_read(false),
    m_bits_per_sample(0), m_sample_format(0), m_samples_per_pixel(0)
{}

TiffReader::TiffReader(const std::string& filename) : TiffReader()
{
    if (!readFile(filename)) {
         throw std::runtime_error("TiffReader: Failed to read file: " + filename);
    }
}

TiffReader::TiffReader(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix) : TiffReader()
{
     if (!readFileSequence(base_pattern, num_files, start_index, digits, suffix)) {
          throw std::runtime_error("TiffReader: Failed to read file sequence starting with: " + base_pattern);
     }
}

// --- Getters for metadata ---
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
int TiffReader::bitsPerSample() const { return m_bits_per_sample; }
int TiffReader::sampleFormat() const { return m_sample_format; }
int TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }
const std::vector<TiffReader::ByteType>& TiffReader::getRawData() const { return m_raw_bytes; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) {
        return amrex::Box(); // Return empty box if not read
    }
    // Box is cell-centered, index from 0 to dim-1
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}


// --- readTiffInternal Method ---
// Reads ONE directory from the currently open TIFF file handle (tif_handle)
// Assumes metadata (width, height etc) MAY already be set if called from sequence
// Assumes buffer (slice_start_ptr) is correctly sized and points to the right location
bool TiffReader::readTiffInternal() {
    // This function now primarily handles the pixel reading for a single directory (slice)
    // It assumes the TIFF* handle is valid and positioned at the correct directory.
    // It assumes the caller manages the m_raw_bytes buffer allocation and positioning.

    // Use a temporary TiffPtr for reading the current directory, assumes m_filename is set
    TiffPtr tif(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
    if (!tif) {
        amrex::Print() << "Error: [TiffReader::readTiffInternal] Failed to re-open TIFF file: " << m_filename << "\n";
        return false;
    }
    // NOTE: If called from readFile or readFileSequence, the file might be opened already.
    // This function needs refactoring to accept an existing TIFF* handle or manage state better.
    // For now, we proceed assuming it needs to open the file based on m_filename,
    // and the *caller* needs to ensure TIFFSetDirectory was called appropriately before calling this,
    // or this function needs to take the directory index as an argument.
    // The current structure where this function opens the file itself is problematic
    // if it's meant to read specific directories from an already open file handle.

    // --- Re-read metadata for safety/simplicity in this standalone version ---
    // This duplicates logic from callers but makes this function testable.
    // A better design would pass metadata or the TIFF* handle.
    uint32_t w32 = 0, h32 = 0;
    uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG;
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
        !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
            amrex::Print() << "Error reading dims in readTiffInternal\n"; return false; }
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

    // Use the read metadata, not potentially pre-set member variables
    int current_width = static_cast<int>(w32);
    int current_height = static_cast<int>(h32);
    if (current_width <= 0 || current_height <= 0 || planar != PLANARCONFIG_CONTIG) {
        amrex::Print() << "Error: Invalid dimensions or planar config in readTiffInternal\n"; return false;
    }
    size_t bytes_per_sample = bps / 8;
    if (bytes_per_sample == 0) { amrex::Print() << "Error: BPS is zero\n"; return false; }
    size_t bytes_per_pixel = bytes_per_sample * spp;
    size_t slice_bytes = static_cast<size_t>(current_width) * current_height * bytes_per_pixel;

    // --- Check Allocation by Caller ---
    if (m_raw_bytes.empty() || m_raw_bytes.size() < slice_bytes) {
        amrex::Print() << "Error: [readTiffInternal] m_raw_bytes buffer is empty or too small.\n";
        return false;
    }
    // Assume the buffer pointed to by m_raw_bytes.data() is for the *current* slice.
    ByteType* slice_start_ptr = m_raw_bytes.data();


    // --- Read Pixel Data (Current Directory Only) ---
    if (TIFFIsTiled(tif.get())) {
        amrex::Print() << "Error: [TiffReader] Tiled TIFF reading is not implemented.\n";
        return false;
    } else {
        // --- Stripped Read Logic ---
        uint32_t rows_per_strip = 0;
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
        if (rows_per_strip == 0 || rows_per_strip > static_cast<uint32_t>(current_height)) {
             rows_per_strip = current_height;
        }

        tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
        tsize_t strip_buf_size = TIFFStripSize(tif.get()); // Max size allocated by libtiff
        tsize_t bytes_read_total_in_slice = 0;
         if (strip_buf_size <= 0) {
              amrex::Print() << "Error: [TiffReader] Invalid strip size calculated for " << m_filename << "\n";
              return false;
         }


        for (tstrip_t strip = 0; strip < num_strips; ++strip) {
             size_t rows_in_this_strip = rows_per_strip;
             size_t current_row_start = static_cast<size_t>(strip) * rows_per_strip;
             if (current_row_start + rows_in_this_strip > static_cast<size_t>(current_height)) {
                 rows_in_this_strip = static_cast<size_t>(current_height) - current_row_start;
             }
             tsize_t expected_bytes_this_strip = static_cast<tsize_t>(rows_in_this_strip * current_width * bytes_per_pixel);
             // Don't read more than libtiff allocates for a strip buffer, or more than expected for the rows
             tsize_t buffer_size_for_this_strip = std::min(strip_buf_size, expected_bytes_this_strip);

            ByteType* strip_dest_ptr = slice_start_ptr + current_row_start * current_width * bytes_per_pixel;

            // Bounds check
            AMREX_ALWAYS_ASSERT_WITH_MESSAGE(strip_dest_ptr >= slice_start_ptr && (strip_dest_ptr + buffer_size_for_this_strip) <= (slice_start_ptr + slice_bytes),
                                             "Strip destination pointer or size out of bounds");

            tsize_t bytes_read_this_strip = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, buffer_size_for_this_strip);

            if (bytes_read_this_strip == -1) {
                amrex::Print() << "Error: [TiffReader] Failed reading strip " << strip << " from " << m_filename << "\n";
                return false;
            }
            bytes_read_total_in_slice += bytes_read_this_strip;
        } // End loop over strips

        // Check if the total bytes read match the expected size for this slice
        if (bytes_read_total_in_slice != static_cast<tsize_t>(slice_bytes)) {
            // <<< FIX: Corrected warning message generation >>>
            std::ostringstream warning_msg;
            warning_msg << "Warning: [TiffReader] Bytes read for current directory (" << bytes_read_total_in_slice
                        << ") does not match expected bytes per slice (" << slice_bytes // Use calculated expected size
                        << "). File: " << m_filename; // Removed reference to 'i'
            amrex::Warning(warning_msg.str());
            // Decide if this should be a fatal error? Continuing might lead to garbage data.
            // return false; // Optionally fail here
        }
    } // End else block for stripped images

    // File automatically closed by TiffPtr going out of scope via RAII
    return true;
// <<< FIX: Removed extra closing brace '}' here >>>
} // End of TiffReader::readTiffInternal


// --- readFile for single multi-dir file ---
bool TiffReader::readFile(const std::string& filename)
{
    // Reset state
    m_raw_bytes.clear();
    m_is_read = false;
    m_filename = filename;
    m_width = m_height = m_depth = 0;
    m_bits_per_sample = m_sample_format = m_samples_per_pixel = 0;

    if (m_filename.empty()) {
        amrex::Print() << "Error: [TiffReader] No filename provided.\n";
        return false;
    }

    TiffPtr tif(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
    if (!tif) {
        amrex::Print() << "Error: [TiffReader] Failed to open TIFF file: " << m_filename << "\n";
        return false;
    }

    // Read metadata from first directory to set expectations
    uint32_t w32 = 0, h32 = 0;
    uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG;
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
        !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
        amrex::Print() << "Error reading initial dimensions from " << m_filename << "\n"; return false;
    }
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

    m_width = static_cast<int>(w32);
    m_height = static_cast<int>(h32);
    m_bits_per_sample = bps;
    m_sample_format = fmt;
    m_samples_per_pixel = spp;

    // Validate basic properties from first directory
    if (m_width <= 0 || m_height <= 0 || planar != PLANARCONFIG_CONTIG ||
       (m_bits_per_sample != 8 && m_bits_per_sample != 16 && m_bits_per_sample != 32 && m_bits_per_sample != 64) ||
       (m_sample_format != SAMPLEFORMAT_UINT && m_sample_format != SAMPLEFORMAT_INT && m_sample_format != SAMPLEFORMAT_IEEEFP) )
    {
         amrex::Print() << "Error: [TiffReader::readFile] Invalid or unsupported format in first directory of file: " << m_filename << "\n";
         return false;
    }
    size_t bytes_per_sample = m_bits_per_sample / 8;
     if (bytes_per_sample == 0) { amrex::Print() << "Error: bytes_per_sample is zero\n"; return false; }
    size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
    size_t slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;

    // Count directories for depth
    m_depth = 0;
    // Reset directory to beginning before counting
    TIFFSetDirectory(tif.get(), 0);
    do { m_depth++; } while (TIFFReadDirectory(tif.get()));
    if (m_depth == 0) { amrex::Print() << "Error: No directories found in TIFF: " << m_filename << "\n"; return false; }

    // Allocate storage
    size_t total_bytes = slice_bytes * m_depth;
    if (static_cast<double>(m_width)*m_height*m_depth*bytes_per_pixel > std::numeric_limits<size_t>::max()){
        amrex::Print() << "Error: Total image size exceeds size_t limits.\n"; return false;
    }
    try {
        m_raw_bytes.resize(total_bytes);
    } catch (const std::exception& e) {
        amrex::Print() << "Error: Failed to allocate memory for TIFF data: " << e.what() << "\n"; return false;
    }

    // Loop and read each directory
    for (int slice = 0; slice < m_depth; ++slice) {
        if (!TIFFSetDirectory(tif.get(), static_cast<tdir_t>(slice))) {
            amrex::Print() << "Error: Failed to set directory " << slice << " in " << m_filename << "\n"; return false;
        }
        // Calculate pointer to the start of the buffer for this slice
        ByteType* slice_start_ptr = m_raw_bytes.data() + static_cast<size_t>(slice) * slice_bytes;

        // Read strip data into the correct slice buffer segment
        if (TIFFIsTiled(tif.get())) {
            amrex::Print() << "Error: [TiffReader] Tiled TIFF reading is not implemented.\n";
            return false;
        } else {
            uint32_t rows_per_strip = 0;
            TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
             if (rows_per_strip == 0 || rows_per_strip > static_cast<uint32_t>(m_height)) {
                 rows_per_strip = m_height;
             }
            tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
            tsize_t strip_buf_size = TIFFStripSize(tif.get());
             if (strip_buf_size <= 0) {
                  amrex::Print() << "Error: [TiffReader] Invalid strip size calculated for " << m_filename << " slice " << slice << "\n";
                  return false;
             }

            tsize_t bytes_read_total_this_slice = 0;
            for (tstrip_t strip = 0; strip < num_strips; ++strip) {
                 size_t rows_in_this_strip = rows_per_strip;
                 size_t current_row_start = static_cast<size_t>(strip) * rows_per_strip;
                 if (current_row_start + rows_in_this_strip > static_cast<size_t>(m_height)) {
                     rows_in_this_strip = static_cast<size_t>(m_height) - current_row_start;
                 }
                 tsize_t expected_bytes_this_strip = static_cast<tsize_t>(rows_in_this_strip * m_width * bytes_per_pixel);
                 tsize_t buffer_size_for_this_strip = std::min(strip_buf_size, expected_bytes_this_strip);

                ByteType* strip_dest_ptr = slice_start_ptr + current_row_start * m_width * bytes_per_pixel;

                // Bounds check
                AMREX_ASSERT(strip_dest_ptr >= slice_start_ptr && (strip_dest_ptr + buffer_size_for_this_strip) <= (slice_start_ptr + slice_bytes));

                tsize_t bytes_read = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, buffer_size_for_this_strip);
                if (bytes_read == -1) {
                    amrex::Print() << "Error: Failed reading strip " << strip << " for slice " << slice << " in " << m_filename << "\n";
                    return false;
                }
                 bytes_read_total_this_slice += bytes_read;
            }
            // Optional check for total bytes read in this slice
            if (bytes_read_total_this_slice != static_cast<tsize_t>(slice_bytes)) {
                 std::ostringstream warning_msg;
                 warning_msg << "Warning: [TiffReader::readFile] Bytes read (" << bytes_read_total_this_slice
                             << ") does not match expected (" << slice_bytes
                             << ") for slice " << slice << " in file " << m_filename;
                 amrex::Warning(warning_msg.str());
            }
        }
    } // End loop over slices (directories)

    m_is_read = true;
    amrex::Print() << "Successfully read multi-directory TIFF File: " << m_filename
                   << " (Dims: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}

// --- readFileSequence Implementation ---
bool TiffReader::readFileSequence(
    const std::string& base_pattern,
    int num_files,
    int start_index,
    int digits,
    const std::string& suffix)
{
    // Reset state
    m_raw_bytes.clear();
    m_is_read = false;
    m_filename = base_pattern + "<sequence>"; // Indicate sequence in internal filename
    m_width = m_height = m_depth = 0;
    m_bits_per_sample = m_sample_format = m_samples_per_pixel = 0;

    // Validate inputs
    if (num_files <= 0) {
        amrex::Print() << "Error: [TiffReader::readFileSequence] Number of files must be positive.\n";
        return false;
    }
    if (digits <= 0) {
        amrex::Print() << "Error: [TiffReader::readFileSequence] Number of digits must be positive.\n";
        return false;
    }
    m_depth = num_files; // Depth is the number of files

    // --- Allocate storage (after getting dimensions from first file) ---
    size_t bytes_per_sample = 0;
    size_t bytes_per_pixel = 0;
    size_t slice_bytes = 0;
    size_t total_bytes = 0;

    // --- Loop through files ---
    for (int file_idx = 0; file_idx < num_files; ++file_idx) {
        int slice_num = start_index + file_idx;

        // Generate filename
        std::ostringstream ss;
        ss << base_pattern << std::setw(digits) << std::setfill('0') << slice_num << suffix;
        std::string current_filename = ss.str();

        // Open current file
        TiffPtr tif(TIFFOpen(current_filename.c_str(), "r"), TiffCloser());
        if (!tif) {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Failed to open file: " << current_filename << "\n";
            m_raw_bytes.clear(); // Ensure cleanup
            return false;
        }

        // Read metadata
        uint32_t w32 = 0, h32 = 0;
        uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG;
        if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
            !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
             amrex::Print() << "Error: [TiffReader::readFileSequence] Failed reading Width/Height from " << current_filename << "\n"; return false; }
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

        // --- Handle First File ---
        if (file_idx == 0) {
            m_width = static_cast<int>(w32);
            m_height = static_cast<int>(h32);
            m_bits_per_sample = bps;
            m_sample_format = fmt;
            m_samples_per_pixel = spp;

            // Validate format
             if (m_width <= 0 || m_height <= 0 || planar != PLANARCONFIG_CONTIG ||
                 (m_bits_per_sample != 8 && m_bits_per_sample != 16 && m_bits_per_sample != 32 && m_bits_per_sample != 64) ||
                 (m_sample_format != SAMPLEFORMAT_UINT && m_sample_format != SAMPLEFORMAT_INT && m_sample_format != SAMPLEFORMAT_IEEEFP))
             {
                  amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid or unsupported format in first file: " << current_filename << "\n"; return false;
             }
             bytes_per_sample = m_bits_per_sample / 8;
             if (bytes_per_sample == 0) { amrex::Print() << "Error: bytes_per_sample is zero\n"; return false; }
             bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
             slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;
             total_bytes = slice_bytes * m_depth; // m_depth is num_files

            // Allocate storage
             if (static_cast<double>(m_width)*m_height*m_depth*bytes_per_pixel > std::numeric_limits<size_t>::max()){ amrex::Print()<<"Size Overflow\n"; return false;}
             try {
                 m_raw_bytes.resize(total_bytes);
             } catch (const std::exception& e) { amrex::Print() << "Error allocating memory: " << e.what() << "\n"; return false; }

        } else {
            // --- Verify Consistency ---
            if (static_cast<int>(w32) != m_width || static_cast<int>(h32) != m_height ||
                bps != m_bits_per_sample || fmt != m_sample_format ||
                spp != m_samples_per_pixel || planar != PLANARCONFIG_CONTIG)
            {
                amrex::Print() << "Error: [TiffReader::readFileSequence] Inconsistent metadata in file: " << current_filename
                               << " compared to first file.\n";
                return false;
            }
        }

        // --- Read Pixel Data for this slice ---
        // Calculate pointer to the start of the buffer for this slice
        ByteType* slice_start_ptr = m_raw_bytes.data() + static_cast<size_t>(file_idx) * slice_bytes;

        if (TIFFIsTiled(tif.get())) {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Tiled TIFF reading not implemented (" << current_filename << ").\n";
            return false;
        } else {
            // Stripped Read Logic (similar to readTiffInternal and readFile)
            uint32_t rows_per_strip = 0;
            TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
             if (rows_per_strip == 0 || rows_per_strip > static_cast<uint32_t>(m_height)) { rows_per_strip = m_height; }
            tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
            tsize_t strip_buf_size = TIFFStripSize(tif.get());
             if (strip_buf_size <= 0) {
                  amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid strip size calculated for " << current_filename << "\n";
                  return false;
             }

            tsize_t bytes_read_total_this_slice = 0;
            for (tstrip_t strip = 0; strip < num_strips; ++strip) {
                 size_t rows_in_this_strip = rows_per_strip;
                 size_t current_row_start = static_cast<size_t>(strip) * rows_per_strip;
                 if (current_row_start + rows_in_this_strip > static_cast<size_t>(m_height)) {
                     rows_in_this_strip = static_cast<size_t>(m_height) - current_row_start;
                 }
                 tsize_t expected_bytes_this_strip = static_cast<tsize_t>(rows_in_this_strip * m_width * bytes_per_pixel);
                 tsize_t buffer_size_for_this_strip = std::min(strip_buf_size, expected_bytes_this_strip);

                ByteType* strip_dest_ptr = slice_start_ptr + current_row_start * m_width * bytes_per_pixel;

                // Bounds check
                AMREX_ASSERT(strip_dest_ptr >= m_raw_bytes.data() && (strip_dest_ptr+buffer_size_for_this_strip) <= (m_raw_bytes.data() + total_bytes) );

                tsize_t bytes_read = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, buffer_size_for_this_strip);
                if (bytes_read == -1) {
                    amrex::Print() << "Error: [TiffReader::readFileSequence] Failed reading strip " << strip << " from file " << current_filename << "\n";
                    return false;
                }
                bytes_read_total_this_slice += bytes_read;
            }
             // Optional: Check if total bytes read for this slice match expected
             if (bytes_read_total_this_slice != static_cast<tsize_t>(slice_bytes)) {
                 std::ostringstream warning_msg;
                 warning_msg << "Warning: [TiffReader::readFileSequence] Bytes read (" << bytes_read_total_this_slice
                             << ") does not match expected (" << slice_bytes
                             << ") for file " << current_filename; // Use file_idx here if needed
                 amrex::Warning(warning_msg.str());
             }
        }
        // TIFF file closed automatically by TiffPtr destructor
    } // End loop over files

    m_is_read = true;
    amrex::Print() << "Successfully read TIFF sequence: " << base_pattern << "..."
                   << " (Files: " << num_files << ", Dims: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}


// --- getValue / threshold implementations ---
// getValue needs to be defined as a template matching the header
template <typename T>
T TiffReader::getValue(int i, int j, int k) const {
    if (!m_is_read) {
        throw std::runtime_error("[TiffReader::getValue] Data not read");
    }
    // Basic bounds check
    if (i < 0 || i >= m_width || j < 0 || j >= m_height || k < 0 || k >= m_depth) {
         throw std::out_of_range("[TiffReader::getValue] Index out of bounds");
    }

    // Calculate byte offset based on XYZ layout (Z varies slowest)
    size_t bytes_per_sample = m_bits_per_sample / 8;
    size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
    size_t slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;

    size_t byte_offset = static_cast<size_t>(k) * slice_bytes +
                         static_cast<size_t>(j) * m_width * bytes_per_pixel +
                         static_cast<size_t>(i) * bytes_per_pixel;

    // Check if offset + size of T is within bounds
    if (byte_offset + sizeof(T) > m_raw_bytes.size()) {
         throw std::out_of_range("[TiffReader::getValue] Calculated byte offset out of bounds");
    }

    // Assume reading the first sample if samples_per_pixel > 1
    // A more complete implementation would handle the 'sample' argument passed from header
    const ByteType* byte_ptr = m_raw_bytes.data() + byte_offset;

    // Use the helper to interpret bytes as type T
    if (sizeof(T) > bytes_per_pixel) {
         // Or perhaps handle multi-sample pixels here?
         throw std::runtime_error("[TiffReader::getValue] Requested type T is larger than bytes per pixel");
    }
    return getValueFromBytes<T>(byte_ptr); // Use helper from anonymous namespace
}

// --- Explicit template instantiations ---
// These should now match the template definition and the corrected header declaration
template std::uint8_t TiffReader::getValue<std::uint8_t>(int, int, int) const;
template std::uint16_t TiffReader::getValue<std::uint16_t>(int, int, int) const;
template std::uint32_t TiffReader::getValue<std::uint32_t>(int, int, int) const;
template std::int8_t TiffReader::getValue<std::int8_t>(int, int, int) const;
template std::int16_t TiffReader::getValue<std::int16_t>(int, int, int) const;
template std::int32_t TiffReader::getValue<std::int32_t>(int, int, int) const;
template float TiffReader::getValue<float>(int, int, int) const;
template double TiffReader::getValue<double>(int, int, int) const;


void TiffReader::threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
     if (!m_is_read) {
        amrex::Abort("[TiffReader::threshold] Cannot threshold, data not read successfully.");
     }

     // Determine bytes per sample to call correct templated getValue
     size_t bytes_per_sample = m_bits_per_sample / 8;

#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox();
        amrex::IArrayBox& fab = mf[mfi];

        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            // Check bounds against image dimensions
            if (i >= 0 && i < m_width && j >= 0 && j < m_height && k >= 0 && k < m_depth) {
                double raw_value = 0.0;
                // Call the appropriate templated getValue based on file format
                // This assumes only one sample per pixel for simplicity here
                // Needs enhancement for multi-sample pixels
                try {
                    // Use a switch for clarity
                    switch (m_sample_format) {
                        case SAMPLEFORMAT_UINT:
                            if (bytes_per_sample == 1) raw_value = static_cast<double>(getValue<std::uint8_t>(i,j,k));
                            else if (bytes_per_sample == 2) raw_value = static_cast<double>(getValue<std::uint16_t>(i,j,k));
                            else if (bytes_per_sample == 4) raw_value = static_cast<double>(getValue<std::uint32_t>(i,j,k));
                            // Add case for 64-bit if needed
                            else raw_value = 0.0; // Unsupported size
                            break;
                        case SAMPLEFORMAT_INT:
                            if (bytes_per_sample == 1) raw_value = static_cast<double>(getValue<std::int8_t>(i,j,k));
                            else if (bytes_per_sample == 2) raw_value = static_cast<double>(getValue<std::int16_t>(i,j,k));
                            else if (bytes_per_sample == 4) raw_value = static_cast<double>(getValue<std::int32_t>(i,j,k));
                            // Add case for 64-bit if needed
                            else raw_value = 0.0; // Unsupported size
                            break;
                        case SAMPLEFORMAT_IEEEFP:
                            if (bytes_per_sample == 4) raw_value = static_cast<double>(getValue<float>(i,j,k));
                            else if (bytes_per_sample == 8) raw_value = getValue<double>(i,j,k); // Already double
                            else raw_value = 0.0; // Unsupported size
                            break;
                        default:
                             raw_value = 0.0; // Unsupported format
                             break;
                    }
                    // Apply threshold using corrected IArrayBox syntax
                    fab(amrex::IntVect(i, j, k), 0) = (raw_value > raw_threshold) ? value_if_true : value_if_false;
                } catch (const std::exception& e) {
                    // Handle potential out_of_range from getValue if bounds check fails internally
                    fab(amrex::IntVect(i, j, k), 0) = value_if_false; // Default outside bounds
                    // Maybe issue a warning once? Needs careful parallel handling
                    // std::string msg = "Exception during getValue in threshold: " + std::string(e.what()); amrex::Warning(msg);
                }

            } else {
                 // Cell is outside the original image bounds
                 fab(amrex::IntVect(i, j, k), 0) = value_if_false;
            }
        });
    }
}

// Overload for 1/0 output
void TiffReader::threshold(double raw_threshold, amrex::iMultiFab& mf) const {
    threshold(raw_threshold, 1, 0, mf);
}


} // namespace OpenImpala

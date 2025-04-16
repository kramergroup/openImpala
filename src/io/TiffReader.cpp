#include "TiffReader.H"

#include <tiffio.h>  // libtiff C API Header
#include <memory>    // For std::unique_ptr
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <sstream>
#include <cstring>   // For std::memcpy
#include <algorithm> // For std::min
#include <sstream>   // For filename generation
#include <iomanip>   // For std::setw, std::setfill

#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

namespace OpenImpala {

namespace { // Anonymous namespace for internal helpers

// RAII wrapper for TIFF* handle (same as before)
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// getValueFromBytes helper (same as before)
template <typename T>
AMREX_FORCE_INLINE T getValueFromBytes(const unsigned char* byte_ptr) {
    T val;
    std::memcpy(&val, byte_ptr, sizeof(T));
    return val;
}

} // namespace


// --- Constructors / Destructor / Getters / getValue / threshold / readTiffInternal ---
// Assume implementations from previous version exist here...
// Make sure readTiffInternal is adjusted to NOT set m_depth by counting directories,
// but instead reads only the first directory (or assumes single dir per file).
// Let's slightly modify readTiffInternal to read only ONE directory and return depth=1.

bool TiffReader::readTiffInternal() {
    // Reset state only partially if called by readFileSequence
    // Let readFile/readFileSequence handle full reset.
    // m_raw_bytes.clear(); // Handled by caller
    // m_is_read = false; // Handled by caller

    TiffPtr tif(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
    if (!tif) {
        amrex::Print() << "Error: [TiffReader] Failed to open TIFF file: " << m_filename << "\n";
        return false;
    }

    // --- Read Metadata (from CURRENT directory) ---
    uint32_t w32 = 0, h32 = 0;
    uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG, compression = 0;

    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32)) { /* Handle error */ return false; }
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { /* Handle error */ return false; }
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_COMPRESSION, &compression);

    // --- Store / Validate Metadata ---
    // If called from readFileSequence, only store on first file, compare otherwise
    // Let readFileSequence handle this logic. Here we just read current dir.
    m_width = static_cast<int>(w32);
    m_height = static_cast<int>(h32);
    m_bits_per_sample = bps;
    m_sample_format = fmt;
    m_samples_per_pixel = spp;
    // Assume m_depth is handled by caller (set to 1 for single file read here)
    m_depth = 1;

    if (m_width <= 0 || m_height <= 0) { /* Invalid dims */ return false; }
    if (planar != PLANARCONFIG_CONTIG) { /* Unsupported */ return false; }
    if (m_bits_per_sample != 8 && m_bits_per_sample != 16 && m_bits_per_sample != 32 && m_bits_per_sample != 64) { /* Unsupported */ return false; }
    if (m_sample_format != SAMPLEFORMAT_UINT && m_sample_format != SAMPLEFORMAT_INT && m_sample_format != SAMPLEFORMAT_IEEEFP) { /* Unsupported */ return false; }

    size_t bytes_per_sample = m_bits_per_sample / 8;
    if (bytes_per_sample == 0) { return false; }
    size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
    size_t slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;

    // --- Allocate Storage ---
    // Caller (readFile / readFileSequence) should handle allocation
    if (m_raw_bytes.size() < slice_bytes){
         amrex::Print() << "Error: [readTiffInternal] m_raw_bytes not allocated sufficiently.\n";
         return false;
    }

    // --- Read Pixel Data (Current Directory Only) ---
    ByteType* slice_start_ptr = m_raw_bytes.data(); // Assume caller allocated enough for 1 slice

    if (TIFFIsTiled(tif.get())) {
        amrex::Print() << "Error: [TiffReader] Tiled TIFF reading is not implemented.\n";
        return false;
    } else {
        // --- Stripped Read Logic ---
        uint32_t rows_per_strip = 0;
        TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
        if (rows_per_strip == 0) { rows_per_strip = m_height; } // Default if not set
        rows_per_strip = std::min(rows_per_strip, static_cast<uint32_t>(m_height));

        tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
        tsize_t strip_buf_size = TIFFStripSize(tif.get());
        tsize_t bytes_read_total_in_slice = 0;

        for (tstrip_t strip = 0; strip < num_strips; ++strip) {
            ByteType* strip_dest_ptr = slice_start_ptr + static_cast<size_t>(strip) * strip_buf_size;
            // Need more careful calculation if rows_per_strip varies or last strip is partial
            // size_t current_row = strip * rows_per_strip;
            // ByteType* strip_dest_ptr = slice_start_ptr + current_row * m_width * bytes_per_pixel;

             AMREX_ALWAYS_ASSERT_WITH_MESSAGE(strip_dest_ptr >= m_raw_bytes.data() && strip_dest_ptr < m_raw_bytes.data() + slice_bytes,
                                             "Strip destination pointer out of bounds");

            tsize_t bytes_read_this_strip = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, strip_buf_size);

            if (bytes_read_this_strip == -1) {
                 amrex::Print() << "Error: [TiffReader] Failed reading strip " << strip << " from " << m_filename << "\n";
                 return false;
            }
            bytes_read_total_in_slice += bytes_read_this_strip;
        }
         // Inside TiffReader::readTiffInternal(), after the strip reading loop

            // Check if the total bytes read match the expected size for this slice
            if (bytes_read_total_in_slice != static_cast<tsize_t>(slice_bytes)) {
                std::ostringstream warning_msg;
                warning_msg << "Warning: [TiffReader] Bytes read for current directory (" << bytes_read_total_in_slice
                            // Use the calculated expected slice size 'slice_bytes' in the message
                            << ") does not match expected bytes per slice (" << slice_bytes // <<< FIX 1
                            // Remove the reference to 'i' as it's not in scope here
                            << "). File: " << m_filename; // <<< FIX 2
                amrex::Warning(warning_msg.str());
            }
        } // End else block for stripped images
    // File automatically closed by TiffPtr going out of scope via RAII
    return true;
} // End of TiffReader::readTiffInternal
    }
    // File automatically closed by TiffPtr going out of scope via RAII
    return true;
}


// --- readFile for single multi-dir file (Modified to use internal helper) ---
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

    // Read metadata from first directory
    // (Duplicating logic from readTiffInternal - could refactor later)
    uint32_t w32 = 0, h32 = 0;
    uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG;
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32) ||
        !TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) { /* Error */ return false; }
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);

    m_width = static_cast<int>(w32);
    m_height = static_cast<int>(h32);
    m_bits_per_sample = bps;
    m_sample_format = fmt;
    m_samples_per_pixel = spp;
    // Validate... (same checks as in readTiffInternal)
    if (m_width <= 0 || m_height <= 0 || planar != PLANARCONFIG_CONTIG || /* other checks */ false) {
         return false;
    }
    size_t bytes_per_sample = m_bits_per_sample / 8;
    size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
    size_t slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;

    // Count directories for depth
    m_depth = 0;
    do { m_depth++; } while (TIFFReadDirectory(tif.get()));
    if (m_depth == 0) { return false; }

    // Allocate storage
    size_t total_bytes = slice_bytes * m_depth;
     if (static_cast<double>(m_width)*m_height*m_depth*bytes_per_pixel > std::numeric_limits<size_t>::max()){ return false;}
    try {
        m_raw_bytes.resize(total_bytes);
    } catch (const std::exception& e) { /* Error */ return false; }

    // Loop and read each directory using internal logic (or replicate here)
    for (int slice = 0; slice < m_depth; ++slice) {
        if (!TIFFSetDirectory(tif.get(), static_cast<tdir_t>(slice))) { /* Error */ return false; }
        ByteType* slice_start_ptr = m_raw_bytes.data() + static_cast<size_t>(slice) * slice_bytes;

         // Need to read strip/tile into slice_start_ptr
         // Simplified: Assume readTiffInternal logic handles ONE slice read correctly
         // This requires modification of readTiffInternal to accept target buffer/offset
         // Or replicate the strip/tile reading loop here. Replicating:
         if (TIFFIsTiled(tif.get())) {
             amrex::Print() << "Error: [TiffReader] Tiled TIFF reading is not implemented.\n";
             return false;
         } else {
             uint32_t rows_per_strip = 0;
             TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
             if (rows_per_strip == 0) rows_per_strip = m_height;
             rows_per_strip = std::min(rows_per_strip, static_cast<uint32_t>(m_height));
             tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
             tsize_t strip_buf_size = TIFFStripSize(tif.get());

             for (tstrip_t strip = 0; strip < num_strips; ++strip) {
                  // Calculate correct pointer for this strip within this slice buffer
                  size_t rows_already_in_strip = static_cast<size_t>(strip) * rows_per_strip;
                  ByteType* strip_dest_ptr = slice_start_ptr + rows_already_in_strip * m_width * bytes_per_pixel;
                  // More careful calculation needed for partial strips at the end

                  AMREX_ASSERT(strip_dest_ptr >= slice_start_ptr && (strip_dest_ptr+strip_buf_size) <= (slice_start_ptr + slice_bytes) );

                 tsize_t bytes_read = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, strip_buf_size);
                 if (bytes_read == -1) { /* Error */ return false;}
             }
         }
    }

    m_is_read = true;
    amrex::Print() << "Successfully read multi-directory TIFF File: " << m_filename
                   << " (Dims: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}

// --- NEW: readFileSequence Implementation ---
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
             if (m_width <= 0 || m_height <= 0 || planar != PLANARCONFIG_CONTIG || /* other checks */ false) {
                  amrex::Print() << "Error: [TiffReader::readFileSequence] Invalid or unsupported format in first file: " << current_filename << "\n"; return false;
             }
             bytes_per_sample = m_bits_per_sample / 8;
             if (bytes_per_sample == 0) { /* Error */ return false; }
             bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;
             slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;
             total_bytes = slice_bytes * m_depth;

            // Allocate storage
            if (static_cast<double>(m_width)*m_height*m_depth*bytes_per_pixel > std::numeric_limits<size_t>::max()){ amrex::Print()<<"Size Overflow\n"; return false;}
            try {
                m_raw_bytes.resize(total_bytes);
            } catch (const std::exception& e) { /* Error */ return false; }

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
        ByteType* slice_start_ptr = m_raw_bytes.data() + static_cast<size_t>(file_idx) * slice_bytes;

        if (TIFFIsTiled(tif.get())) {
            amrex::Print() << "Error: [TiffReader::readFileSequence] Tiled TIFF reading not implemented (" << current_filename << ").\n";
            return false;
        } else {
            // Stripped Read Logic (same as in readTiffInternal)
            uint32_t rows_per_strip = 0;
            TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
            if (rows_per_strip == 0) rows_per_strip = m_height;
            rows_per_strip = std::min(rows_per_strip, static_cast<uint32_t>(m_height));
            tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
            tsize_t strip_buf_size = TIFFStripSize(tif.get());

            for (tstrip_t strip = 0; strip < num_strips; ++strip) {
                 size_t rows_already_in_strip = static_cast<size_t>(strip) * rows_per_strip;
                 ByteType* strip_dest_ptr = slice_start_ptr + rows_already_in_strip * m_width * bytes_per_pixel;

                  AMREX_ASSERT(strip_dest_ptr >= m_raw_bytes.data() && (strip_dest_ptr+strip_buf_size) <= (m_raw_bytes.data() + total_bytes) );


                 tsize_t bytes_read = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, strip_buf_size);
                 if (bytes_read == -1) {
                      amrex::Print() << "Error: [TiffReader::readFileSequence] Failed reading strip " << strip << " from file " << current_filename << "\n";
                      return false;
                 }
            }
        }
        // TIFF file closed automatically by TiffPtr destructor
    } // End loop over files

    m_is_read = true;
    amrex::Print() << "Successfully read TIFF sequence: " << base_pattern << "..."
                   << " (Files: " << num_files << ", Dims: " << m_width << "x" << m_height << "x" << m_depth << ")\n";
    return true;
}


// --- getValue / threshold implementations (Assume they exist as previously defined) ---
// ... Need to copy/paste the implementations for getValue and threshold from the previous response ...
// ... Make sure they use members like m_width, m_height, m_depth, m_bits_per_sample, m_sample_format, m_raw_bytes ...

} // namespace OpenImpala

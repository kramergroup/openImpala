#include "TiffReader.H"

#include <tiffio.h>  // libtiff C API Header
#include <memory>    // For std::unique_ptr
#include <vector>
#include <string>
#include <stdexcept>
#include <limits>
#include <cmath>
#include <cstring>   // For std::memcpy
#include <algorithm> // For std::min (needed?)

#include <AMReX_Print.H>
#include <AMReX_Box.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_GpuContainers.H>
#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>

// Note: Do not include H5Cpp.h here

namespace OpenImpala {

namespace { // Anonymous namespace for internal helpers

// RAII wrapper for TIFF* handle
struct TiffCloser {
    void operator()(TIFF* tif) const {
        if (tif) TIFFClose(tif);
    }
};
using TiffPtr = std::unique_ptr<TIFF, TiffCloser>;

// Helper to get value using memcpy for type punning safety
template <typename T>
AMREX_FORCE_INLINE T getValueFromBytes(const unsigned char* byte_ptr) {
    T val;
    std::memcpy(&val, byte_ptr, sizeof(T));
    return val;
}

} // namespace


//-----------------------------------------------------------------------
// Constructor / Destructor Implementations
//-----------------------------------------------------------------------

TiffReader::TiffReader() :
    m_width(0), m_height(0), m_depth(0),
    m_bits_per_sample(0), m_sample_format(0), m_samples_per_pixel(0),
    m_is_read(false)
{
    // Default constructor initializes members
}

TiffReader::TiffReader(const std::string& filename) :
    TiffReader() // Delegate for default initialization
{
    if (!readFile(filename)) {
        // readFile should have printed details
        throw std::runtime_error("TiffReader: Failed to read TIFF file '" + filename + "' during construction.");
    }
}

// Default virtual destructor is sufficient as TiffPtr handles cleanup via RAII
// TiffReader::~TiffReader() = default;

//-----------------------------------------------------------------------
// File Reading Implementation
//-----------------------------------------------------------------------

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

    // Call internal implementation
    bool success = readTiffInternal();
    m_is_read = success;

    if (success) {
         amrex::Print() << "Successfully read TIFF File: " << m_filename
                        << " (Dims: " << m_width << "x" << m_height << "x" << m_depth
                        << ", Format: " << m_sample_format
                        << ", Bits: " << m_bits_per_sample
                        << ", Samples/Pixel: " << m_samples_per_pixel << ")\n";
    } else {
        // Ensure state reflects failure
        m_raw_bytes.clear();
        m_width = m_height = m_depth = 0;
        m_bits_per_sample = m_sample_format = m_samples_per_pixel = 0;
        amrex::Print() << "Error: [TiffReader] Failed to read file: " << m_filename << "\n";
    }
    return success;
}


bool TiffReader::readTiffInternal()
{
    // Use unique_ptr for automatic TIFFClose
    TiffPtr tif(TIFFOpen(m_filename.c_str(), "r"), TiffCloser());
    if (!tif) {
        amrex::Print() << "Error: [TiffReader] Failed to open TIFF file (TIFFOpen failed): " << m_filename << "\n";
        return false;
    }

    // --- Read Metadata (from first directory/IFD) ---
    // Use TIFFGetFieldDefaulted where appropriate for robustness
    uint32_t w32 = 0, h32 = 0;
    uint16_t bps = 0, fmt = SAMPLEFORMAT_UINT, spp = 1, planar = PLANARCONFIG_CONTIG, compression = 0;

    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &w32)) {
        amrex::Print() << "Error: [TiffReader] Failed to read TIFFTAG_IMAGEWIDTH.\n"; return false; }
    if (!TIFFGetField(tif.get(), TIFFTAG_IMAGELENGTH, &h32)) {
        amrex::Print() << "Error: [TiffReader] Failed to read TIFFTAG_IMAGELENGTH.\n"; return false; }
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_BITSPERSAMPLE, &bps);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLEFORMAT, &fmt);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_SAMPLESPERPIXEL, &spp);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_PLANARCONFIG, &planar);
    TIFFGetFieldDefaulted(tif.get(), TIFFTAG_COMPRESSION, &compression); // Read compression type

    // Store metadata in members
    m_width = static_cast<int>(w32);
    m_height = static_cast<int>(h32);
    m_bits_per_sample = bps;
    m_sample_format = fmt;
    m_samples_per_pixel = spp;

    // --- Validate Format ---
    if (m_width <= 0 || m_height <= 0) {
        amrex::Print() << "Error: [TiffReader] Invalid dimensions read (W="<<m_width<<", H="<<m_height<<").\n"; return false; }
    if (planar != PLANARCONFIG_CONTIG) {
        amrex::Print() << "Error: [TiffReader] Unsupported PlanarConfiguration (!= CONTIG).\n"; return false; }
    if (m_bits_per_sample != 8 && m_bits_per_sample != 16 && m_bits_per_sample != 32 && m_bits_per_sample != 64) {
        amrex::Print() << "Error: [TiffReader] Unsupported BitsPerSample: " << m_bits_per_sample << ".\n"; return false; }
    if (m_sample_format != SAMPLEFORMAT_UINT && m_sample_format != SAMPLEFORMAT_INT && m_sample_format != SAMPLEFORMAT_IEEEFP) {
        amrex::Print() << "Error: [TiffReader] Unsupported SampleFormat: " << m_sample_format << ".\n"; return false; }
    // Add more validation? Check combination of format and bits?

    size_t bytes_per_sample = m_bits_per_sample / 8;
    if (bytes_per_sample == 0) { // Should be caught by bps check above
         amrex::Print() << "Error: [TiffReader] Invalid BitsPerSample resulting in zero bytes per sample.\n"; return false;
    }
    size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;

    // --- Determine Depth (Number of Directories/Slices) ---
    m_depth = 0;
    do {
        m_depth++;
        // Optional: Could re-read metadata here and verify consistency across directories
        // uint32_t current_w=0; TIFFGetField(tif.get(), TIFFTAG_IMAGEWIDTH, &current_w);
        // if (current_w != static_cast<uint32_t>(m_width)) { /* Inconsistent! */ }
    } while (TIFFReadDirectory(tif.get()));

    if (m_depth == 0) { // Should have at least one directory
         amrex::Print() << "Error: [TiffReader] No directories (slices) found in TIFF file.\n"; return false;
    }

    // --- Allocate Storage ---
    size_t slice_bytes = static_cast<size_t>(m_width) * m_height * bytes_per_pixel;
    size_t total_bytes = slice_bytes * m_depth;
    // Check for potential overflow before resize
    if (static_cast<double>(m_width)*m_height*m_depth*bytes_per_pixel > std::numeric_limits<size_t>::max()){
        amrex::Print() << "Error: [TiffReader] Total data size exceeds maximum vector capacity.\n";
        return false; // Don't attempt resize
    }
    try {
        m_raw_bytes.resize(total_bytes);
    } catch (const std::exception& e) {
        amrex::Print() << "Error: [TiffReader] Failed to allocate memory (" << total_bytes
                       << " bytes) for raw data: " << e.what() << "\n";
        // tif handle closed automatically by TiffPtr going out of scope
        return false;
    }

    // --- Read Pixel Data (Loop through directories again) ---
    for (int slice = 0; slice < m_depth; ++slice) {
        if (!TIFFSetDirectory(tif.get(), static_cast<tdir_t>(slice))) {
             amrex::Print() << "Error: [TiffReader] Failed to set directory to slice " << slice << "\n";
             return false; // RAII handles TIFFClose
        }

        ByteType* slice_start_ptr = m_raw_bytes.data() + static_cast<size_t>(slice) * slice_bytes;

        if (TIFFIsTiled(tif.get())) {
            // --- Tiled Read Logic (Placeholder - Requires Implementation) ---
            amrex::Print() << "Error: [TiffReader] Tiled TIFF reading is not implemented.\n";
            return false; // RAII handles TIFFClose

        } else {
            // --- Stripped Read Logic ---
            uint32_t rows_per_strip = 0;
            TIFFGetFieldDefaulted(tif.get(), TIFFTAG_ROWSPERSTRIP, &rows_per_strip);
            if (rows_per_strip == 0) { // Should not happen? Default might be large
                 amrex::Print() << "Error: [TiffReader] RowsPerStrip tag is zero.\n"; return false; }
            rows_per_strip = std::min(rows_per_strip, static_cast<uint32_t>(m_height)); // Clamp to image height

            tstrip_t num_strips = TIFFNumberOfStrips(tif.get());
            tsize_t strip_buf_size = TIFFStripSize(tif.get()); // Size of ONE decompressed strip
            tsize_t bytes_read_total_in_slice = 0;

            for (tstrip_t strip = 0; strip < num_strips; ++strip) {
                 // Calculate destination pointer for this strip within the slice
                 // Note: This assumes strips cover the image contiguously top-to-bottom
                 ByteType* strip_dest_ptr = slice_start_ptr + static_cast<size_t>(strip) * strip_buf_size;

                 // Safety check for destination pointer
                 if (strip_dest_ptr < m_raw_bytes.data() || strip_dest_ptr >= m_raw_bytes.data() + total_bytes) {
                      amrex::Print() << "Error: [TiffReader] Strip destination pointer out of bounds.\n"; return false;
                 }

                 // Use TIFFReadEncodedStrip to read *and decompress* data.
                 tsize_t bytes_read_this_strip = TIFFReadEncodedStrip(tif.get(), strip, strip_dest_ptr, strip_buf_size);

                 if (bytes_read_this_strip == -1) {
                      amrex::Print() << "Error: [TiffReader] Failed reading strip " << strip << " from slice " << slice << " in " << m_filename << "\n";
                      return false; // RAII handles TIFFClose
                 }
                 bytes_read_total_in_slice += bytes_read_this_strip;
            }
             // Optional: Check if total bytes read matches expected slice size
             if (bytes_read_total_in_slice != static_cast<tsize_t>(slice_bytes)) {
                   amrex::Print() << "Warning: [TiffReader] Bytes read for slice " << slice << " (" << bytes_read_total_in_slice
                                  << ") does not match expected slice size (" << slice_bytes << ").\n";
                   // Decide if this is a fatal error or just a warning
             }
        }
    } // End loop over slices

    // File automatically closed by TiffPtr destructor via RAII
    return true;
}


//-----------------------------------------------------------------------
// Getter Implementations
//-----------------------------------------------------------------------

bool TiffReader::isRead() const { return m_is_read; }
int TiffReader::width() const { return m_width; }
int TiffReader::height() const { return m_height; }
int TiffReader::depth() const { return m_depth; }
uint16_t TiffReader::bitsPerSample() const { return m_bits_per_sample; }
uint16_t TiffReader::sampleFormat() const { return m_sample_format; }
uint16_t TiffReader::samplesPerPixel() const { return m_samples_per_pixel; }

amrex::Box TiffReader::box() const {
    if (!m_is_read) return amrex::Box();
    return amrex::Box(amrex::IntVect(0, 0, 0), amrex::IntVect(m_width - 1, m_height - 1, m_depth - 1));
}

//-----------------------------------------------------------------------
// Data Access Implementation (`getValue`)
//-----------------------------------------------------------------------

double TiffReader::getValue(int i, int j, int k, int sample) const {
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_read, "[TiffReader::getValue] Data not read yet.");

    // --- Bounds Check ---
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        i >= 0 && i < m_width && j >= 0 && j < m_height && k >= 0 && k < m_depth
        && sample >= 0 && sample < m_samples_per_pixel,
        "[TiffReader::getValue] Index (" + std::to_string(i) + "," + std::to_string(j) + "," + std::to_string(k)
        + ", sample=" + std::to_string(sample) + ") out of bounds (W:" + std::to_string(m_width)
        + ", H:" + std::to_string(m_height) + ", D:" + std::to_string(m_depth)
        + ", SPP:" + std::to_string(m_samples_per_pixel) + ")."
    );

    // --- Calculate Byte Offset ---
    const size_t bytes_per_sample = getBytesPerVoxelSample(); // Bytes per single sample value
    if (bytes_per_sample == 0) {
        amrex::Abort("[TiffReader::getValue] Invalid bytes per sample (0).");
    }
    const size_t bytes_per_pixel = bytes_per_sample * m_samples_per_pixel;

    // Index assumes PLANARCONFIG_CONTIG (e.g., RGBRGBRGB...) and XYZ data layout (Z varies slowest)
    size_t idx_1d = static_cast<size_t>(k) * m_height * m_width +
                    static_cast<size_t>(j) * m_width +
                    static_cast<size_t>(i);
    size_t pixel_offset = idx_1d * bytes_per_pixel;
    size_t sample_offset_within_pixel = static_cast<size_t>(sample) * bytes_per_sample;
    size_t offset = pixel_offset + sample_offset_within_pixel;

    // --- Check Offset vs Buffer Size ---
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(
        (offset + bytes_per_sample) <= m_raw_bytes.size(),
        "[TiffReader::getValue] Calculated byte offset (" + std::to_string(offset)
        + ") + size (" + std::to_string(bytes_per_sample)
        + ") exceeds raw data vector size (" + std::to_string(m_raw_bytes.size()) + ")."
    );

    // --- Reconstruct Value ---
    // libtiff usually handles endianness on read, so no swap needed here.
    const ByteType* src_ptr = m_raw_bytes.data() + offset;
    double result = 0.0;

    // Use memcpy for type punning safety, convert to double
    switch (m_sample_format) {
        case SAMPLEFORMAT_UINT:
            if (m_bits_per_sample == 8)       { result = static_cast<double>(getValueFromBytes<uint8_t>(src_ptr)); }
            else if (m_bits_per_sample == 16) { result = static_cast<double>(getValueFromBytes<uint16_t>(src_ptr)); }
            else if (m_bits_per_sample == 32) { result = static_cast<double>(getValueFromBytes<uint32_t>(src_ptr)); }
            // Add 64 if needed and supported by libtiff/your data
            else { amrex::Abort("[TiffReader::getValue] Unsupported UINT BitsPerSample."); }
            break;
        case SAMPLEFORMAT_INT:
             if (m_bits_per_sample == 8)      { result = static_cast<double>(getValueFromBytes<int8_t>(src_ptr)); }
             else if (m_bits_per_sample == 16) { result = static_cast<double>(getValueFromBytes<int16_t>(src_ptr)); }
             else if (m_bits_per_sample == 32) { result = static_cast<double>(getValueFromBytes<int32_t>(src_ptr)); }
             // Add 64 if needed
             else { amrex::Abort("[TiffReader::getValue] Unsupported INT BitsPerSample."); }
             break;
        case SAMPLEFORMAT_IEEEFP:
             if (m_bits_per_sample == 32) {
                 static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "");
                 result = static_cast<double>(getValueFromBytes<float>(src_ptr));
             } else if (m_bits_per_sample == 64) {
                 static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8, "");
                 result = getValueFromBytes<double>(src_ptr); // Already double
             } else { amrex::Abort("[TiffReader::getValue] Unsupported IEEEFP BitsPerSample."); }
             break;
        // SAMPLEFORMAT_VOID, SAMPLEFORMAT_COMPLEXINT, SAMPLEFORMAT_COMPLEXIEEEFP not handled
        default:
             amrex::Abort("[TiffReader::getValue] Unknown or unsupported SampleFormat.");
    }
    return result;
}

//-----------------------------------------------------------------------
// Threshold Implementation
//-----------------------------------------------------------------------

// Helper to get value as double for thresholding (minimal error checking)
// AMREX_FORCE_INLINE double getValueAsDouble(const unsigned char* raw_bytes_ptr, size_t offset, uint16_t format, uint16_t bps)
// {
//      // Simplified reconstruction (assuming offset/bps/format valid)
//      // This could replace direct logic in threshold if kept simple
//      const unsigned char* src = raw_bytes_ptr + offset;
//      switch(format){ /* similar switch as getValue */ }
//      return 0.0; // Default
// }


// Overload with customizable true/false values
void TiffReader::threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const
{
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(m_is_read, "[TiffReader::threshold] Cannot threshold, data not read successfully.");
    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(mf.nComp() == 1, "[TiffReader::threshold] Output iMultiFab must have 1 component.");

    // Capture members needed by the lambda
    const int current_width = m_width;
    const int current_height = m_height;
    const int current_depth = m_depth;
    const uint16_t current_format = m_sample_format;
    const uint16_t current_bps = m_bits_per_sample;
    const uint16_t current_spp = m_samples_per_pixel; // Samples per pixel
    const size_t bytes_per_sample = getBytesPerVoxelSample();
    const size_t bytes_per_pixel = bytes_per_sample * current_spp;
    const size_t raw_bytes_size = m_raw_bytes.size();
    const ByteType* const raw_bytes_ptr = m_raw_bytes.data();

    AMREX_ALWAYS_ASSERT_WITH_MESSAGE(bytes_per_sample > 0, "Thresholding requires valid bytes_per_sample");


#ifdef AMREX_USE_OMP
#pragma omp parallel if (amrex::Gpu::notInLaunchRegion())
#endif
    for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi)
    {
        const amrex::Box& box = mfi.validbox(); // Get the valid box for this FAB
        amrex::IArrayBox& fab = mf[mfi];        // Get the FAB to write to

        amrex::LoopOnCpu(box, [&] (int i, int j, int k)
        {
            // --- Bounds Check for Raw Data Access ---
            if (AMREX_LIKELY(i >= 0 && i < current_width && j >= 0 && j < current_height && k >= 0 && k < current_depth))
            {
                // --- Calculate Offset for Sample 0 ---
                size_t idx_1d = static_cast<size_t>(k) * current_height * current_width +
                                static_cast<size_t>(j) * current_width +
                                static_cast<size_t>(i);
                size_t offset = idx_1d * bytes_per_pixel; // Offset to start of pixel
                // Assuming we threshold on the first sample (sample = 0)
                // offset += 0 * bytes_per_sample;

                // --- Check Calculated Offset (Safety) ---
                if (AMREX_UNLIKELY((offset + bytes_per_sample) > raw_bytes_size)) {
                    amrex::Abort("[TiffReader::threshold] Internal error: Calculated offset exceeds bounds.");
                }

                // --- Inline Reconstruction (Sample 0) & Comparison ---
                const ByteType* src_ptr = raw_bytes_ptr + offset;
                bool comparison_result = false;
                double value_as_double = 0.0; // Value converted to double for comparison

                 // Reconstruct value from bytes based on stored format/bps
                 // Similar logic to getValue() but streamlined
                 switch (current_format) {
                    case SAMPLEFORMAT_UINT:
                        if (current_bps == 8)       { value_as_double = static_cast<double>(getValueFromBytes<uint8_t>(src_ptr)); }
                        else if (current_bps == 16) { value_as_double = static_cast<double>(getValueFromBytes<uint16_t>(src_ptr)); }
                        else if (current_bps == 32) { value_as_double = static_cast<double>(getValueFromBytes<uint32_t>(src_ptr)); }
                        else { amrex::Abort("[TiffReader::threshold] Unsupported UINT BitsPerSample."); }
                        break;
                    case SAMPLEFORMAT_INT:
                         if (current_bps == 8)      { value_as_double = static_cast<double>(getValueFromBytes<int8_t>(src_ptr)); }
                         else if (current_bps == 16) { value_as_double = static_cast<double>(getValueFromBytes<int16_t>(src_ptr)); }
                         else if (current_bps == 32) { value_as_double = static_cast<double>(getValueFromBytes<int32_t>(src_ptr)); }
                         else { amrex::Abort("[TiffReader::threshold] Unsupported INT BitsPerSample."); }
                         break;
                    case SAMPLEFORMAT_IEEEFP:
                         if (current_bps == 32) {
                             static_assert(std::numeric_limits<float>::is_iec559 && sizeof(float) == 4, "");
                             value_as_double = static_cast<double>(getValueFromBytes<float>(src_ptr));
                         } else if (current_bps == 64) {
                             static_assert(std::numeric_limits<double>::is_iec559 && sizeof(double) == 8, "");
                             value_as_double = getValueFromBytes<double>(src_ptr); // Already double
                         } else { amrex::Abort("[TiffReader::threshold] Unsupported IEEEFP BitsPerSample."); }
                         break;
                    default:
                         amrex::Abort("[TiffReader::threshold] Unknown or unsupported SampleFormat.");
                 }

                comparison_result = (value_as_double > threshold_value);
                fab(i, j, k, 0) = comparison_result ? value_if_true : value_if_false;

            } else {
                 // Index (i,j,k) from the iMultiFab box is outside the TIFF image dimensions
                 fab(i, j, k, 0) = value_if_false;
            }
        });
    }
}

// Original overload (output 1/0) - calls the flexible version
void TiffReader::threshold(double threshold_value, amrex::iMultiFab& mf) const
{
    threshold(threshold_value, 1, 0, mf); // Call the flexible version with 1/0
}

} // namespace OpenImpala

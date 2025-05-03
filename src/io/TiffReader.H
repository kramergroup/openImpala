#ifndef TIFF_READER_H
#define TIFF_READER_H

#include <string>
#include <cstdint>   // For standard integer types
#include <stdexcept> // For std::runtime_error
// #include <vector> // No longer needed for m_raw_bytes
// #include <map>    // Only needed if attributes are implemented

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>

// Forward declare TIFF - not strictly necessary now as TIFF* isn't in the
// public/protected interface, but can be kept for clarity if desired.
// struct tiff;
// typedef struct tiff TIFF;

namespace OpenImpala {

/**
 * @brief Reads 3D image data from single/multi-directory TIFFs or file sequences.
 * Performs parallel, chunked reading directly into AMReX iMultiFab.
 *
 * This class uses libtiff to read TIFF files (single stack files or sequences
 * representing Z-slices). It is designed for parallel environments (MPI+OpenMP)
 * and large datasets that may exceed single-node memory.
 *
 * Metadata (dimensions, data type) is read first by rank 0 and broadcast.
 * Data reading is performed by the `threshold` methods, driven by the
 * distribution of the target `iMultiFab`. Each MPI rank reads only the
 * necessary strips or tiles for its owned boxes directly from the file(s).
 * Supports Striped and Tiled TIFF formats.
 *
 * @warning Assumes serial libtiff calls internally. While reading is distributed
 * across ranks via MFIter, concurrent libtiff calls from different
 * threads within a rank to the SAME file handle might require a
 * thread-safe libtiff build or external locking (current implementation
 * uses per-thread handles for sequences, avoiding this issue there).
 * @warning Assumes PLANARCONFIG_CONTIG. Multi-sample data is read, but only
 * the first sample is used during thresholding.
 * @warning Copy operations disabled.
 */
class TiffReader
{
public:
    /**
     * @brief Default constructor. Creates an empty reader. Use readFile() or readFileSequence().
     */
    TiffReader();

    /**
     * @brief Constructs a TiffReader, reads metadata from a single (potentially multi-directory) TIFF file.
     *
     * Reads dimensions and data type info on rank 0 and broadcasts.
     * Does NOT read pixel data. Throws on error.
     * @param filename Path to the TIFF file.
     * @throws std::runtime_error on file open or metadata read errors.
     */
    explicit TiffReader(const std::string& filename);

    /**
     * @brief Constructs a TiffReader, reads metadata from the first file of a sequence.
     *
     * Reads dimensions and data type info from the first file on rank 0 and broadcasts.
     * Does NOT read pixel data. Throws on error.
     * @param base_pattern Base filename pattern (e.g., "path/slice_").
     * @param num_files Total number of files in the sequence (determines Z dimension).
     * @param start_index The starting number for the sequence (e.g., 0 or 1).
     * @param digits The number of digits used for the sequence number padding (e.g., 4 for "0001").
     * @param suffix File extension including dot (e.g., ".tif").
     * @throws std::runtime_error on file open/metadata read errors or inconsistent metadata.
     */
    TiffReader(
        const std::string& base_pattern,
        int num_files,
        int start_index = 0,
        int digits = 1,
        const std::string& suffix = ".tif");

    /**
     * @brief Virtual default destructor.
     */
    virtual ~TiffReader() = default;

    // --- Deleted Copy Operations ---
    TiffReader(const TiffReader&) = delete;
    TiffReader& operator=(const TiffReader&) = delete;

    // --- Default Move Operations ---
    TiffReader(TiffReader&&) = default;
    TiffReader& operator=(TiffReader&&) = default;

    /**
     * @brief Reads metadata from a single (potentially multi-directory) TIFF file.
     *
     * Clears existing metadata. Rank 0 opens file, reads dimensions/type, validates,
     * determines depth, and broadcasts info to all ranks. Does NOT read pixel data.
     * @param filename Path to the TIFF file.
     * @return true on success (metadata broadcast), false otherwise. Aborts on critical errors.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Reads metadata from the first file of a TIFF sequence.
     *
     * Clears existing metadata. Rank 0 opens first file, reads dimensions/type, validates,
     * sets depth from num_files, and broadcasts info (including sequence params) to all ranks.
     * Does NOT read pixel data.
     * @param base_pattern Base filename pattern (e.g., "path/slice_").
     * @param num_files Total number of files in the sequence (determines Z dimension).
     * @param start_index The starting number for the sequence (e.g., 0 or 1).
     * @param digits The number of digits used for the sequence number padding (e.g., 4 for "0001").
     * @param suffix File extension including dot (e.g., ".tif").
     * @return true on success (metadata broadcast), false otherwise. Aborts on critical errors.
     */
    bool readFileSequence(
        const std::string& base_pattern,
        int num_files,
        int start_index = 0,
        int digits = 1,
        const std::string& suffix = ".tif");

    /**
     * @brief Reads data chunk-by-chunk into an iMultiFab based on thresholding (output 1/0).
     *
     * Performs distributed reading. Iterates through the patches (`MFIter`) of the output
     * iMultiFab `mf`. For each patch, reads the corresponding strips or tiles
     * from the TIFF file(s) and fills the patch's FArrayBox based on the threshold.
     * Sets output cells to 1 if `native_value > raw_threshold`, else 0.
     * Requires metadata to have been read successfully first via readFile() or readFileSequence().
     * Handles conversion from native TIFF type to double for comparison. Uses first sample if SPP > 1.
     *
     * @param raw_threshold The threshold value (compared against data converted to double).
     * @param mf Output amrex::iMultiFab reference to fill (must be defined and distributed).
     * @throws std::runtime_error on read errors or if metadata not ready.
     */
    void threshold(double raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Reads data chunk-by-chunk into an iMultiFab based on thresholding with custom output values.
     *
     * Performs distributed reading. Iterates through the patches (`MFIter`) of the output
     * iMultiFab `mf`. For each patch, reads the corresponding strips or tiles
     * from the TIFF file(s) and fills the patch's FArrayBox based on the threshold.
     * Sets output cells to `value_if_true` if `native_value > raw_threshold`, else `value_if_false`.
     * Requires metadata to have been read successfully first via readFile() or readFileSequence().
     * Handles conversion from native TIFF type to double for comparison. Uses first sample if SPP > 1.
     *
     * @param raw_threshold The threshold value (compared against data converted to double).
     * @param value_if_true Integer value if condition is true.
     * @param value_if_false Integer value if condition is false.
     * @param mf Output amrex::iMultiFab reference to fill (must be defined and distributed).
     * @throws std::runtime_error on read errors or if metadata not ready.
     */
    void threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    // --- Metadata Getters ---
    /** @brief Returns the index space Box covering the entire image domain [0, w-1]x[0, h-1]x[0, d-1]. */
    amrex::Box box() const;

    /** @brief Get width (X-dimension) in pixels/voxels. Requires metadata to be read. */
    int width() const;
    /** @brief Get height (Y-dimension) in pixels/voxels. Requires metadata to be read. */
    int height() const;
    /** @brief Get depth (Z-dimension) in pixels/voxels. Requires metadata to be read. */
    int depth() const;

    /** @brief Get bits per sample (e.g., 8, 16, 32). Requires metadata to be read. */
    int bitsPerSample() const;
    /** @brief Get sample format (e.g., SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT, SAMPLEFORMAT_IEEEFP from tiff.h). Requires metadata to be read. */
    int sampleFormat() const;
    /** @brief Get samples per pixel (e.g., 1 for grayscale, 3 for RGB). Requires metadata to be read. */
    int samplesPerPixel() const;

    /** @brief Get TIFF FillOrder tag value (e.g., 1=MSB->LSB, 2=LSB->MSB). Requires metadata to be read. */
    uint16_t getFillOrder() const { return m_fill_order; } // <<< NEW GETTER ADDED HERE <<<

    /** @brief Check if metadata has been successfully read and broadcast. */
    bool isRead() const { return m_is_read; }


private:
    // --- Member Variables ---
    // Source Info
    std::string m_filename;       /**< Filename (single stack) */
    std::string m_base_pattern;   /**< Base pattern for sequence */
    std::string m_suffix;         /**< Suffix for sequence */
    int m_start_index = 0;        /**< Starting index for sequence */
    int m_digits = 1;             /**< Number of digits for sequence padding */
    bool m_is_sequence = false;   /**< Flag indicating if source is a sequence */

    // Domain/Metadata Info (Read by Rank 0, Broadcast to all)
    int m_width = 0;              /**< Width of the domain (X) */
    int m_height = 0;             /**< Height of the domain (Y) */
    int m_depth = 0;              /**< Depth of the domain (Z) */
    bool m_is_read = false;       /**< Flag indicating if metadata has been successfully read and broadcast */

    // TIFF Format Info (Read by Rank 0, Broadcast to all)
    uint16_t m_bits_per_sample = 0; /**< TIFFTAG_BITSPERSAMPLE */
    uint16_t m_sample_format = 0;   /**< TIFFTAG_SAMPLEFORMAT */
    uint16_t m_samples_per_pixel = 0;/**< TIFFTAG_SAMPLESPERPIXEL */
    uint16_t m_fill_order; // Initialized in constructor(s)

    // Private helper function implementing the core distributed read logic
    // Called by the public threshold methods.
    void readDistributedIntoFab(
        amrex::iMultiFab& dest_mf,
        int value_if_true,
        int value_if_false,
        double raw_threshold
    ) const;


}; // class TiffReader

} // namespace OpenImpala

#endif // TIFF_READER_H

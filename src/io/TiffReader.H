#ifndef OPENIMPALA_TIFF_READER_H
#define OPENIMPALA_TIFF_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For uint16_t, uint32_t etc.
#include <stdexcept> // For exceptions
#include <map>       // For attribute map

#include <AMReX_REAL.H>
#include <AMReX_Box.H>

// Forward declarations for AMReX types
namespace amrex {
    class iMultiFab;
}
// Forward declaration for HDF5 types (If including H5Cpp.h causes issues here)
// namespace H5 { class H5File; class DataSet; ... }
// Best practice if possible: include <H5Cpp.h> if needed by header declarations,
// otherwise only include it in the .cpp file. Here, it seems only needed in .cpp.

namespace OpenImpala {

/**
 * @brief Reads image data from TIFF files using libtiff.
 *
 * Can read single multi-directory TIFF files or a sequence of single-directory TIFF files.
 * Assumes each directory (in multi-dir file) or each file (in sequence) represents a Z-slice.
 * Attempts to decompress data and stores it internally as raw bytes.
 * Metadata (dimensions, data type) is read from the TIFF tags. Implementation relies on libtiff.
 *
 * Currently supports common scalar data types (uint, int, float) and assumes
 * SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT, or SAMPLEFORMAT_IEEEFP.
 * Assumes PlanarConfiguration = PLANARCONFIG_CONTIG for multi-sample images.
 * Handles multiple directories as Z-slices. Does not handle complex layouts like
 * sub-IFDs beyond the basic multi-directory stack. Tiled TIFFs are not supported.
 *
 * @warning This implementation reads the *entire* 3D volume into memory (`m_raw_bytes`).
 * It is unsuitable for datasets that exceed available RAM.
 * @warning Assumes serial I/O. In parallel runs, typically only Rank 0 should interact
 * with the file(s).
 * @warning Copy operations are disabled. Move operations are enabled.
 */
class TiffReader
{
public:
    // Define internal storage type for raw bytes
    using ByteType = unsigned char;

    /** @brief Default constructor. Creates an empty reader. */
    TiffReader();

    /** @brief Creates TiffReader & reads single multi-directory TIFF or first file of a sequence.
     * @param filename Path to the single multi-directory TIFF file.
     * @throws std::runtime_error If reading fails.
     */
    explicit TiffReader(const std::string& filename);

    /** @brief Virtual default destructor. */
    virtual ~TiffReader() = default;

    // --- Resource Management ---
    TiffReader(const TiffReader&) = delete; // Disable copying
    TiffReader& operator=(const TiffReader&) = delete;
    TiffReader(TiffReader&&) = default; // Enable moving
    TiffReader& operator=(TiffReader&&) = default;

    /**
     * @brief Attempts to read data from a single multi-directory TIFF file.
     * Resets state. Reads metadata and pixel data using libtiff.
     * @param filename Path to the single multi-directory TIFF file.
     * @return true if successful, false otherwise (errors printed).
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Reads a sequence of individual TIFF files as a 3D stack.
     *
     * Assumes files are named according to a base pattern and sequential numbering
     * (e.g., base_pattern + zero_padded_number + suffix).
     * Reads metadata from the first file and expects subsequent files to be consistent.
     * Resets any previously read data.
     *
     * @param base_pattern The base filename pattern (e.g., "/path/to/data/slice_").
     * @param num_files The total number of files/slices in the sequence (determines depth).
     * @param start_index The starting number for the sequence (default: 0).
     * @param digits The number of digits used for zero-padding the sequence number (default: 4).
     * @param suffix The file extension including the dot (default: ".tif").
     * @return true if the entire sequence was read successfully, false otherwise (errors printed).
     */
    bool readFileSequence(
        const std::string& base_pattern,
        int num_files,
        int start_index = 0,
        int digits = 4,
        const std::string& suffix = ".tif");

    /** @brief Checks if data has been successfully read. */
    bool isRead() const;

    // --- Metadata Getters (const) ---
    int width() const;          ///< Image width (X dimension) in pixels.
    int height() const;         ///< Image height (Y dimension) in pixels.
    int depth() const;          ///< Number of slices/directories (Z dimension).
    uint16_t bitsPerSample() const; ///< Bits per sample (e.g., 8, 16, 32).
    uint16_t sampleFormat() const;  ///< Sample format (e.g., SAMPLEFORMAT_UINT). Use libtiff constants.
    uint16_t samplesPerPixel() const; ///< Number of samples per pixel (e.g., 1 for grayscale).

    /** @brief Gets the index space (bounding box) covering the dataset. */
    amrex::Box box() const;

    /** @brief Retrieves value at (i,j,k) [sample 0], converted to double. */
    double getValue(int i, int j, int k, int sample = 0) const;

    /** @brief Thresholds image (sample 0) and fills iMultiFab (1 if > threshold, 0 otherwise). */
    void threshold(double threshold_value, amrex::iMultiFab& mf) const;

    /** @brief Thresholds image (sample 0) and fills iMultiFab (custom values). */
    void threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;


private:
    /** @brief Internal implementation using libtiff to read a single file (multi-directory). */
    bool readTiffInternal();

    /** @brief Helper to calculate bytes per voxel based on read metadata. */
    size_t getBytesPerVoxelSample() const;

    // --- Member Variables ---
    std::string m_filename;         ///< Filename (or base pattern for sequence)
    bool m_is_read = false;         ///< Flag indicating successful read

    // --- Metadata read from TIFF ---
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;
    uint16_t m_bits_per_sample = 0;
    uint16_t m_sample_format = 0;
    uint16_t m_samples_per_pixel = 0;
    // Add more metadata if needed (compression, planar config stored if logic differs)

    // Internal storage for decompressed pixel data
    std::vector<ByteType> m_raw_bytes;

    // TIFF* handle is managed within .cpp methods using RAII or careful closing.
};

} // namespace OpenImpala

#endif // OPENIMPALA_TIFF_READER_H

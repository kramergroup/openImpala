#ifndef OPENIMPALA_TIFF_READER_H
#define OPENIMPALA_TIFF_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For uint16_t, uint32_t etc.
#include <stdexcept> // For exceptions

// Forward declarations for AMReX types
namespace amrex {
    class Box;
    class iMultiFab;
}

namespace OpenImpala {

/**
 * @brief Reads image data from TIFF files using libtiff.
 *
 * This class reads image data from single or multi-page (directory) TIFF files,
 * assuming each directory represents a Z-slice. It attempts to decompress
 * the data and stores it internally as raw bytes. Metadata (dimensions,
 * data type) is read from the TIFF tags. The implementation relies on the
 * libtiff library.
 *
 * Currently supports common scalar data types (uint, int, float) and assumes
 * SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT, or SAMPLEFORMAT_IEEEFP.
 * Assumes PlanarConfiguration = PLANARCONFIG_CONTIG for multi-sample images.
 * Handles multiple directories as Z-slices. Does not handle complex layouts like
 * sub-IFDs beyond the basic multi-directory stack.
 */
class TiffReader
{
public:
    /**
     * @brief Default constructor. Creates an empty reader.
     */
    TiffReader();

    /**
     * @brief Creates a new TiffReader and attempts to read the specified file.
     *
     * Reads metadata and pixel data using libtiff. Throws std::runtime_error
     * on failure (e.g., file not found, libtiff error, unsupported format).
     *
     * @param filename Path to the TIFF file.
     * @throws std::runtime_error If reading fails.
     */
    explicit TiffReader(const std::string& filename);

    /**
     * @brief Virtual destructor. (Implementation might need to clean up libtiff resources if PIMPL isn't used).
     */
    virtual ~TiffReader() = default; // Or implement if needed for resource cleanup

    // --- Resource Management ---
    TiffReader(const TiffReader&) = delete; // Disable copying
    TiffReader& operator=(const TiffReader&) = delete;
    TiffReader(TiffReader&&) = default; // Enable moving
    TiffReader& operator=(TiffReader&&) = default;

    /**
     * @brief Attempts to read data from the specified TIFF file.
     *
     * Clears any existing data. Reads metadata and pixel data using libtiff.
     * Returns true on success, false on failure. Detailed errors printed via amrex::Print.
     *
     * @param filename Path to the TIFF file.
     * @return true if the file was read successfully, false otherwise.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Checks if data has been successfully read.
     * @return true if readFile or the constructor succeeded, false otherwise.
     */
    bool isRead() const;

    // --- Metadata Getters (const) ---
    int width() const;          ///< Image width (X dimension) in pixels.
    int height() const;         ///< Image height (Y dimension) in pixels.
    int depth() const;          ///< Number of slices/directories (Z dimension).
    uint16_t bitsPerSample() const; ///< Bits per sample (e.g., 8, 16, 32).
    uint16_t sampleFormat() const;  ///< Sample format (e.g., SAMPLEFORMAT_UINT, _INT, _IEEEFP). Use libtiff constants.
    uint16_t samplesPerPixel() const; ///< Number of samples per pixel (e.g., 1 for grayscale, 3 for RGB).

    /**
     * @brief Gets the index space (bounding box) covering the dataset.
     * @return An amrex::Box covering [0,0,0] to [width-1, height-1, depth-1],
     * or an empty box if data is not read.
     */
    amrex::Box box() const;

    /**
     * @brief Retrieves the value at a specific voxel/sample, converted to double.
     *
     * Reads the raw bytes for the voxel (i, j, k), reconstructs the value
     * based on the stored bitsPerSample and sampleFormat, handling potential
     * endianness issues (usually handled by libtiff), and converts it to double.
     * Assumes XYZ layout (k index varies slowest).
     *
     * @param i X-index (0 <= i < width).
     * @param j Y-index (0 <= j < height).
     * @param k Z-index/Slice index (0 <= k < depth).
     * @param sample The sample index (default 0). For multi-sample images.
     * @return The voxel value converted to double.
     * @throws std::out_of_range If indices are out of bounds.
     * @throws std::runtime_error If data type is unsupported or conversion fails.
     */
    double getValue(int i, int j, int k, int sample = 0) const;

    /**
     * @brief Thresholds the image data (first sample) and fills an iMultiFab.
     *
     * Iterates over the cells (i, j, k) in the iMultiFab. Retrieves the voxel
     * value for sample 0 (converted to double), compares it to `threshold_value`,
     * and sets the cell in mf to `value_if_true` (if value > threshold) or
     * `value_if_false`. Cells outside the raw data bounds are set to `value_if_false`.
     *
     * @param threshold_value The threshold value (comparison is raw_value > threshold_value).
     * @param value_if_true The integer value to set in mf if the condition is true.
     * @param value_if_false The integer value to set in mf if the condition is false.
     * @param mf The amrex::iMultiFab to fill (must have 1 component).
     */
    void threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Thresholds the image data (first sample) and fills an iMultiFab (outputting 1 or 0).
     *
     * Convenience overload for threshold(threshold_value, 1, 0, mf).
     *
     * @param threshold_value The threshold value (comparison is raw_value > threshold_value).
     * @param mf The amrex::iMultiFab to fill (must have 1 component).
     */
    void threshold(double threshold_value, amrex::iMultiFab& mf) const;


private:
    /**
     * @brief Internal implementation using libtiff to read the file.
     * @return true on success, false on failure.
     */
    bool readTiffInternal();

    /**
     * @brief Helper to calculate bytes per voxel based on read metadata.
     * @return Bytes per voxel per sample, or 0 if invalid.
    */
    size_t getBytesPerVoxelSample() const;


    // --- Member Variables ---
    std::string m_filename;         ///< Filename of the source TIFF file.
    bool m_is_read = false;         ///< Flag indicating if data was successfully read.

    // --- Metadata read from TIFF ---
    int m_width = 0;                ///< Image width (pixels)
    int m_height = 0;               ///< Image height (pixels)
    int m_depth = 0;                ///< Number of directories/slices
    uint16_t m_bits_per_sample = 0; ///< Bits per sample
    uint16_t m_sample_format = 0;   ///< SAMPLEFORMAT tag value
    uint16_t m_samples_per_pixel = 0; ///< Samples per pixel
    // Could add more metadata members if needed (e.g., compression type)

    // Store decompressed data as raw bytes. Use unsigned char for compatibility.
    using ByteType = unsigned char;
    std::vector<ByteType> m_raw_bytes; ///< Vector containing the raw, decompressed pixel data (interleaved)

    // Note: TIFF* handle is intentionally omitted here. It should be managed
    // within the .cpp file (e.g., inside readFile/readTiffInternal) or using PIMPL.
};

} // namespace OpenImpala

#endif // OPENIMPALA_TIFF_READER_H

#ifndef OPENIMPALA_RAW_READER_H
#define OPENIMPALA_RAW_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For fixed-width integers
#include <stdexcept> // For exceptions

// Forward declarations for AMReX types
namespace amrex {
    class Box;
    class iMultiFab;
    // *** ADDED for IntVect usage in .cpp ***
    class IntVect;
}

namespace OpenImpala {

/**
 * @brief Specifies the expected primitive data type and endianness within a raw binary file.
 *
 * Endianness must be specified for multi-byte types.
 */
enum class RawDataType {
    UNKNOWN,   ///< Default/invalid type
    UINT8,     ///< 8-bit unsigned integer
    INT8,      ///< 8-bit signed integer
    INT16_LE, ///< 16-bit signed integer, Little Endian (e.g., x86)
    INT16_BE, ///< 16-bit signed integer, Big Endian
    UINT16_LE,///< 16-bit unsigned integer, Little Endian
    UINT16_BE,///< 16-bit unsigned integer, Big Endian
    INT32_LE, ///< 32-bit signed integer, Little Endian
    INT32_BE, ///< 32-bit signed integer, Big Endian
    UINT32_LE,///< 32-bit unsigned integer, Little Endian
    UINT32_BE,///< 32-bit unsigned integer, Big Endian
    FLOAT32_LE,///< 32-bit float (IEEE 754), Little Endian
    FLOAT32_BE,///< 32-bit float (IEEE 754), Big Endian
    FLOAT64_LE,///< 64-bit double (IEEE 754), Little Endian
    FLOAT64_BE ///< 64-bit double (IEEE 754), Big Endian
};

/**
 * @brief A Reader for raw binary data files where metadata is provided externally.
 *
 * This class reads a raw binary file assuming a simple, contiguous layout
 * (XYZ order, Z varies slowest). The caller MUST provide the correct dimensions,
 * primitive data type, and endianness of the data within the file.
 *
 * The class stores the data as raw bytes and interprets them on demand
 * (e.g., during thresholding or value retrieval), handling potential
 * byte swapping.
 */
class RawReader
{
public: // <-- Moved ByteType here
    // Store data as raw bytes. Using unsigned char for compatibility. Use std::byte in C++17+.
    using ByteType = unsigned char;

    /**
     * @brief Default constructor. Creates an empty reader.
     */
    RawReader();

    /**
     * @brief Creates a new RawReader and attempts to read data.
     *
     * Reads the raw binary file specified by filename, assuming the provided
     * dimensions and data type. Throws std::runtime_error on failure
     * (e.g., file not found, incorrect size, read error).
     *
     * @param filename Path to the .raw file.
     * @param width Width (X dimension) of the dataset.
     * @param height Height (Y dimension) of the dataset.
     * @param depth Depth (Z dimension) of the dataset.
     * @param data_type The data type and endianness of voxels in the file.
     * @throws std::runtime_error If reading fails.
     */
    explicit RawReader(const std::string& filename,
                       int width, int height, int depth,
                       RawDataType data_type);

    /**
     * @brief Virtual default destructor.
     */
    virtual ~RawReader() = default;

    // --- Resource Management ---
    // Disable copying to prevent expensive duplication of potentially large data
    RawReader(const RawReader&) = delete;
    RawReader& operator=(const RawReader&) = delete;

    // Enable moving (transfers ownership of internal data)
    RawReader(RawReader&&) = default;
    RawReader& operator=(RawReader&&) = default;

    /**
     * @brief Attempts to read data from the specified raw file.
     *
     * Clears any existing data. Returns true on success, false on failure.
     * Detailed errors are printed via amrex::Print/Warning on failure.
     *
     * @param filename Path to the .raw file.
     * @param width Width (X dimension) of the dataset.
     * @param height Height (Y dimension) of the dataset.
     * @param depth Depth (Z dimension) of the dataset.
     * @param data_type The data type and endianness of voxels in the file.
     * @return true if the file was read successfully, false otherwise.
     */
    bool readFile(const std::string& filename,
                  int width, int height, int depth,
                  RawDataType data_type);

    /**
     * @brief Checks if data has been successfully read.
     * @return true if readFile or the constructor succeeded, false otherwise.
     */
    bool isRead() const;

    /**
     * @brief Gets the width (X dimension) of the dataset.
     * @return Width, or 0 if data not read.
     */
    int width() const;

    /**
     * @brief Gets the height (Y dimension) of the dataset.
     * @return Height, or 0 if data not read.
     */
    int height() const;

    /**
     * @brief Gets the depth (Z dimension) of the dataset.
     * @return Depth, or 0 if data not read.
     */
    int depth() const;

    /**
     * @brief Gets the index space (bounding box) covering the dataset.
     * @return An amrex::Box covering [0,0,0] to [width-1, height-1, depth-1],
     * or an empty box if data is not read.
     */
    amrex::Box box() const;

    /**
     * @brief Gets the stored data type enum.
     * @return The RawDataType specified during reading.
     */
    RawDataType getDataType() const;

    /**
     * @brief Retrieves the value at a specific voxel, converted to double.
     *
     * Reads the raw bytes for the voxel (i, j, k), reconstructs the value
     * based on the stored data type (including byte swapping if necessary),
     * and converts it to double.
     * Assumes XYZ layout (k index varies slowest).
     *
     * @param i X-index (0 <= i < width).
     * @param j Y-index (0 <= j < height).
     * @param k Z-index (0 <= k < depth).
     * @return The voxel value converted to double.
     * @throws std::out_of_range If indices are out of bounds.
     * @throws std::runtime_error If data type is unknown or conversion fails.
     */
    double getValue(int i, int j, int k) const;

    /**
     * @brief Thresholds the image data and fills an iMultiFab.
     *
     * Iterates over the cells in the iMultiFab's valid boxes. For each cell (i, j, k)
     * within the bounds of the raw data, it retrieves the voxel value (converted
     * to double), compares it to `threshold_value`, and sets the corresponding
     * cell in the iMultiFab to `value_if_true` (if value > threshold) or
     * `value_if_false` otherwise. Cells outside the raw data bounds are set
     * to `value_if_false`.
     *
     * @param threshold_value The threshold value (comparison is raw_value > threshold_value).
     * @param value_if_true The integer value to set in mf if the condition is true.
     * @param value_if_false The integer value to set in mf if the condition is false.
     * @param mf The amrex::iMultiFab to fill (must have 1 component).
     */
    void threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Thresholds the image data and fills an iMultiFab (outputting 1 or 0).
     *
     * Convenience overload for threshold(threshold_value, 1, 0, mf).
     *
     * @param threshold_value The threshold value (comparison is raw_value > threshold_value).
     * @param mf The amrex::iMultiFab to fill (must have 1 component).
     */
    void threshold(double threshold_value, amrex::iMultiFab& mf) const;


private:
    /**
     * @brief Internal implementation to read the raw file.
     * @return true on success, false on failure.
     */
    bool readRawFileInternal();

    /**
     * @brief Calculates the size in bytes of a single voxel based on m_data_type.
     * @return Size in bytes, or 0 for UNKNOWN type.
     */
    size_t getBytesPerVoxel() const;

    /**
     * @brief Checks the endianness of the host machine.
     * @return true if the host is little-endian, false otherwise.
     */
    bool isHostLittleEndian() const;


    // --- Member Variables ---
    std::string m_filename;         ///< Filename of the source raw file.
    int m_width = 0;                ///< Width (X) of the domain.
    int m_height = 0;               ///< Height (Y) of the domain.
    int m_depth = 0;                ///< Depth (Z) of the domain.
    RawDataType m_data_type = RawDataType::UNKNOWN; ///< Type/endianness of data in file.

    // ByteType using alias moved to public section
    std::vector<ByteType> m_raw_bytes; ///< Vector containing the raw byte data.

    bool m_is_read = false;         ///< Flag indicating if data was successfully read.
};

} // namespace OpenImpala

#endif // OPENIMPALA_RAW_READER_H

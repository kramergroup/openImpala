#ifndef OPENIMPALA_RAW_READER_H
#define OPENIMPALA_RAW_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For fixed-width integers (e.g., uint8_t, int16_t)
#include <stdexcept> // For standard exception classes (e.g., std::runtime_error)

// AMReX Includes
// Include necessary AMReX headers directly instead of relying on forward declarations
// where feasible, as class definitions might change between versions (e.g., IntVect).
#include <AMReX_Box.H>       // Defines amrex::Box for representing index space
#include <AMReX_iMultiFab.H> // Defines amrex::iMultiFab for integer multifabs
#include <AMReX_IntVect.H>   // Defines amrex::IntVect for integer vectors/indices

// Forward declarations are generally discouraged for AMReX types unless strictly needed
// to manage complex include dependencies, as their definitions can change.
// The necessary headers are included above.

namespace OpenImpala {

/**
 * @brief Specifies the expected primitive data type and endianness of voxels
 * within a raw binary data file.
 *
 * @details Endianness (Little Endian - LE, Big Endian - BE) must be specified
 * for multi-byte data types (16-bit, 32-bit, 64-bit) to ensure
 * correct interpretation on different machine architectures.
 */
enum class RawDataType {
    UNKNOWN,    ///< Default, indicating an uninitialized or invalid type.
    UINT8,      ///< 8-bit unsigned integer (char, byte). Endianness irrelevant.
    INT8,       ///< 8-bit signed integer (signed char). Endianness irrelevant.
    INT16_LE,   ///< 16-bit signed integer, Little Endian (least significant byte first, common on x86).
    INT16_BE,   ///< 16-bit signed integer, Big Endian (most significant byte first).
    UINT16_LE,  ///< 16-bit unsigned integer, Little Endian.
    UINT16_BE,  ///< 16-bit unsigned integer, Big Endian.
    INT32_LE,   ///< 32-bit signed integer, Little Endian.
    INT32_BE,   ///< 32-bit signed integer, Big Endian.
    UINT32_LE,  ///< 32-bit unsigned integer, Little Endian.
    UINT32_BE,  ///< 32-bit unsigned integer, Big Endian.
    FLOAT32_LE, ///< 32-bit single-precision float (IEEE 754), Little Endian.
    FLOAT32_BE, ///< 32-bit single-precision float (IEEE 754), Big Endian.
    FLOAT64_LE, ///< 64-bit double-precision float (IEEE 754), Little Endian.
    FLOAT64_BE  ///< 64-bit double-precision float (IEEE 754), Big Endian.
};

/**
 * @brief Reads and provides access to data from raw binary files where metadata
 * (dimensions, data type, endianness) is provided externally.
 *
 * @details This class handles reading a flat binary file containing voxel data.
 * It assumes a simple, contiguous layout corresponding to standard C/Fortran
 * XYZ array ordering (where the Z index varies slowest). The caller is
 * *required* to supply the correct dimensions (width, height, depth) and
 * the primitive data type (including endianness for multi-byte types)
 * present in the file via the constructor or the `readFile` method.
 *
 * The entire file content is read into an internal byte buffer (`m_raw_bytes`).
 * Voxel values are interpreted and reconstructed from this byte buffer
 * on demand (e.g., when `getValue` or `threshold` is called), handling
 * potential byte swapping between the file's endianness and the host
 * machine's endianness if they differ.
 */
class RawReader
{
public:
    /**
     * @brief Type alias for the underlying byte storage.
     * @details Using `unsigned char` for byte representation. `std::byte` could be
     * used if C++17 is guaranteed and preferred.
     */
    using ByteType = unsigned char;

    /**
     * @brief Default constructor. Creates an empty reader instance.
     * @details The reader will be in an invalid state (`isRead()` returns false)
     * until `readFile` is successfully called.
     */
    RawReader();

    /**
     * @brief Constructs a RawReader and immediately attempts to read the specified file.
     *
     * @details Reads the raw binary file specified by `filename`, interpreting the
     * data according to the provided `width`, `height`, `depth`, and `data_type`.
     * Throws a `std::runtime_error` if any part of the reading process fails
     * (e.g., file cannot be opened, file size doesn't match expected dimensions
     * and data type, or a read error occurs).
     *
     * @param filename Path to the input .raw binary file.
     * @param width Width (X dimension) of the dataset in voxels. Must be > 0.
     * @param height Height (Y dimension) of the dataset in voxels. Must be > 0.
     * @param depth Depth (Z dimension) of the dataset in voxels. Must be > 0.
     * @param data_type The data type and endianness of the voxels stored in the file.
     * Must not be `RawDataType::UNKNOWN`.
     * @throws std::runtime_error If reading or validation fails.
     */
    explicit RawReader(const std::string& filename,
                       int width, int height, int depth,
                       RawDataType data_type);

    /**
     * @brief Virtual default destructor.
     * @details Ensures proper cleanup if RawReader is used as a base class.
     */
    virtual ~RawReader() = default;

    // --- Resource Management (Rule of 5/3) ---
    // Prevent copying: Copying potentially large raw data buffers is expensive
    // and usually undesirable. Deleted copy constructor and assignment operator.
    RawReader(const RawReader&) = delete;
    RawReader& operator=(const RawReader&) = delete;

    // Allow moving: Efficiently transfer ownership of the internal data buffer
    // from one RawReader instance to another. Default move constructor and
    // assignment operator are suitable here.
    RawReader(RawReader&&) = default;
    RawReader& operator=(RawReader&&) = default;

    /**
     * @brief Reads or re-reads data from the specified raw binary file.
     *
     * @details Clears any previously loaded data. Opens the specified file, verifies
     * its size against the expected size based on dimensions and data type,
     * and reads the entire content into the internal byte buffer.
     * Prints detailed error messages using `amrex::Print` or `amrex::Warning`
     * upon failure.
     *
     * @param filename Path to the input .raw binary file.
     * @param width Width (X dimension) of the dataset in voxels. Must be > 0.
     * @param height Height (Y dimension) of the dataset in voxels. Must be > 0.
     * @param depth Depth (Z dimension) of the dataset in voxels. Must be > 0.
     * @param data_type The data type and endianness of the voxels stored in the file.
     * Must not be `RawDataType::UNKNOWN`.
     * @return `true` if the file was read and validated successfully, `false` otherwise.
     */
    bool readFile(const std::string& filename,
                  int width, int height, int depth,
                  RawDataType data_type);

    /**
     * @brief Checks if data has been successfully read and the reader is ready.
     * @return `true` if the constructor or `readFile` succeeded, `false` otherwise.
     */
    bool isRead() const;

    /**
     * @brief Gets the width (X dimension) of the loaded dataset.
     * @return Width in voxels, or 0 if data has not been successfully read.
     */
    int width() const;

    /**
     * @brief Gets the height (Y dimension) of the loaded dataset.
     * @return Height in voxels, or 0 if data has not been successfully read.
     */
    int height() const;

    /**
     * @brief Gets the depth (Z dimension) of the loaded dataset.
     * @return Depth in voxels, or 0 if data has not been successfully read.
     */
    int depth() const;

    /**
     * @brief Gets the index space (bounding box) corresponding to the dataset dimensions.
     * @details The box covers the 0-based index range [0, 0, 0] to
     * [width-1, height-1, depth-1].
     * @return An `amrex::Box` representing the dataset's index space. Returns an
     * empty (invalid) box if data has not been successfully read.
     */
    amrex::Box box() const;

    /**
     * @brief Gets the data type and endianness specified when the file was read.
     * @return The `RawDataType` enum value.
     */
    RawDataType getDataType() const;

    /**
     * @brief Retrieves the numerical value at a specific voxel, converting it to `double`.
     *
     * @details Accesses the raw bytes corresponding to the voxel at logical coordinates
     * (i, j, k), reconstructs the original numerical value based on the
     * `m_data_type` (performing byte swapping if the file's endianness differs
     * from the host machine's), and then casts the result to `double`.
     * Assumes standard XYZ layout where the k index (depth) varies slowest.
     *
     * @param i X-index (must be 0 <= i < width).
     * @param j Y-index (must be 0 <= j < height).
     * @param k Z-index (must be 0 <= k < depth).
     * @return The voxel's numerical value converted to `double`.
     * @throws std::out_of_range If the provided indices (i, j, k) are outside the
     * valid dataset dimensions [0..width-1, 0..height-1, 0..depth-1].
     * @throws std::runtime_error If the reader is not in a valid state (`isRead()` is false),
     * if the stored `m_data_type` is `UNKNOWN`, or if an internal error occurs
     * during value reconstruction.
     */
    double getValue(int i, int j, int k) const;

    /**
     * @brief Fills an AMReX iMultiFab based on thresholding the raw data.
     *
     * @details Iterates through the cells defined by the valid boxes of the input `mf`.
     * For each cell (i, j, k) that falls within the bounds of the loaded raw
     * dataset [0..width-1, 0..height-1, 0..depth-1], this function retrieves
     * the corresponding voxel value using `getValue(i, j, k)`. It then compares
     * this value against `threshold_value`.
     * - If `getValue(i, j, k) > threshold_value`, the cell in `mf` is set to `value_if_true`.
     * - Otherwise (including if the cell is outside the raw data bounds), the cell
     * in `mf` is set to `value_if_false`.
     *
     * @param threshold_value The value to compare against. The comparison is strictly greater than (>).
     * @param value_if_true The integer value to write to `mf` for cells exceeding the threshold.
     * @param value_if_false The integer value to write to `mf` for cells not exceeding the threshold
     * or outside the bounds of the raw data.
     * @param[out] mf The `amrex::iMultiFab` to be filled. It must be defined (allocated) and
     * is expected to have exactly one component (`nComp() == 1`). The operation
     * is performed in parallel using OpenMP if available.
     * @throws std::runtime_error If the reader is not in a valid state (`isRead()` is false) or
     * if `mf.nComp() != 1`.
     */
    void threshold(double threshold_value, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Fills an AMReX iMultiFab based on thresholding the raw data (outputting 1 or 0).
     *
     * @details This is a convenience overload that calls the main `threshold` method with
     * `value_if_true = 1` and `value_if_false = 0`. It effectively creates a binary
     * mask in the `iMultiFab` where cells corresponding to raw data values greater
     * than `threshold_value` are set to 1, and all others are set to 0.
     *
     * @param threshold_value The value to compare against. The comparison is strictly greater than (>).
     * @param[out] mf The `amrex::iMultiFab` to be filled. It must be defined (allocated) and
     * is expected to have exactly one component (`nComp() == 1`).
     * @throws std::runtime_error If the reader is not in a valid state (`isRead()` is false) or
     * if `mf.nComp() != 1`.
     */
    void threshold(double threshold_value, amrex::iMultiFab& mf) const;


private:
    /**
     * @brief Internal helper function that performs the actual file reading and validation.
     * @details Called by the constructor and `readFile`. Handles file opening, size checking,
     * reading bytes into `m_raw_bytes`, and setting `m_is_read`.
     * @return `true` on success, `false` on failure (errors printed internally).
     */
    bool readRawFileInternal();

    /**
     * @brief Calculates the size in bytes of a single voxel based on the stored `m_data_type`.
     * @return The number of bytes per voxel (e.g., 1 for UINT8, 2 for INT16, 4 for FLOAT32, 8 for FLOAT64).
     * Returns 0 if `m_data_type` is `UNKNOWN`.
     */
    size_t getBytesPerVoxel() const;

    /**
     * @brief Determines the endianness of the host machine at runtime.
     * @details Used to decide if byte swapping is needed when reading multi-byte data types.
     * @return `true` if the host machine is little-endian, `false` if it's big-endian.
     */
    bool isHostLittleEndian() const;


    // --- Member Variables ---
    std::string m_filename;       ///< Stores the path to the source raw file provided to the constructor or `readFile`.
    int m_width = 0;              ///< Width (X dimension) of the dataset in voxels, set by constructor or `readFile`.
    int m_height = 0;             ///< Height (Y dimension) of the dataset in voxels, set by constructor or `readFile`.
    int m_depth = 0;              ///< Depth (Z dimension) of the dataset in voxels, set by constructor or `readFile`.
    RawDataType m_data_type = RawDataType::UNKNOWN; ///< Data type (incl. endianness) specified for the file's content.

    // Internal buffer holding the raw byte data read from the file.
    std::vector<ByteType> m_raw_bytes; ///< Contiguous storage for all voxel data as raw bytes.

    bool m_is_read = false;       ///< Status flag indicating if data was successfully read and validated.
};

} // namespace OpenImpala

#endif // OPENIMPALA_RAW_READER_H

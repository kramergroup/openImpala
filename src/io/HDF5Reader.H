#ifndef HDF5_READER_H // Corrected header guard style
#define HDF5_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For standard integer types if needed elsewhere
#include <stdexcept> // For std::runtime_error
#include <map>       // For attribute map

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>

// --- HDF5 C++ API Include ---
// Ensure HDF5 C++ bindings are available and linked.
#include <H5Cpp.h>

namespace OpenImpala {

/**
 * @brief Reads 3D image/scalar field data from a specified dataset within an HDF5 file.
 *
 * This class uses the HDF5 C++ API to open an HDF5 file, access a specific
 * 3D dataset, read its dimensions and data, and provide methods for access
 * and thresholding into AMReX iMultiFabs.
 *
 * It assumes the target dataset is 3-dimensional. It attempts to read the dataset's
 * native datatype and converts supported types (e.g., integers, float) to double
 * for internal storage.
 *
 * @warning This implementation reads the *entire* 3D HDF5 dataset specified into
 * memory (`m_raw`). It is unsuitable for datasets that exceed available RAM.
 *
 * @warning Assumes serial HDF5 I/O. In parallel runs, typically only Rank 0
 * should interact with the file, or parallel HDF5 features would be needed.
 *
 * @warning Copy operations are disabled due to potentially large data size. Move
 * operations can be enabled if needed.
 */
class HDF5Reader
{
public:
    // <<< FIX: Define the public DataType alias >>>
    // IMPORTANT: Verify this matches the actual dataset type in your HDF5 file!
    // Common types for segmented data: std::uint8_t, std::uint16_t, std::int32_t
    using DataType = std::uint16_t; // Example: Assuming 16-bit unsigned integer labels

    // Define the internal storage type. Converting to double provides flexibility
    // but uses more memory and might lose precision for large integer types.
    // Consider changing this to match DataType if appropriate.
    using InternalDataType = double;

    /**
     * @brief Default constructor. Creates an empty reader. Use readFile() later.
     */
    HDF5Reader();

    /**
     * @brief Constructs an HDF5Reader, opens the file, and reads the specified dataset.
     *
     * Reads dataset dimensions and voxel data. Queries dataset datatype and attempts
     * conversion to InternalDataType (double).
     * Uses HDF5 C++ exceptions internally, rethrowing as std::runtime_error on failure.
     *
     * @param filename Path to the HDF5 file.
     * @param hdf5dataset Full path to the 3D dataset within the HDF5 file (e.g., "/group/data").
     * @throws std::runtime_error on HDF5 errors (file not found, dataset not found, read error, unsupported type/rank, etc.).
     */
    explicit HDF5Reader(const std::string& filename, const std::string& hdf5dataset);

    /**
     * @brief Virtual default destructor.
     * (HDF5 C++ API handles resource closing via RAII).
     */
    virtual ~HDF5Reader() = default;

    // --- Deleted Copy Operations ---
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;

    // --- Optional: Default Move Operations (C++11) ---
    HDF5Reader(HDF5Reader&&) = default;
    HDF5Reader& operator=(HDF5Reader&&) = default;

    /**
     * @brief Reads data from the specified HDF5 file and dataset into this object.
     *
     * Clears any existing data. Opens file, finds dataset, reads dimensions/metadata,
     * checks datatype, reads data (converting to InternalDataType), and closes resources.
     * Uses HDF5 C++ exceptions internally for error handling.
     *
     * @param filename Path to the HDF5 file.
     * @param hdf5dataset Full path to the 3D dataset within the HDF5 file.
     * @return true if successful, false otherwise. Prints errors via amrex::Print/Warning.
     * @note The constructor already calls this logic; use this method primarily
     * after default construction or to reuse the object.
     */
    bool readFile(const std::string& filename, const std::string& hdf5dataset);

    /**
     * @brief Fills an iMultiFab based on thresholding the raw data.
     *
     * Sets output cells to 1 if `getRawValue(i,j,k) > raw_threshold`, else 0.
     * Assumes mf is defined on a box compatible with the dataset dimensions.
     *
     * @param raw_threshold The threshold value (compared against internal double data).
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     */
    void threshold(double raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Fills an iMultiFab based on thresholding with custom output values.
     *
     * Sets output cells to `value_if_true` if `getRawValue(i,j,k) > raw_threshold`, else `value_if_false`.
     * Assumes mf is defined on a box compatible with the dataset dimensions.
     *
     * @param raw_threshold The threshold value (compared against internal double data).
     * @param value_if_true The integer value to set if condition is true.
     * @param value_if_false The integer value to set if condition is false.
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     */
    void threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Returns the index space Box covering the entire image domain [0, w-1] x [0, h-1] x [0, d-1].
     * @return amrex::Box representing the full image dimensions. Returns empty box if data not read.
     */
    amrex::Box box() const;

    /** @brief Get width (X-dimension) of the dataset in pixels/voxels. */
    int width() const;

    /** @brief Get height (Y-dimension) of the dataset in pixels/voxels. */
    int height() const;

    /** @brief Get depth (Z-dimension) of the dataset in pixels/voxels. */
    int depth() const;

    /**
     * @brief Get the raw data value (converted to double) at a specific coordinate.
     * Performs bounds checking. Throws std::out_of_range if indices are invalid or data not read.
     * @param i X-coordinate (0 to width-1).
     * @param j Y-coordinate (0 to height-1).
     * @param k Z-coordinate (0 to depth-1).
     * @return The raw data value as double at (i,j,k).
     * @throws std::out_of_range if indices are out of bounds or data not read.
     */
    InternalDataType getRawValue(int i, int j, int k) const;

    /**
     * @brief Get read-only access to the entire raw data vector (stored as doubles).
     * @return Constant reference to the internal raw data vector. Returns empty vector if not read.
     */
    const std::vector<InternalDataType>& getRawData() const;

    /**
     * @brief Reads a specific string attribute from the HDF5 dataset.
     * @param attr_name The name of the attribute to read.
     * @return The attribute value as a string.
     * @throws std::runtime_error if attribute not found or other HDF5 error occurs.
     */
    std::string getAttribute(const std::string& attr_name) const;

    /**
     * @brief Reads all attributes from the HDF5 dataset.
     * Attempts to read attribute values as strings.
     * @return A map of attribute names to string values. Returns empty map if no attributes or error.
     */
    std::map<std::string, std::string> getAllAttributes() const;

    /** @brief Check if data has been successfully read. */
    bool isRead() const { return m_is_read; }

    // Consider adding metadata methods if needed by tests/application:
    // H5::DataType getNativeDataType() const; // Example returning HDF5 type object
    // size_t getDataTypeSize() const; // Example returning size in bytes


private:
    /**
     * @brief Internal implementation for reading the HDF5 file and dataset.
     * Called by constructor/readFile. Uses HDF5 C++ API and exceptions.
     * Handles datatype query/conversion to InternalDataType.
     * @return true on success (sets members), false on failure (prints errors).
     */
    bool readHDF5FileInternal();

    // --- Member Variables ---
    std::string m_filename;                   /**< Filename of the source HDF5 file */
    std::string m_hdf5dataset;                /**< Path to the dataset within the HDF5 file */
    std::vector<InternalDataType> m_raw;    /**< Vector containing the raw data (converted to double) */
    int m_width = 0;                          /**< Width of the domain (X) */
    int m_height = 0;                         /**< Height of the domain (Y) */
    int m_depth = 0;                          /**< Depth of the domain (Z) */
    bool m_is_read = false;                   /**< Flag indicating if data has been successfully read */

    // Store HDF5 IDs if needed across methods, though C++ API often avoids this.
    // If C API were used, you'd store H5F_t file_id, H5D_t dset_id etc. and close in destructor.
    // With C++ API, objects like H5::H5File manage their own lifetime via RAII.

}; // class HDF5Reader

} // namespace OpenImpala

#endif // HDF5_READER_H

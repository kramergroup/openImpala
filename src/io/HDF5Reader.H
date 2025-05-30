#ifndef HDF5_READER_H
#define HDF5_READER_H

#include <string>
#include <vector> // Still needed for other things potentially
#include <map>
#include <stdexcept> // For std::runtime_error

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H> // Now central

// --- HDF5 C++ API Include ---
#include <H5Cpp.h>

namespace OpenImpala {

/**
 * @brief Reads 3D image/scalar field data from a specified dataset within an HDF5 file
 * directly into AMReX MultiFabs, suitable for large datasets.
 *
 * This class uses the HDF5 C++ API to open an HDF5 file, access a specific
 * 3D dataset, read its dimensions, and provide methods for reading data
 * chunk-by-chunk (via AMReX MFIter) into an iMultiFab while applying a threshold.
 *
 * It assumes the target dataset is 3-dimensional. It detects the dataset's
 * native datatype and performs thresholding based on that type.
 *
 * @warning Assumes serial HDF5 I/O within the reading methods. In parallel runs,
 * all ranks will open the file, but MFIter ensures each rank only reads
 * data needed for its owned patches. For true parallel I/O performance,
 * parallel HDF5 features would be needed.
 */
class HDF5Reader
{
public:
    /**
     * @brief Default constructor. Creates an empty reader. Use readFile() later.
     */
    HDF5Reader();

    /**
     * @brief Constructs an HDF5Reader, opens the file, and reads dataset metadata.
     *
     * Reads dataset dimensions and detects native datatype.
     * Does NOT read the full dataset into memory.
     * Uses HDF5 C++ exceptions internally, rethrowing as std::runtime_error on failure.
     *
     * @param filename Path to the HDF5 file.
     * @param hdf5dataset Full path to the 3D dataset within the HDF5 file (e.g., "/group/data").
     * @throws std::runtime_error on HDF5 errors (file not found, dataset not found, invalid rank, etc.).
     */
    explicit HDF5Reader(const std::string& filename, const std::string& hdf5dataset);

    /**
     * @brief Virtual default destructor.
     */
    virtual ~HDF5Reader() = default;

    // --- Deleted Copy Operations ---
    HDF5Reader(const HDF5Reader&) = delete;
    HDF5Reader& operator=(const HDF5Reader&) = delete;

    // --- Optional: Default Move Operations (C++11) ---
    HDF5Reader(HDF5Reader&&) = default;
    HDF5Reader& operator=(HDF5Reader&&) = default;

    /**
     * @brief Reads metadata (dimensions, datatype) from the specified HDF5 file and dataset.
     *
     * Clears any existing metadata. Opens file, finds dataset, reads dimensions/metadata.
     * Does NOT read voxel data.
     * Uses HDF5 C++ exceptions internally for error handling.
     *
     * @param filename Path to the HDF5 file.
     * @param hdf5dataset Full path to the 3D dataset within the HDF5 file.
     * @return true if successful, false otherwise. Prints errors via amrex::Print/Warning.
     */
    bool readFile(const std::string& filename, const std::string& hdf5dataset);

    /**
     * @brief Reads data chunk-by-chunk into an iMultiFab based on thresholding.
     *
     * Iterates through the patches of the output iMultiFab `mf`. For each patch,
     * reads the corresponding hyperslab from the HDF5 dataset and fills the
     * patch's FArrayBox based on the threshold comparison.
     * Sets output cells to 1 if `native_value > raw_threshold`, else 0.
     * Assumes mf is defined on a box compatible with the dataset dimensions read previously.
     * Handles conversion from native HDF5 type to double for comparison.
     *
     * @param raw_threshold The threshold value (compared against data converted to double).
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     * @throws std::runtime_error on HDF5 read errors or if metadata not read.
     */
    void threshold(double raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Reads data chunk-by-chunk into an iMultiFab based on thresholding with custom output values.
     *
     * Iterates through the patches of the output iMultiFab `mf`. For each patch,
     * reads the corresponding hyperslab from the HDF5 dataset and fills the
     * patch's FArrayBox based on the threshold comparison.
     * Sets output cells to `value_if_true` if `native_value > raw_threshold`, else `value_if_false`.
     * Assumes mf is defined on a box compatible with the dataset dimensions read previously.
     * Handles conversion from native HDF5 type to double for comparison.
     *
     * @param raw_threshold The threshold value (compared against data converted to double).
     * @param value_if_true The integer value to set if condition is true.
     * @param value_if_false The integer value to set if condition is false.
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     * @throws std::runtime_error on HDF5 read errors or if metadata not read.
     */
    void threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Returns the index space Box covering the entire image domain [0, w-1] x [0, h-1] x [0, d-1].
     * @return amrex::Box representing the full image dimensions. Returns empty box if metadata not read.
     */
    amrex::Box box() const;

    /** @brief Get width (X-dimension) of the dataset in pixels/voxels. */
    int width() const;

    /** @brief Get height (Y-dimension) of the dataset in pixels/voxels. */
    int height() const;

    /** @brief Get depth (Z-dimension) of the dataset in pixels/voxels. */
    int depth() const;

    /**
     * @brief Reads a specific string attribute from the HDF5 dataset.
     * @param attr_name The name of the attribute to read.
     * @return The attribute value as a string. Returns error string if fails.
     */
    std::string getAttribute(const std::string& attr_name) const;

    /**
     * @brief Reads all attributes from the HDF5 dataset.
     * Attempts to read attribute values as strings.
     * @return A map of attribute names to string values. Returns empty map if no attributes or error.
     */
    std::map<std::string, std::string> getAllAttributes() const;

    /** @brief Check if metadata has been successfully read. */
    bool isRead() const { return m_is_read; }

    /** @brief Get the native HDF5 data type detected in the dataset. */
    H5::DataType getNativeDataType() const { return m_native_type; }


private:
    /**
     * @brief Internal implementation for reading HDF5 metadata only.
     * Called by constructor/readFile. Uses HDF5 C++ API and exceptions.
     * Stores dimensions and native data type.
     * @return true on success (sets members), false on failure (prints errors).
     */
    bool readMetadataInternal();

    // Template function to read a hyperslab and apply thresholding
    // Needed because the native type read from file varies.
    template<typename T_Native>
    void readAndThresholdFab(H5::DataSet& dataset, double raw_threshold,
                             int value_if_true, int value_if_false,
                             const amrex::Box& box, amrex::IArrayBox& fab) const;

    // --- Member Variables ---
    std::string m_filename;         /**< Filename of the source HDF5 file */
    std::string m_hdf5dataset;      /**< Path to the dataset within the HDF5 file */
    int m_width = 0;                /**< Width of the domain (X) */
    int m_height = 0;               /**< Height of the domain (Y) */
    int m_depth = 0;                /**< Depth of the domain (Z) */
    bool m_is_read = false;         /**< Flag indicating if *metadata* has been successfully read */
    H5::DataType m_native_type;     /**< Detected native HDF5 data type of the dataset */

}; // class HDF5Reader

} // namespace OpenImpala

#endif // HDF5_READER_H

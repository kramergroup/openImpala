#ifndef DAT_READER_H // Corrected header guard style
#define DAT_READER_H

#include <cstddef> // <--- ADD THIS LINE
#include <vector>
#include <string>
#include <cstdint> // Use <cstdint> for fixed-width types
#include <stdexcept> // For exceptions in constructor

#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Box.H>

namespace OpenImpala {

/**
 * @brief Reads and provides access to raw image data from specific DAT files.
 *
 * This class reads a binary DAT file (assumed format from Heidelberg context,
 * details need verification), stores the raw voxel data, and provides methods
 * to query dimensions and generate thresholded AMReX iMultiFabs.
 *
 * @warning The exact DAT file format (data type, endianness, header presence/format,
 * data layout) is assumed and needs verification. This implementation assumes a
 * specific raw binary layout. Endianness might need handling in readFile().
 *
 * @warning Copying is disabled due to potentially large raw data size.
 */
class DatReader
{
public:
    // --- Define the expected raw data type ---
    // IMPORTANT: Verify this matches the actual DAT file format!
    // Common types: std::uint8_t, std::uint16_t, std::uint32_t, float, double
    using DataType = std::uint16_t; // Example: Assuming 16-bit unsigned integer data

    /**
     * @brief Default constructor. Creates an empty reader. Use readFile() later.
     */
    DatReader();

    /**
     * @brief Constructs a DatReader and reads data from the specified filename.
     *
     * Reads the DAT file header (if any, assumed structure) to get dimensions
     * and then reads the raw voxel data.
     * Throws std::runtime_error if the file cannot be opened or read correctly.
     *
     * @param filename Path to the DAT file containing the image data.
     * @throws std::runtime_error on file open or read errors.
     */
    explicit DatReader(const std::string& filename);

    /**
     * @brief Virtual default destructor.
     */
    virtual ~DatReader() = default;

    // --- Deleted Copy Operations ---
    // Prevent copying due to potentially large m_raw vector.
    DatReader(const DatReader&) = delete;
    DatReader& operator=(const DatReader&) = delete;

    // --- Optional: Default Move Operations (C++11) ---
    // DatReader(DatReader&&) = default;
    // DatReader& operator=(DatReader&&) = default;

    /**
     * @brief Reads data from the specified filename into this object.
     *
     * Clears any existing data before reading. Reads header (if any) and raw data.
     * Use this after default construction or to reuse the object for another file.
     *
     * @param filename Path to the DAT file.
     * @return true if the file was read successfully, false otherwise.
     * Prints error messages using amrex::Print on failure.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Checks if the reader has successfully read metadata and data.
     * @return true if data is read and the reader is operational, false otherwise.
     */
    bool isRead() const;

    /**
     * @brief Fills an iMultiFab based on thresholding the raw data.
     *
     * Iterates through the cells of the iMultiFab 'mf'. For each cell (i,j,k),
     * if the corresponding raw data value is greater than 'raw_threshold',
     * sets mf(i,j,k) = 1, otherwise sets it to 0.
     *
     * @warning Caller must ensure mf's BoxArray is compatible with the dimensions
     * read from the DAT file (available via box(), width(), etc.). Internal
     * checks might be added in implementation for safety.
     *
     * @param raw_threshold The threshold value (in raw data units).
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     */
    void threshold(DataType raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Fills an iMultiFab based on thresholding the raw data with custom output values.
     *
     * Iterates through the cells of the iMultiFab 'mf'. For each cell (i,j,k),
     * if the corresponding raw data value is greater than 'raw_threshold',
     * sets mf(i,j,k) = value_if_true, otherwise sets it to value_if_false.
     *
     * @warning Caller must ensure mf's BoxArray is compatible with the dimensions.
     *
     * @param raw_threshold The threshold value (in raw data units).
     * @param value_if_true The integer value to set if raw_data[idx] > raw_threshold.
     * @param value_if_false The integer value to set otherwise.
     * @param mf Output amrex::iMultiFab reference to fill (must be defined).
     */
    void threshold(DataType raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /**
     * @brief Returns the index space Box covering the entire image domain.
     * The Box typically starts at (0,0,0).
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
     * @brief Get the raw data value at a specific coordinate.
     * Performs bounds checking. Throws std::out_of_range if indices are invalid.
     * @param i X-coordinate (0 to width-1).
     * @param j Y-coordinate (0 to height-1).
     * @param k Z-coordinate (0 to depth-1).
     * @return The raw data value at (i,j,k).
     * @throws std::out_of_range if indices are out of bounds.
     */
    DataType getRawValue(int i, int j, int k) const;

    /**
     * @brief Get read-only access to the entire raw data vector.
     * Provides direct access but exposes internal storage layout (XYZ order assumed).
     * @return Constant reference to the internal raw data vector.
     */
    const std::vector<DataType>& getRawData() const;


private:
    /**
     * @brief Internal implementation for reading the DAT file. Called by constructor/readFile.
     * @return true on success, false on failure. Prints errors internally.
     */
    bool readDatFileInternal();

    // --- Member Variables ---
    std::string m_filename;             /**< Filename of the source Dat file */
    std::vector<DataType> m_raw;        /**< Vector containing the raw data (verify DataType!) */
    int m_width = 0;                    /**< Width of the domain (X) */
    int m_height = 0;                   /**< Height of the domain (Y) */
    int m_depth = 0;                    /**< Depth of the domain (Z) */
    bool m_is_read = false;             /**< Flag indicating if data has been successfully read */

}; // class DatReader

} // namespace OpenImpala

#endif // DAT_READER_H

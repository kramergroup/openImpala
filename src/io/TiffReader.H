#ifndef TIFF_READER_H // Corrected header guard style
#define TIFF_READER_H

#include <vector>
#include <string>
#include <cstdint>   // For standard integer types
#include <stdexcept> // For std::runtime_error
#include <map>       // For attribute map
#include <memory>    // For std::unique_ptr (used internally in .cpp)

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>

// Forward declare TIFF to avoid including tiffio.h in the header if possible
// If TIFF* is used directly in public/protected interface, tiffio.h might be needed here
// struct tiff; // Opaque struct forward declaration
// typedef struct tiff TIFF; // Typedef for TIFF*
// For simplicity now, assume tiffio.h is only needed in .cpp

namespace OpenImpala {

/**
 * @brief Reads 3D image data from single or multi-directory/multi-file TIFFs.
 *
 * This class uses libtiff to read TIFF files, including sequences or multi-directory
 * files representing a 3D volume. It handles various sample formats and bits per sample,
 * storing the data internally as raw bytes. Methods are provided for thresholding
 * into AMReX iMultiFabs and querying metadata.
 *
 * @warning Only supports CONTIGUOUS planar configuration. Tiled TIFFs not supported.
 * @warning Reads entire dataset into memory - unsuitable for datasets exceeding RAM.
 * @warning Assumes serial I/O. Parallel reads require different strategies.
 * @warning Copy operations disabled.
 */
class TiffReader
{
public:
    // Define the type used for internal byte storage
    using ByteType = unsigned char;

    /**
     * @brief Default constructor. Creates an empty reader.
     */
    TiffReader();

    /**
     * @brief Constructs a TiffReader and reads a single (potentially multi-directory) TIFF file.
     * @param filename Path to the TIFF file.
     * @throws std::runtime_error on file open or read errors, or unsupported format.
     */
    explicit TiffReader(const std::string& filename);

    /**
     * @brief Constructs a TiffReader and reads a sequence of TIFF files.
     * @param base_pattern Base filename pattern (e.g., "path/slice_").
     * @param num_files Total number of files in the sequence (determines Z dimension).
     * @param start_index The starting number for the sequence (e.g., 0 or 1).
     * @param digits The number of digits used for the sequence number padding (e.g., 4 for "0001").
     * @param suffix File extension including dot (e.g., ".tif").
     * @throws std::runtime_error on file open/read errors, inconsistent metadata, or unsupported format.
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

    // --- Optional: Default Move Operations ---
    TiffReader(TiffReader&&) = default;
    TiffReader& operator=(TiffReader&&) = default;

    /**
     * @brief Reads a single (potentially multi-directory) TIFF file. Clears existing data.
     * @param filename Path to the TIFF file.
     * @return true on success, false otherwise. Prints errors via amrex::Print.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Reads a sequence of TIFF files representing Z-slices. Clears existing data.
     * @param base_pattern Base filename pattern (e.g., "path/slice_").
     * @param num_files Total number of files in the sequence (determines Z dimension).
     * @param start_index The starting number for the sequence (e.g., 0 or 1).
     * @param digits The number of digits used for the sequence number padding (e.g., 4 for "0001").
     * @param suffix File extension including dot (e.g., ".tif").
     * @return true on success, false otherwise. Prints errors via amrex::Print.
     */
    bool readFileSequence(
        const std::string& base_pattern,
        int num_files,
        int start_index = 0,
        int digits = 1,
        const std::string& suffix = ".tif");

    /**
     * @brief Fills an iMultiFab based on thresholding the raw data. Output 1/0.
     * @param raw_threshold Threshold value (compared against data converted to double).
     * @param mf Output amrex::iMultiFab reference.
     */
    void threshold(double raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Fills an iMultiFab based on thresholding with custom output values.
     * @param raw_threshold Threshold value (compared against data converted to double).
     * @param value_if_true Integer value if condition is true.
     * @param value_if_false Integer value if condition is false.
     * @param mf Output amrex::iMultiFab reference.
     */
    void threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    /** @brief Returns the index space Box covering the entire image domain [0, w-1]x[0, h-1]x[0, d-1]. */
    amrex::Box box() const;

    /** @brief Get width (X-dimension) in pixels/voxels. */
    int width() const;
    /** @brief Get height (Y-dimension) in pixels/voxels. */
    int height() const;
    /** @brief Get depth (Z-dimension) in pixels/voxels. */
    int depth() const;

    // --- Metadata Getters ---
    /** @brief Get bits per sample (e.g., 8, 16, 32). */
    int bitsPerSample() const;
    /** @brief Get sample format (e.g., SAMPLEFORMAT_UINT, SAMPLEFORMAT_INT, SAMPLEFORMAT_IEEEFP). */
    int sampleFormat() const;
    /** @brief Get samples per pixel (e.g., 1 for grayscale, 3 for RGB). */
    int samplesPerPixel() const;

    /**
     * @brief Get the raw data value interpreted as type T at a specific coordinate.
     * Handles conversion based on stored metadata (bitsPerSample, sampleFormat).
     * @warning Assumes T matches the data type or is convertible. Only reads first sample if SPP > 1.
     * @param i X-coordinate (0 to width-1).
     * @param j Y-coordinate (0 to height-1).
     * @param k Z-coordinate (0 to depth-1).
     * @return The raw data value interpreted as type T.
     * @throws std::out_of_range if indices are out of bounds or data not read.
     * @throws std::runtime_error if requested type T doesn't match data size.
     */
    // <<< FIX: Changed declaration to template >>>
    template <typename T>
    T getValue(int i, int j, int k) const;

    /** @brief Get read-only access to the raw byte data vector. */
    const std::vector<ByteType>& getRawData() const;

    /** @brief Check if data has been successfully read. */
    bool isRead() const { return m_is_read; }


private:
    /** @brief Internal implementation for reading ONE directory of the currently open TIFF. */
    bool readTiffInternal(); // Assumes m_filename and tif handle are set by caller

    // --- Member Variables ---
    std::string m_filename;             /**< Filename (or pattern) of the source */
    std::vector<ByteType> m_raw_bytes;  /**< Vector containing the raw byte data */
    int m_width = 0;                    /**< Width of the domain (X) */
    int m_height = 0;                   /**< Height of the domain (Y) */
    int m_depth = 0;                    /**< Depth of the domain (Z) */
    bool m_is_read = false;             /**< Flag indicating if data has been successfully read */

    // Metadata read from TIFF
    uint16_t m_bits_per_sample = 0;     /**< TIFFTAG_BITSPERSAMPLE */
    uint16_t m_sample_format = 0;       /**< TIFFTAG_SAMPLEFORMAT */
    uint16_t m_samples_per_pixel = 0;   /**< TIFFTAG_SAMPLESPERPIXEL */

}; // class TiffReader

} // namespace OpenImpala

#endif // TIFF_READER_H

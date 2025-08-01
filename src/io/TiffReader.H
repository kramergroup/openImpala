#ifndef TIFF_READER_H
#define TIFF_READER_H

#include <string>
#include <cstdint>
#include <stdexcept>

#include <AMReX_REAL.H>
#include <AMReX_Box.H>
#include <AMReX_iMultiFab.H>

namespace OpenImpala {

/**
 * @brief Reads 3D image data from single/multi-directory TIFFs or file sequences.
 * Performs parallel, chunked reading directly into AMReX iMultiFab.
 */
class TiffReader
{
public:
    /**
     * @brief Default constructor. Creates an empty reader.
     */
    TiffReader();

    /**
     * @brief Constructs a TiffReader and reads metadata from a single TIFF file.
     * @param filename Path to the TIFF file.
     */
    explicit TiffReader(const std::string& filename);

    /**
     * @brief Constructs a TiffReader and reads metadata from a TIFF sequence.
     * @param base_pattern Base filename pattern (e.g., "path/slice_").
     * @param num_files Total number of files in the sequence.
     * @param start_index The starting number for the sequence.
     * @param digits The number of digits for padding (e.g., 4 for "0001").
     * @param suffix File extension (e.g., ".tif").
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
     * @brief Reads metadata from a single TIFF file.
     * @param filename Path to the TIFF file.
     * @return true on success.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Reads metadata from the first file of a TIFF sequence.
     * @return true on success.
     */
    bool readFileSequence(
        const std::string& base_pattern,
        int num_files,
        int start_index = 0,
        int digits = 1,
        const std::string& suffix = ".tif");

    /**
     * @brief Reads data into an iMultiFab, segmenting to 1/0 based on a threshold.
     * @param raw_threshold The threshold value.
     * @param mf Output amrex::iMultiFab to fill.
     */
    void threshold(double raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Reads data into an iMultiFab, segmenting to custom values based on a threshold.
     * @param raw_threshold The threshold value.
     * @param value_if_true Integer value if condition is true.
     * @param value_if_false Integer value if condition is false.
     * @param mf Output amrex::iMultiFab to fill.
     */
    void threshold(double raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    // --- NEW METHOD FOR MULTI-PHASE ---
    /**
     * @brief Reads raw voxel values directly as phase IDs.
     *
     * This method is for reading pre-segmented, multi-phase images where
     * the integer value of each voxel already corresponds to a phase ID.
     * The raw data value is read, rounded to the nearest integer, and
     * stored directly in the destination iMultiFab.
     *
     * @param mf Output amrex::iMultiFab reference to fill with phase IDs.
     */
    void readPhaseIDs(amrex::iMultiFab& mf) const;

    // --- Metadata Getters ---
    amrex::Box box() const;
    int width() const;
    int height() const;
    int depth() const;
    int bitsPerSample() const;
    int sampleFormat() const;
    int samplesPerPixel() const;
    uint16_t getFillOrder() const { return m_fill_order; }
    bool isRead() const { return m_is_read; }

private:
    // --- Member Variables ---
    std::string m_filename;
    std::string m_base_pattern;
    std::string m_suffix;
    int m_start_index = 0;
    int m_digits = 1;
    bool m_is_sequence = false;
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;
    bool m_is_read = false;
    uint16_t m_bits_per_sample = 0;
    uint16_t m_sample_format = 0;
    uint16_t m_samples_per_pixel = 0;
    uint16_t m_fill_order;

    // Private helper for thresholding
    void readDistributedIntoFab(
        amrex::iMultiFab& dest_mf,
        int value_if_true,
        int value_if_false,
        double raw_threshold
    ) const;
    
    // --- NEW PRIVATE HELPER FOR MULTI-PHASE ---
    // Note: The implementation in the .cpp file duplicates the logic for simplicity.
    // A more advanced refactoring could merge them, but this header is correct
    // for the implementation we discussed.

}; // class TiffReader

} // namespace OpenImpala

#endif // TIFF_READER_H

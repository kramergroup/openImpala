#ifndef DAT_READER_H
#define DAT_READER_H

#include <vector>
#include <string>
#include <cstdint>
#include <stdexcept>

#include <AMReX_REAL.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_Box.H>

namespace OpenImpala {

/**
 * @brief Reads and provides access to raw image data from specific DAT files.
 */
class DatReader
{
public:
    // --- Define the expected raw data type ---
    using DataType = std::uint16_t;

    /**
     * @brief Default constructor. Creates an empty reader.
     */
    DatReader();

    /**
     * @brief Constructs a DatReader and reads data from the specified filename.
     * @param filename Path to the DAT file.
     */
    explicit DatReader(const std::string& filename);

    /**
     * @brief Virtual default destructor.
     */
    virtual ~DatReader() = default;

    // --- Deleted Copy Operations ---
    DatReader(const DatReader&) = delete;
    DatReader& operator=(const DatReader&) = delete;

    /**
     * @brief Reads data from the specified filename into this object.
     * @param filename Path to the DAT file.
     * @return true if the file was read successfully, false otherwise.
     */
    bool readFile(const std::string& filename);

    /**
     * @brief Checks if the reader has successfully read data.
     * @return true if data is read, false otherwise.
     */
    bool isRead() const;

    /**
     * @brief Fills an iMultiFab based on thresholding (output 1/0).
     * @param raw_threshold The threshold value.
     * @param mf Output amrex::iMultiFab to fill.
     */
    void threshold(DataType raw_threshold, amrex::iMultiFab& mf) const;

    /**
     * @brief Fills an iMultiFab based on thresholding with custom output values.
     * @param raw_threshold The threshold value.
     * @param value_if_true The integer value if condition is true.
     * @param value_if_false The integer value if condition is false.
     * @param mf Output amrex::iMultiFab to fill.
     */
    void threshold(DataType raw_threshold, int value_if_true, int value_if_false, amrex::iMultiFab& mf) const;

    // --- NEW METHOD FOR MULTI-PHASE ---
    /**
     * @brief Reads raw voxel values directly as phase IDs.
     *
     * This method is for reading pre-segmented, multi-phase images where
     * the integer value of each voxel already corresponds to a phase ID.
     * It copies the raw data value directly into the destination iMultiFab.
     *
     * @param mf Output amrex::iMultiFab reference to fill with phase IDs.
     */
    void readPhaseIDs(amrex::iMultiFab& mf) const;

    // --- Metadata Getters ---
    amrex::Box box() const;
    int width() const;
    int height() const;
    int depth() const;
    DataType getRawValue(int i, int j, int k) const;
    const std::vector<DataType>& getRawData() const;

private:
    // --- Member Variables ---
    std::string m_filename;
    std::vector<DataType> m_raw;
    int m_width = 0;
    int m_height = 0;
    int m_depth = 0;
    bool m_is_read = false;

}; // class DatReader

} // namespace OpenImpala

#endif // DAT_READER_H

#ifndef _RAWREADER_H
#define _RAWREADER_H

#include <vector>
#include <string>
#include <inttypes.h>

#include <AMReX_iMultiFab.H>

/** A Reader of Raw files
 *
 * This class provides a reader of .raw files. This class can be used to
 * create IArrayBox structures filled with thresholding data.
 *
 */

class RawReader
{
  public:

    /** Creates a new RawReader and reads data from filename
     *
     *  This default constructor reads the .raw file and stores its data
     *  in private member variables.
     *
     *  @param filename name of the .raw file containing the data
     */
    RawReader(const std::string& filename);


    /** Thresholds the image
     *
     * This function fills an IArrayBox
     * with 1 for voxels with a larger value than threshold
     * and 0 otherwise. The function takes a iMultiFab by reference.
     * It is the responsibility of the caller to ensure that the dimension
     * of the iMultiFab do not exceed to dimensions of the raw file.
     *
     * @param threshold the threshold value
     * @param mf amrex::iMultiFab reference for holding the data.
     *
     * @return IArrayBox with threshold values
     */
    void threshold(const uint32_t threshold, amrex::iMultiFab& mf);

    /** Index space covering the image
     *
     * @return an amrex:Box object with the extend of the image
     */
    amrex::Box box();

    /** Get width of the dataset */
    uint32_t width();

    /** Get height of the dataset */
    uint32_t height();

    /** Get depth of the dataset */
    uint32_t depth();

  private:

    void readRawFile();

    /*! Filename of the source raw file */
    std::string m_filename;

    /*! Vector<int> containing the raw data */
    std::vector<uint32_t> m_raw; 

    /*! Dimensions of the domain */
    int m_width, m_height, m_depth;

};

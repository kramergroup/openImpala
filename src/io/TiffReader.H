#ifndef _TIFFREADER_H
#define _TIFFREADER_H

#include <vector>
#include <string>
#include <inttypes.h>

#include <AMReX_iMultiFab.H>

/** A Reader of Tiff slices 
 *
 * This class provides a reader of TIFF images. This class can be used to 
 * create IArrayBox structures filled with thresholding data.
 *  
 */

class TiffReader 
{
  public:

    /** Creates a new TiffReader and reads data from filename
     *  
     *  This default constructor reads the TIFF file and stores its data 
     *  in private member variables. 
     * 
     *  @param filename name of the TIFF file containing the data
     */
    TiffReader(const std::string& filename);


    /** Thresholds the image 
     * 
     * This function fills an IArrayBox 
     * with 1 for voxels with a larger value than threshold 
     * and 0 otherwise. The function takes a iMultiFab by reference.
     * It is the responsibility of the caller to ensure that the dimension
     * of the iMultiFab do not exceed to dimensions of the TIFF image.
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
  
    void readTiffFile();

    /*! Filename of the source TIFF file */
    std::string m_filename;

    /*! Vector<int> containing the raw data */
    std::vector<uint32_t> m_raw; 
    
    /*! Dimensions of the domain */
    int m_width, m_height, m_depth; 

};

#endif // TIFFREADER_H
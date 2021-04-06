#include "HDF5Reader.H"
#include <h5cpp/hdf5.hpp>
#include <boost/filesystem.hpp>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>

using namespace hdf5;
using namespace boost::filesystem;

HDF5Reader::HDF5Reader( std::string const& filename) : m_filename(filename)
{
  readHDF5File();
}

void HDF5Reader::readHDF5File() 
{
  //open hdf5 file
  path file_path(m_filename);
  file::File f1 = file::open(file_path);
   
    uint32_t w, h;
    size_t npixels;
    uint32_t* raster;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_height);

    uint16_t* data;
    m_depth = 0;
    do 
    {
      npixels = m_width * m_height;
      raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32_t));
      if (raster != NULL) {
          if (TIFFReadRGBAImage(tif, m_width, m_height, raster, 0)) {
              for (long i=0; i<m_height*m_width; ++i) 
              { 
                m_raw.push_back(-raster[i]-1);
              }
          }
          _TIFFfree(raster);
      }
      m_depth++;
    } while (TIFFReadDirectory(tif));

}

uint32_t HDF5Reader::depth()
{
  return m_depth;
}

uint32_t HDF5Reader::height()
{
  return m_height;
}

uint32_t HDF5Reader::width()
{
  return m_width;
}

amrex::Box HDF5Reader::box() 
{
  amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{m_width-1,m_height-1,m_depth-1});
  return box;
}

void HDF5Reader::threshold(const uint32_t threshold, amrex::iMultiFab& mf)
{

  for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) // Loop over grids
  {
    const amrex::Box& box = mfi.validbox();
    amrex::IArrayBox& fab = mf[mfi];
    
    size_t idx;
    // Iterate over all cells in Box and threshold
    for (amrex::BoxIterator bit(box); bit.ok(); ++bit) 
    {
      idx = bit()[0] + bit()[1]*m_width + bit()[2]*m_height*m_width;
      // bit() returns IntVect
      fab(bit(),0) = (m_raw[idx] < threshold);
    } 
  }

}

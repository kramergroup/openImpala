#include "TiffStackReader.H"
#include <tiffio.h>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>

#include <sstream>
#include <iomanip>

TiffStackReader::TiffStackReader( std::string const& filename, const int tiffstack) : m_filename(filename),
                                                                                      m_tiffstack(tiffstack)
{
  readTiffFile();
}

void TiffStackReader::readTiffFile()
{
  TIFFSetWarningHandler(NULL);
  for (int k=0; k<m_tiffstack; ++k)
  {
  std::string name = m_filename;
  std::stringstream ss;
  ss << std::setw(4) << std::setfill('0') << k;
  std::string image_number = ss.str();

  name += image_number;
  name += ".tif";


  std::cout << name << std::endl;

  TIFF* tif = TIFFOpen(name.c_str(), "r");
  if (tif)
  {

    uint32_t w, h;
    size_t npixels;
    uint32_t* raster;

    TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &m_width);
    TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &m_height);
    
    // Add ghost cells to initial slice
    if (k==0) { 
       for (long j=0; j<m_height; ++j) {
          for (long i=0; i<m_width; ++i) {
              m_raw.push_back(0);
           }
        }
    }

    uint16_t* data;
    m_depth = 0;
    do
    {
      npixels = m_width * m_height;
      raster = (uint32*) _TIFFmalloc(npixels * sizeof (uint32_t));
      if (raster != NULL) {
          if (TIFFReadRGBAImageOriented(tif, m_width, m_height, raster, ORIENTATION_TOPLEFT, 0)) {
              for (long j=0; j<m_height; ++j)
              {
                for (long i=0; i<m_width; ++i)
                 {
                  if (i==0)
                  {
                    m_raw.push_back(0);
                  }
                   m_raw.push_back(-raster[i+j*m_width]-1);
                }
              }
          }
          _TIFFfree(raster);
      }

    } while (TIFFReadDirectory(tif));

    m_depth=k;
    
    // Add ghost cells to final slice
    if (k==m_tiffstack-1) { 
       for (long j=0; j<m_height; ++j) {
          for (long i=0; i<m_width; ++i) {
              m_raw.push_back(0);
           }
        }
    }

    TIFFClose(tif);
  }
   
}

}

uint32_t TiffStackReader::depth()
{
  return m_depth;
}

uint32_t TiffStackReader::height()
{
  return m_height;
}

uint32_t TiffStackReader::width()
{
  return m_width;
}

amrex::Box TiffStackReader::box()
{
  amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{m_width+1,m_height-1,m_depth+2});
  return box;
}

void TiffStackReader::threshold(const uint32_t threshold, amrex::iMultiFab& mf)
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

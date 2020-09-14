#include "DatReader.H"
#include <fstream>

#include <AMReX.H>
#include <AMReX_IntVect.H>
#include <AMReX_iMultiFab.H>

DatReader::DatReader( std::string const& filename) : m_filename(filename)
{
  readDatFile();
}

void DatReader::readDatFile()
{
  std::ifstream file(m_filename.c_str());
  if (file.is_open()) {
    std::string line;
    int counter=0;
    // Read one line at a time into the variable line:
    while(std::getline(file, line))
    {
        std::stringstream  lineStream(line);

        uint32_t value;
        // Read an integer at a time from the line
        while(lineStream >> value)
        {
          if (counter ==0){
            m_width=value;
          }
          else if (counter ==1){
            m_height=value;
          }
          else if (counter ==2){
            m_depth=value;
          }
          else
          {
            // Add the integers from a line to a 1D array (vector)
            m_raw.push_back(value);
          }
          counter++;
        }

    }
    file.close();
}

}

uint32_t DatReader::depth()
{
  return m_depth;
}

uint32_t DatReader::height()
{
  return m_height;
}

uint32_t DatReader::width()
{
  return m_width;
}

amrex::Box DatReader::box()
{
  amrex::Box box(amrex::IntVect{0,0,0}, amrex::IntVect{m_width-1,m_height-1,m_depth-1});
  return box;
}

void DatReader::threshold(const uint32_t threshold, amrex::iMultiFab& mf)
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

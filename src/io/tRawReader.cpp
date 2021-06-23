#include "RawReader.H"
#include <iostream>
#include <vector>
#include <stdlib.h>
#include <ctime>
#include <chrono>
#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_IArrayBox.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_MultiFab.H>
#include <AMReX_BoxArray.H>
#include <AMReX_CoordSys.H>
#include <AMReX_DistributionMapping.H>

#include <AMReX_PlotFileUtil.H>

/** Test RawReader class
 *
 * This test test the RawReader class. It requires the SAMPLE_RAW_FILENAME file
 * as test data.
 *
 * The test will open the sample raw file, read its data, and then assert the
 * following conditions:
 *
 *  1) Image width, height, and depth
 *
 */

#define SAMPLE_RAW_FILENAME "/openImpala/data/SampleData_2Phase.raw"
#define SAMPLE_THRESHOLD_FILENAME "/openImpala/data/SampleData_Threshold"
#define BOX_SIZE 32

int main (int argc, char* argv[])
{

  amrex::Initialize(argc, argv);
  {
  // What time is it now?
  std::time_t strt_time;
  std::tm* timeinfo;
  char datetime [80];
  
  std::time(&strt_time);
  timeinfo = std::localtime(&strt_time);
             
  std::strftime(datetime,80,"%Y%m%d%H%M",timeinfo);

  // Parameters
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
  std::cout << AMREX_SPACEDIM << "D test" << std::endl;

  // Reading the tiff file
  std::cout << "tRawReader - Reading file " << SAMPLE_RAW_FILENAME << std::endl;
  RawReader reader(SAMPLE_RAW_FILENAME);

  std::cout << "Image dimensions: " << reader.width() << "x" << reader.height() << "x" << reader.depth() << std::endl;
  assert(reader.width() == 100);
  assert(reader.height() == 100);
  assert(reader.depth() == 100);
  std::cout << "Image dimensions as expected." << std::endl;

  // Thresholding the data

  // We only have one component holding the data
  int ncomp = 1;

  amrex::Geometry geom;
  {
    amrex::RealBox rb({-1.0,-1.0,-1.0}, {1.0,1.0,1.0}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
    amrex::Geometry::Setup(&rb, 0, is_periodic.data());
    geom.define(reader.box());
  }

  amrex::BoxArray ba(geom.Domain());
  ba.maxSize(BOX_SIZE);
  std::cout << ba;
  amrex::DistributionMapping dm(ba);
  amrex::iMultiFab mf(ba,dm,1,0);

  reader.threshold(1,mf);

  std::cout << "Maximum threshold value: " << mf.max(0,0,false) << std::endl;
  std::cout << "Minumum threshold value: " << mf.min(0,0,false) << std::endl;
  assert( mf.max(0,0,false) == 1);
  assert( mf.min(0,0,false) == 0);
  std::cout << "Threshold value range as expected." << std::endl;

  // Write the data to disk
  // Note: The writing routines only work with float-valued MultiFab.
  //       Therefore, we first copy over the threshold values into a
  //       MultiFab and then write the results.
  amrex::MultiFab mfv(ba,dm,1,0);
  for (amrex::MFIter mfi(mf); mfi.isValid(); ++mfi) // Loop over grids
  {
    const amrex::Box& box = mfi.validbox();
    const amrex::IArrayBox& fab1 = mf[mfi];
    amrex::FArrayBox& fab2 = mfv[mfi];

    // Iterate over all cells in Box and threshold
    for (amrex::BoxIterator bit(box); bit.ok(); ++bit)
    {
      // bit() returns IntVect
      fab2(bit(),0) = fab1(bit(),0);
    }
  }
  // Get the users home directory to write plot file to right place
  const char* homeDir = getenv("HOME");
  // Write plot file to home dir with datetime in YYmmDDHHMM format appended
  amrex::WriteSingleLevelPlotfile(homeDir + std::string("/openimpalaresults/rawreadertest") += datetime, mfv, {"phase"}, geom, 0.0, 0);

  } // Ensure amrex related destructors have been called before tearing down the whole thing
  // by putting everything in curly brackets.
  amrex::Finalize();
  
}

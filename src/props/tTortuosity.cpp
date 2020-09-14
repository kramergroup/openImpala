/** Test VolumeFractions class
 *
 * This test tests the VolumeFraction class. It requires the SAMPLE_TIFF_FILENAME file
 * as test data.
 *
 * The test will open the sample tiff file, read its data, and then asset the
 * following conditions:
 *
 *  1) volume fractions of the two phases
 *
 */

#include "../io/TiffReader.H"
#include "TortuosityHypre.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>

#include <sstream>

#define SAMPLE_TIFF_FILENAME "../../data/SampleData_3Phase.tif"
#define BOX_SIZE 32
#define EPS 1e-10

#define DIRECTION Direction::X

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
  // What time is it now?  We'll use this to compute total run time.
  amrex::Real strt_time = amrex::second();

  // Parameters
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
  amrex::Print() << AMREX_SPACEDIM << "D test" << std::endl;

  // Define physical geometry, index space, and multifab
  amrex::Geometry geom;
  amrex::BoxArray ba;
  amrex::DistributionMapping dm;
  amrex::iMultiFab mf_phase;

  {
    // Reading the tiff file
    // The TiffReader potentially holds significant data in memory (the full voxel set).
    // The code is not parallelised, potentially creating a large memory burden per node.
    // It's best to let the reader go out of scope as soon as it is not needed anymore
    // to free up memory before further computations.
    amrex::Print() << "tTiffReader - Reading file " << SAMPLE_TIFF_FILENAME << std::endl;
    TiffReader reader(SAMPLE_TIFF_FILENAME);

    const amrex::Box bx = reader.box();
    amrex::Real fx = 1.0*bx.size()[0]/bx.size()[DIRECTION];
    amrex::Real fy = 1.0*bx.size()[1]/bx.size()[DIRECTION];
    amrex::Real fz = 1.0*bx.size()[2]/bx.size()[DIRECTION];
    amrex::RealBox rb({-1.0*fx,-1.0*fy,-1.0*fz}, {1.0*fx,1.0*fy,1.0*fz}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
    geom.define(bx, &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);

    dm.define(ba);
    mf_phase.define(ba,dm,1,1);

    // Threshold image data
    reader.threshold(1,mf_phase);

    // We have used a fab with one ghost cell to allow for stencil-type operations
    // over the fab. This requires to distribute the ghost cells
    mf_phase.FillBoundary(geom.periodicity());
  }

  // Compute tortuosity
  TortuosityHypre tortuosity(geom,ba,dm,mf_phase,0,DIRECTION,TortuosityHypre::SolverType::FlexGMRES);

  amrex::Real tau_value = tortuosity.value();
  amrex::Print() << "Tortuosity value: " << tau_value << std::endl;

  // Call the timer again and compute the maximum difference between the start time and stop time
  //   over all processors
  amrex::Real stop_time = amrex::second() - strt_time;
  const int IOProc = amrex::ParallelDescriptor::IOProcessorNumber();
  amrex::ParallelDescriptor::ReduceRealMax(stop_time,IOProc);

  // Tell the I/O Processor to write out the "run time"
  amrex::Print() << "Run time = " << stop_time << std::endl;

  } // Ensure amrex related destructors have been called before tearing down the whole thing
    // by putting everything in curly brackets.
  amrex::Finalize();
}

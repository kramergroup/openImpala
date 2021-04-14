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
#include "VolumeFraction.H"

#include <AMReX.H>
#include <AMReX_Array.H>
#include <AMReX_Geometry.H>
#include <AMReX_BoxArray.H>
#include <AMReX_DistributionMapping.H>
#include <AMReX_iMultiFab.H>

#include <sstream>

#define SAMPLE_TIFF_FILENAME "/openImpala/data/SampleData_2Phase.tif"
#define BOX_SIZE 32
#define EPS 1e-10

int main (int argc, char* argv[])
{
  amrex::Initialize(argc, argv);
  {
  // Parameters
  amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
  std::cout << AMREX_SPACEDIM << "D test" << std::endl;

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
    std::cout << "tTiffReader - Reading file " << SAMPLE_TIFF_FILENAME << std::endl;
    TiffReader reader(SAMPLE_TIFF_FILENAME);

    amrex::RealBox rb({-1.0,-1.0,-1.0}, {1.0,1.0,1.0}); // physical domain
    amrex::Array<int,AMREX_SPACEDIM> is_periodic{false, false, false};
    geom.define(reader.box(), &rb, 0, is_periodic.data());

    // Define computational domain and index space
    ba.define(geom.Domain());
    ba.maxSize(BOX_SIZE);

    dm.define(ba);
    mf_phase.define(ba,dm,1,0);

    // Threshold image data
    reader.threshold(1,mf_phase);
  }

  // Compute Volume Fractions
  amrex::Real vf_value_0;
  amrex::Real vf_value_1;

  {
    VolumeFraction vf(mf_phase, 0);
    vf_value_0 = vf.value();
  }

  {
    VolumeFraction vf(mf_phase, 1);
    vf_value_1 = vf.value();
  }

  std::cout << "Volume Fraction of phase 0: " << vf_value_0 << std::endl;
  std::cout << "Volume Fraction of phase 1: " << vf_value_1 << std::endl;

  assert( vf_value_0 >= 0.0 );
  assert( vf_value_0 <= 1.0 );
  assert( vf_value_1 >= 0.0 );
  assert( vf_value_1 <= 1.0 );

  assert( abs(1.0-vf_value_0-vf_value_1) < EPS );

  } // Ensure amrex related destructors have been called before tearing down the whole thing
    // by putting everything in curly brackets.
  amrex::Finalize();

}
